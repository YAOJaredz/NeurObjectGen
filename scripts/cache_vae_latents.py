"""Cache FLUX VAE latents for all stimuli in spatial format.

Writes:
  cache/rust_vae_latents.pt  — (300, 16, 28, 28)  float32, grayscale-luminance encoded
  cache/hvm_vae_latents.pt   — (450, 16, 28, 28)  float32
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_const import (
    N_STIMULI,
    HVM_N_STIMULI,
    RUST_NOFIXATION_DIR,
    HVM_NOFIXATION_DIR,
    RUST_VAE_LATENTS_PATH,
    HVM_VAE_LATENTS_PATH,
)


FLUX_VAE_ID = "black-forest-labs/FLUX.1-dev"


def load_vae(device: str) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(FLUX_VAE_ID, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def encode_pil(vae: AutoencoderKL, pil_image: Image.Image, device: str) -> torch.Tensor:
    """Encode a PIL image to a spatial VAE latent (16, H//8, W//8).

    Applies grayscale-luminance replication to match the Rust/HVM stimulus
    distribution (achromatic objects on gray background).
    """
    x = T.ToTensor()(pil_image.convert("RGB"))
    gray = 0.2989 * x[0:1] + 0.5870 * x[1:2] + 0.1140 * x[2:3]
    x = gray.expand(3, -1, -1).contiguous()
    x = (x * 2 - 1).unsqueeze(0).to(device)
    with torch.no_grad():
        latent = vae.encode(x).latent_dist.sample()
        latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor
    return latent.squeeze(0).cpu()  # (16, H//8, W//8)


def cache_rust(vae: AutoencoderKL, device: str, force: bool) -> None:
    if RUST_VAE_LATENTS_PATH.exists() and not force:
        print(f"{RUST_VAE_LATENTS_PATH} exists; skipping RUST")
        return

    latents = []
    for i in tqdm(range(N_STIMULI), desc="RUST VAE encode"):
        img_path = RUST_NOFIXATION_DIR / f"{i:04d}.png"
        pil = Image.open(img_path)
        latents.append(encode_pil(vae, pil, device))

    out = torch.stack(latents)  # (300, 16, H//8, W//8)
    torch.save(out, RUST_VAE_LATENTS_PATH)
    print(f"saved {tuple(out.shape)} -> {RUST_VAE_LATENTS_PATH}")


def cache_hvm(vae: AutoencoderKL, device: str, force: bool) -> None:
    if HVM_VAE_LATENTS_PATH.exists() and not force:
        print(f"{HVM_VAE_LATENTS_PATH} exists; skipping HVM")
        return

    img_paths = sorted(HVM_NOFIXATION_DIR.rglob("*.png"))
    assert len(img_paths) == HVM_N_STIMULI, (
        f"Expected {HVM_N_STIMULI} HVM images, found {len(img_paths)}"
    )

    latents = []
    for img_path in tqdm(img_paths, desc="HVM VAE encode"):
        pil = Image.open(img_path)
        latents.append(encode_pil(vae, pil, device))

    out = torch.stack(latents)  # (450, 16, H//8, W//8)
    torch.save(out, HVM_VAE_LATENTS_PATH)
    print(f"saved {tuple(out.shape)} -> {HVM_VAE_LATENTS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dataset", choices=["rust", "hvm", "both"], default="rust")
    args = parser.parse_args()

    RUST_VAE_LATENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    vae = load_vae(args.device)

    if args.dataset in ("rust", "both"):
        cache_rust(vae, args.device, args.force)
    if args.dataset in ("hvm", "both"):
        cache_hvm(vae, args.device, args.force)


if __name__ == "__main__":
    main()
