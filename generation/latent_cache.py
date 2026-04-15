"""FLUX VAE latent cache: encode all stimuli once and reuse across training runs.

Packed latents match the layout expected by generate_img2img — (1, L, C_packed)
where L = (image_size/16)^2 * 4.  The cache is stored flat as (N, L*C_packed)
and split/indexed by callers using the same seeded permutation as make_rust_loader.

Also provides build_pixel_targets() for a lightweight grayscale-pixel target
(no VAE needed) used by the low-dim ridge baseline.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from config_const import CACHE_DIR, N_STIMULI, RUST_STIM_DIR
from generation.flux_instantx import encode_image, load_pipeline


def build_pixel_targets(image_size: int) -> torch.Tensor:
    """Load all stimuli as grayscale images resized to image_size x image_size.

    Returns:
        (N_STIMULI, image_size * image_size) float32 tensor, values in [0, 1].
    """
    pixels = []
    for idx in range(N_STIMULI):
        pil = Image.open(RUST_STIM_DIR / f"{idx:04d}.png").convert("L")
        pil = pil.resize((image_size, image_size), Image.LANCZOS)
        pixels.append(TF.to_tensor(pil).flatten())  # (H*W,)
    flat = torch.stack(pixels, dim=0)  # (N, H*W)
    print(f"Built pixel targets: {tuple(flat.shape)}")
    return flat


def latent_cache_path(image_size: int) -> Path:
    return CACHE_DIR / f"spatial_latents_{image_size}.pt"


def build_latent_cache(image_size: int, device: str) -> torch.Tensor:
    """VAE-encode all stimuli and write the packed latent cache to disk.

    Args:
        image_size: Square pixel resolution (must match inference size).
        device:     Device for the FLUX VAE encoder.

    Returns:
        (N_STIMULI, L*C_packed) float32 CPU tensor.
    """
    print("Loading FLUX pipeline for VAE encoding...")
    pipe, _ = load_pipeline(device=device, default_scale=0.0)

    latents = []
    for idx in range(N_STIMULI):
        pil = Image.open(RUST_STIM_DIR / f"{idx:04d}.png").convert("RGB")
        latent = encode_image(pipe, pil.resize((image_size, image_size), Image.LANCZOS))
        h = 2 * (image_size // (pipe.vae_scale_factor * 2))
        w = 2 * (image_size // (pipe.vae_scale_factor * 2))
        C = latent.shape[1]
        packed = pipe._pack_latents(latent.to(device), 1, C, h, w)  # (1, L, C_packed)
        latents.append(packed.squeeze(0).cpu().float())
        if (idx + 1) % 50 == 0:
            print(f"  encoded {idx + 1}/{N_STIMULI}")

    stacked = torch.stack(latents, dim=0)   # (N, L, C_packed)
    flat = stacked.flatten(1)               # (N, L*C_packed)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(flat, latent_cache_path(image_size))
    print(f"Saved latent cache: {tuple(flat.shape)} -> {latent_cache_path(image_size)}")
    return flat


def load_latent_cache(image_size: int, device: str, force: bool = False) -> torch.Tensor:
    """Load the packed latent cache, rebuilding it if missing or forced.

    Args:
        image_size: Must match the size used at inference.
        device:     Used only if a rebuild is triggered.
        force:      If True, re-encode even if the cache file exists.

    Returns:
        (N_STIMULI, L*C_packed) float32 CPU tensor.
    """
    path = latent_cache_path(image_size)
    if path.exists() and not force:
        flat = torch.load(path, weights_only=True)
        print(f"Loaded latent cache: {tuple(flat.shape)} from {path}")
        return flat
    return build_latent_cache(image_size, device)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Pre-encode Rust stimuli into FLUX VAE packed latents.")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true",
                   help="Re-encode even if the cache already exists")
    args = p.parse_args()

    load_latent_cache(args.image_size, args.device, force=args.force)
