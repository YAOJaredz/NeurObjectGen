"""Quick visual test for a trained IP-Adapter checkpoint.

Loads the IP-Adapter, runs FLUX generation conditioned on the ground-truth
SigLIP embeddings for a handful of held-out stimuli, and saves a side-by-side
grid of (original | generated) to checkpoints/ip_adapter/<run>/test_grid.png.

Usage:
    python scripts/test_ip_adapter.py \
        --checkpoint checkpoints/ip_adapter/ntok128_hid1024_lr0.001_wd0.0001_size512/best.pt \
        --n-samples 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config_const import N_STIMULI, N_TRAIN, SEED, SIGLIP_EMBEDDINGS_PATH
from data_utils.stimuli import load_rust_stimuli
from generation.flux_ipadapter import load_pipeline, generate
from get_device import get_device


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(3, H, W) float [0,1] → PIL."""
    arr = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def make_grid(pairs: list[tuple[Image.Image, Image.Image]], size: int = 512) -> Image.Image:
    """Side-by-side (original | generated) grid."""
    n = len(pairs)
    grid = Image.new("RGB", (size * 2 * n, size), (30, 30, 30))
    for i, (orig, gen) in enumerate(pairs):
        grid.paste(orig.resize((size, size)), (i * size * 2, 0))
        grid.paste(gen.resize((size, size)),  (i * size * 2 + size, 0))
    return grid


def main(args):
    device = get_device()

    # --- Load checkpoint and resolve architecture args ----------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["args"]
    n_tokens  = saved.get("n_tokens", 4)
    hidden    = saved.get("hidden", 1024)
    image_size = saved.get("image_size", 512)
    print(f"Checkpoint: epoch {ckpt['epoch']}  train_loss={ckpt['loss']:.5f}")
    print(f"  n_tokens={n_tokens}  hidden={hidden}  image_size={image_size}")

    # --- Load pipeline + adapter weights ------------------------------------
    pipe, ip_adapter = load_pipeline(
        device=device,
        ip_adapter_hidden_dim=hidden,
        ip_adapter_n_tokens=n_tokens,
    )
    ip_adapter.load_state_dict(ckpt["ip_adapter_state"])
    ip_adapter.eval()

    # --- Pick held-out (val/test) indices -----------------------------------
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_STIMULI)
    held_out_idx = perm[N_TRAIN:]          # 100 held-out stimuli
    chosen = held_out_idx[:args.n_samples]

    all_images = load_rust_stimuli()       # (300, 3, 224, 224)
    all_embeds = torch.load(SIGLIP_EMBEDDINGS_PATH, weights_only=True)  # (300, 1152)

    # --- Generate -----------------------------------------------------------
    pairs = []
    for rank, idx in enumerate(chosen):
        emb = all_embeds[idx].to(device)   # (1152,)
        orig_t = all_images[idx]           # (3, 224, 224) in [0, 1]

        print(f"[{rank+1}/{args.n_samples}] stimulus {idx} …", flush=True)
        gen_img = generate(
            pipe,
            ip_adapter,
            image_embedding=emb,
            prompt=None,
            height=image_size,
            width=image_size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=int(idx),
            show_progress=False,
        )
        orig_img = tensor_to_pil(orig_t)
        pairs.append((orig_img, gen_img))

    # --- Save grid ----------------------------------------------------------
    grid = make_grid(pairs, size=image_size)
    out_path = Path(args.checkpoint).parent / "test_grid.png"
    grid.save(out_path)
    print(f"\nSaved grid ({args.n_samples} pairs) → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to best.pt or last.pt from train_ip_adapter.py")
    p.add_argument("--n-samples", type=int, default=8,
                   help="Number of held-out stimuli to generate")
    p.add_argument("--steps", type=int, default=20,
                   help="FLUX denoising steps")
    p.add_argument("--guidance-scale", type=float, default=3.5)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
