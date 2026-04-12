"""Train the IP-Adapter projection to reconstruct images from SigLIP embeddings.

Given frozen FLUX.1-dev and the cached SigLIP image embeddings of the Rust
training stimuli, we train IPAdapterProjection so that FLUX — conditioned on
the projected extra text tokens — reconstructs the original stimulus via a
flow-matching loss. Only the IP-Adapter MLP updates; every other weight is
frozen.

At test time the same adapter is fed embeddings predicted by the neural
encoder (see generation/flux_ipadapter.generate).

Usage:
    python scripts/train_ip_adapter.py --epochs 50 --batch-size 1 --lr 1e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config_const import (
    CHECKPOINT_DIR,
    N_STIMULI,
    N_TRAIN,
    RUST_LOSS_MASK_PATH,
    SEED,
    SIGLIP_EMBEDDINGS_PATH,
)
from data_utils.stimuli import load_rust_stimuli
from generation.flux_ipadapter import load_pipeline
from get_device import get_device


# ---------------------------------------------------------------------------
# Target preparation: VAE-encode all training stimuli once, then pack into
# FLUX's sequence-of-2x2-patches latent format.
# ---------------------------------------------------------------------------

@torch.no_grad()
def prepare_latents(pipe, images: torch.Tensor, size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """VAE-encode images and pack them for the FLUX transformer.

    Args:
        pipe:    Loaded FluxPipeline.
        images:  (N, 3, H0, W0) tensor in [0, 1] (any size, gets resized).
        size:    Target square resolution in pixels.
        device:  Target device.

    Returns:
        packed_latents:    (N, seq_len, 64) bfloat16 tensor.
        latent_image_ids:  (seq_len, 3) position ids shared across samples.
    """
    # Resize to target size and shift to [-1, 1] for the VAE.
    x = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    x = x * 2 - 1

    n = x.shape[0]
    latents = []
    for i in tqdm(range(n), desc="VAE encoding"):
        xi = x[i : i + 1].to(device, dtype=pipe.vae.dtype)
        latent = pipe.vae.encode(xi).latent_dist.sample()
        latent = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        latents.append(latent)
    latents_bchw = torch.cat(latents, dim=0)  # (N, 16, size/8, size/8)

    _, c, h, w = latents_bchw.shape
    packed = pipe._pack_latents(latents_bchw, n, c, h, w)  # (N, (h/2)*(w/2), c*4)

    latent_image_ids = pipe._prepare_latent_image_ids(
        batch_size=1,
        height=h // 2,
        width=w // 2,
        device=device,
        dtype=packed.dtype,
    )
    return packed, latent_image_ids


# ---------------------------------------------------------------------------
# Loss mask: circle (valid stimulus) minus center square (fixation marker)
# ---------------------------------------------------------------------------

def make_packed_loss_mask(
    image_size: int,
    center_frac: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a weighted mask over packed-latent positions.

    The Rust stimuli are presented in a circular aperture with a small white
    fixation square at the center. This mask is 1 inside the circle, 0 outside,
    and 0 in the center square — then soft-downsampled from pixel space to the
    packed-latent grid (each packed position covers a 16x16 pixel block for
    a 512px image: VAE /8 then pack /2).

    The result is cached at ``RUST_LOSS_MASK_PATH`` together with the
    parameters that produced it, and reloaded on subsequent calls when the
    parameters match.

    Args:
        image_size:  Pixel resolution used for training (square).
        center_frac: Side length of the excluded center square as a fraction
                     of image_size (e.g. 0.06 ≈ 30 px at 512).
        device:      Target device.

    Returns:
        (1, packed_seq_len, 1) float tensor — ready to multiply into
        per-position squared error over FLUX's packed latents.
    """
    cached_params = {"image_size": image_size, "center_frac": center_frac}
    if RUST_LOSS_MASK_PATH.exists():
        blob = torch.load(RUST_LOSS_MASK_PATH, weights_only=False, map_location="cpu")
        if blob.get("params") == cached_params:
            return blob["mask"].to(device)

    yy, xx = torch.meshgrid(
        torch.arange(image_size, dtype=torch.float32),
        torch.arange(image_size, dtype=torch.float32),
        indexing="ij",
    )
    c = (image_size - 1) / 2
    r = image_size / 2
    circle = ((yy - c) ** 2 + (xx - c) ** 2) <= r ** 2

    half = int(round(image_size * center_frac / 2))
    center_sq = torch.zeros_like(circle, dtype=torch.bool)
    lo = int(round(c)) - half
    hi = int(round(c)) + half + 1
    center_sq[lo:hi, lo:hi] = True

    pixel_mask = (circle & ~center_sq).float().view(1, 1, image_size, image_size)
    # Each packed-latent position spans a 16x16 pixel block → soft coverage in [0, 1].
    packed_grid = image_size // 16
    mask = F.avg_pool2d(pixel_mask, kernel_size=16).view(1, packed_grid * packed_grid, 1)

    RUST_LOSS_MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mask": mask.cpu(), "params": cached_params}, RUST_LOSS_MASK_PATH)
    print(f"saved loss mask to {RUST_LOSS_MASK_PATH}")

    return mask.to(device)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def run_dir(args) -> Path:
    name = f"ntok{args.n_tokens}_hid{args.hidden}_lr{args.lr}_wd{args.weight_decay}_size{args.image_size}"
    return CHECKPOINT_DIR / "ip_adapter" / name


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(SEED)
    device = get_device()

    # --- Load FLUX + fresh IP-Adapter ---
    pipe, ip_adapter = load_pipeline(
        device=device,
        ip_adapter_hidden_dim=args.hidden,
        ip_adapter_n_tokens=args.n_tokens,
    )
    # Keep ip_adapter in float32 for stable optimizer updates; we'll cast its
    # output to bfloat16 before feeding it to the transformer.
    ip_adapter.float()
    for p in ip_adapter.parameters():
        p.requires_grad_(True)
    print(f"IP-Adapter params: {sum(p.numel() for p in ip_adapter.parameters()):,}")

    # --- Load data: training-split images + matching SigLIP embeddings ---
    if not SIGLIP_EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"SigLIP cache not found at {SIGLIP_EMBEDDINGS_PATH}. "
            "Run: python scripts/cache_siglip.py"
        )
    all_images = load_rust_stimuli()  # (300, 3, 224, 224)
    all_embeds = torch.load(SIGLIP_EMBEDDINGS_PATH, weights_only=True)  # (300, 1152)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_STIMULI)
    train_idx = torch.from_numpy(perm[:N_TRAIN]).long()
    images = all_images[train_idx]
    siglip_embeds = all_embeds[train_idx].to(device)

    # --- VAE-encode and pack all training latents once ---
    clean_latents, latent_image_ids = prepare_latents(pipe, images, args.image_size, device)
    print(f"clean_latents: {clean_latents.shape}  latent_image_ids: {latent_image_ids.shape}")

    # --- Loss mask: circle minus center fixation square, in packed-latent space ---
    loss_mask = make_packed_loss_mask(args.image_size, args.mask_center_frac, device)
    loss_mask = loss_mask.to(clean_latents.dtype)
    mask_sum = loss_mask.sum()
    print(f"loss_mask: {loss_mask.shape}  fraction active: {(mask_sum / loss_mask.numel()).item():.3f}")

    # --- Precompute empty-prompt text conditioning (shared across samples) ---
    with torch.no_grad():
        prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
            prompt="",
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
    # prompt_embeds: (1, 512, 4096)  pooled_embeds: (1, 768)  text_ids: (512, 3)

    # Offload text encoders — they're no longer needed and freeing ~9.6 GB
    # gives enough headroom for batch_size=4 activation retention during backprop.
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    # --- Optimizer ---
    optimizer = AdamW(ip_adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = run_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints -> {ckpt_dir}")

    best_loss = float("inf")
    n_train = clean_latents.shape[0]
    # Gradient accumulation: accumulate args.batch_size individual samples
    # before each optimizer step (FLUX cannot hold multiple samples in memory).
    accum_steps = args.batch_size
    bf16 = torch.bfloat16
    mask = loss_mask.float()  # (1, seq, 1) — constant across steps

    for epoch in range(1, args.epochs + 1):
        ip_adapter.train()
        perm_e = torch.randperm(n_train)
        epoch_loss = 0.0
        n_updates = 0

        optimizer.zero_grad()
        pbar = tqdm(range(n_train), desc=f"epoch {epoch:3d}/{args.epochs}", leave=False)
        for step, i in enumerate(pbar):
            idx = perm_e[i : i + 1]  # one sample at a time through FLUX

            x0 = clean_latents[idx].to(device, dtype=bf16)  # (1, seq, 64)
            emb = siglip_embeds[idx]                         # (1, 1152)

            # --- Flow matching ---
            t = torch.rand(1, device=device, dtype=bf16)
            noise = torch.randn_like(x0)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * noise
            target = noise - x0

            # --- IP-Adapter tokens ---
            ip_tokens = ip_adapter(emb.float()).to(bf16)     # (1, n_tokens, 4096)
            prompt_b = torch.cat([ip_tokens, prompt_embeds], dim=1)
            ip_ids = torch.zeros(
                ip_adapter.n_tokens, text_ids.shape[-1], device=device, dtype=text_ids.dtype
            )
            text_ids_b = torch.cat([ip_ids, text_ids], dim=0)
            guidance = torch.full([1], args.guidance_scale, device=device, dtype=bf16)

            pred = pipe.transformer(
                hidden_states=xt,
                timestep=t,
                guidance=guidance,
                encoder_hidden_states=prompt_b,
                pooled_projections=pooled_embeds,
                txt_ids=text_ids_b,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            sq_err = (pred.float() - target.float()) ** 2   # (1, seq, 64)
            # Divide by accum_steps so gradients average over the logical batch.
            loss = (sq_err * mask).sum() / (mask.sum() * sq_err.shape[-1] * accum_steps)
            loss.backward()

            epoch_loss += loss.item() * accum_steps          # log the un-scaled loss
            is_last = (i == n_train - 1)

            if (step + 1) % accum_steps == 0 or is_last:
                optimizer.step()
                optimizer.zero_grad()
                n_updates += 1
                avg = epoch_loss / (step + 1)
                pbar.set_postfix(loss=f"{avg:.5f}")

        scheduler.step()
        epoch_loss /= n_train
        print(f"epoch {epoch:3d}/{args.epochs}  loss={epoch_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

        torch.save(
            {
                "epoch": epoch,
                "ip_adapter_state": ip_adapter.state_dict(),
                "loss": epoch_loss,
                "args": vars(args),
            },
            ckpt_dir / "last.pt",
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "epoch": epoch,
                    "ip_adapter_state": ip_adapter.state_dict(),
                    "loss": epoch_loss,
                    "args": vars(args),
                },
                ckpt_dir / "best.pt",
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-tokens", type=int, default=4, help="Number of IP-Adapter text tokens")
    p.add_argument("--hidden", type=int, default=1024, help="IP-Adapter MLP hidden dim")
    p.add_argument("--image-size", type=int, default=512, help="Target image resolution (pixels)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--guidance-scale", type=float, default=1.0,
                   help="FLUX distilled guidance passed at training time (1.0 = neutral)")
    p.add_argument("--mask-center-frac", type=float, default=0.06,
                   help="Side of the excluded center square as a fraction of image_size")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
