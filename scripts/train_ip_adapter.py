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
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

sys.path.insert(0, ".")
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
    # Resize to target size.
    x = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    # The Rust stimuli are grayscale — collapse to luminance and replicate across
    # RGB so the VAE sees a proper 3-channel input. This makes both the clean
    # latents and the flow-matching target grayscale; the predicted velocity is
    # regressed against grayscale content with no pixel-space decode needed.
    gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3])
    x = gray.expand(-1, 3, -1, -1).contiguous()
    # Shift to [-1, 1] for the VAE.
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

    # During training we pass ONLY the IP-Adapter tokens as encoder_hidden_states
    # — no T5 context.  With empty T5 tokens FLUX already produces a usable
    # baseline prediction that swamps the 4 adapter tokens, flattening their
    # gradient signal.  Dropping T5 entirely forces all conditioning through the
    # adapter and maximises gradient flow to the MLP.
    # pooled_embeds (CLIP) is still needed by the transformer's pooled projection.
    with torch.no_grad():
        _, pooled_embeds, _ = pipe.encode_prompt(
            prompt="",
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
    assert pooled_embeds is not None
    # pooled_embeds: (1, 768)

    # Offload text encoders — no longer needed, frees ~9.6 GB.
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    n_train = clean_latents.shape[0]
    # Two-level batching:
    #   micro_batch: samples processed concurrently in one FLUX forward (GPU-limited)
    #   batch_size:  logical gradient-update size — grads accumulate over
    #                (batch_size / micro_batch) forward passes before each optimizer step
    micro_batch = args.micro_batch
    assert args.batch_size % micro_batch == 0, \
        f"batch_size ({args.batch_size}) must be divisible by micro_batch ({micro_batch})"
    accum_steps = args.batch_size // micro_batch
    bf16 = torch.bfloat16
    mask = loss_mask.float()  # (1, seq, 1) — constant across steps

    # --- Optimizer ---
    optimizer = AdamW(ip_adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # OneCycleLR: warm up to args.lr, then cosine-anneal down. One scheduler
    # step per optimizer step (= per logical batch of args.batch_size samples).
    steps_per_epoch = (n_train + args.batch_size - 1) // args.batch_size
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    ckpt_dir = run_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints -> {ckpt_dir}")

    best_loss = float("inf")

    # Shared constants across micro-batches — allocated once.
    ip_ids = torch.zeros(
        ip_adapter.n_tokens, 3, device=device, dtype=torch.bfloat16
    )

    for epoch in range(1, args.epochs + 1):
        ip_adapter.train()
        perm_e = torch.randperm(n_train)
        epoch_loss = 0.0
        n_micro = 0      # forward-pass count (for logging average)
        n_updates = 0

        optimizer.zero_grad()
        n_forwards = (n_train + micro_batch - 1) // micro_batch
        pbar = tqdm(range(n_forwards), desc=f"epoch {epoch:3d}/{args.epochs}", leave=False)
        for step in pbar:
            start = step * micro_batch
            end = min(start + micro_batch, n_train)
            idx = perm_e[start:end]
            b = idx.shape[0]

            x0 = clean_latents[idx].to(device, dtype=bf16)   # (b, seq, 64)
            emb = siglip_embeds[idx]                         # (b, 1152)

            # --- Flow matching with logit-normal timestep sampling ---
            # SD3/FLUX-standard: t = sigmoid(N(0,1)) concentrates mass in the mid-
            # noise regime where conditioning actually matters. Uniform sampling
            # wastes capacity on t≈1 (pure noise, target ≈ noise, input-independent).
            t = torch.sigmoid(torch.randn(b, device=device)).to(bf16)   # (b,) in (0,1)
            noise = torch.randn_like(x0)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * noise
            target = noise - x0

            ip_tokens = ip_adapter(emb.float()).to(bf16)     # (b, n_tokens, 4096)
            guidance = torch.full([b], args.guidance_scale, device=device, dtype=bf16)

            pred = pipe.transformer(
                hidden_states=xt,
                timestep=t,  # transformer expects normalised t in (0, 1); inference divides scheduler steps by 1000
                guidance=guidance,
                encoder_hidden_states=ip_tokens,
                pooled_projections=pooled_embeds.expand(b, -1),
                txt_ids=ip_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            sq_err = (pred.float() - target.float()) ** 2    # (b, seq, 64)
            # Masked mean over (seq, 64) per sample, then mean across b.
            denom = mask.sum() * sq_err.shape[-1]
            per_sample_loss = (sq_err * mask).sum(dim=(1, 2)) / denom  # (b,)
            micro_loss = per_sample_loss.mean()
            # Scale by (b / batch_size) so the summed gradient over one logical
            # batch equals the mean gradient over batch_size samples.
            loss = micro_loss * (b / args.batch_size)
            loss.backward()

            epoch_loss += micro_loss.item() * b
            n_micro += b
            is_last = (step == n_forwards - 1)

            if (step + 1) % accum_steps == 0 or is_last:
                torch.nn.utils.clip_grad_norm_(ip_adapter.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                n_updates += 1
                pbar.set_postfix(loss=f"{epoch_loss / n_micro:.5f}")

        epoch_loss /= n_train
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}/{args.epochs}  loss={epoch_loss:.5f}  lr={current_lr:.2e}")

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
    p.add_argument("--n-tokens", type=int, default=16, help="Number of IP-Adapter text tokens")
    p.add_argument("--hidden", type=int, default=1024, help="IP-Adapter MLP hidden dim")
    p.add_argument("--image-size", type=int, default=512, help="Target image resolution (pixels)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32,
                   help="Logical gradient-update batch size (samples per optimizer step)")
    p.add_argument("--micro-batch", type=int, default=2,
                   help="Samples per FLUX forward pass — limited by GPU memory")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--guidance-scale", type=float, default=3.5,
                   help="FLUX distilled guidance passed at training time (3.5 matches distillation)")
    p.add_argument("--mask-center-frac", type=float, default=0.06,
                   help="Side of the excluded center square as a fraction of image_size")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
