"""Train the IP-Adapter projection with an img2img flow-matching objective.

Unlike ``train_ip_adapter.py`` which samples timesteps over the full range
(0, 1) and uses an MSE velocity loss, this script makes two changes:

1. Restricts training timesteps to [strength_min, strength_max] to match the
   partial-noise regime used by ``generate_img2img()`` at inference.
2. Uses an x0-prediction L1 loss: the predicted velocity is converted back to
   a clean-latent estimate (x0_pred = (xt - t*v) / (1-t)) and L1 is applied
   against the true clean latent. This directly optimises for image similarity
   rather than velocity-space agreement and produces sharper reconstructions.

Embedding source is switchable via ``--embedding-source``:
  siglip  (default) — ground-truth SigLIP embeddings from cache
  neural             — embeddings predicted by the best trained neural encoder

Usage:
    python scripts/train_ip_adapter_img2img.py \\
        --embedding-source siglip \\
        --strength-min 0.5 --strength-max 0.9 \\
        --epochs 100 --batch-size 32 --micro-batch 2
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
    N_VAL,
    RUST_LOSS_MASK_PATH,
    SEED,
    SIGLIP_DIM,
    SIGLIP_EMBEDDINGS_PATH,
)
from data_utils.stimuli import load_rust_stimuli
from generation.flux_ipadapter import load_pipeline
from get_device import get_device


# ---------------------------------------------------------------------------
# Target preparation — identical to train_ip_adapter.py
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
    x = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    gray = (0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3])
    x = gray.expand(-1, 3, -1, -1).contiguous()
    x = x * 2 - 1

    n = x.shape[0]
    latents = []
    for i in tqdm(range(n), desc="VAE encoding", file=sys.stdout):
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
# Loss mask — identical to train_ip_adapter.py
# ---------------------------------------------------------------------------

def make_packed_loss_mask(
    image_size: int,
    center_frac: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Build a weighted mask over packed-latent positions.

    Circle aperture minus center fixation square, soft-downsampled to the
    packed-latent grid. Cached to RUST_LOSS_MASK_PATH.

    Returns:
        (1, packed_seq_len, 1) float tensor.
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
    packed_grid = image_size // 16
    mask = F.avg_pool2d(pixel_mask, kernel_size=16).view(1, packed_grid * packed_grid, 1)

    RUST_LOSS_MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mask": mask.cpu(), "params": cached_params}, RUST_LOSS_MASK_PATH)
    print(f"saved loss mask to {RUST_LOSS_MASK_PATH}")

    return mask.to(device)


# ---------------------------------------------------------------------------
# Checkpoint directory naming
# ---------------------------------------------------------------------------

def run_dir(args) -> Path:
    name = (
        f"ntok{args.n_tokens}_hid{args.hidden}"
        f"_lr{args.lr}_wd{args.weight_decay}"
        f"_size{args.image_size}"
        f"_emb{args.embedding_source}"
        f"_s{args.strength_min}-{args.strength_max}"
    )
    return CHECKPOINT_DIR / "ip_adapter_img2img" / name


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_train_embeddings(args, train_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return (N_TRAIN, SIGLIP_DIM) embedding tensor on device.

    siglip  — ground-truth SigLIP embeddings from cache
    neural  — embeddings predicted by the best trained neural encoder
    """
    if args.embedding_source == "siglip":
        if not SIGLIP_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"SigLIP cache not found at {SIGLIP_EMBEDDINGS_PATH}. "
                "Run: python scripts/cache_siglip.py"
            )
        all_embeds = torch.load(SIGLIP_EMBEDDINGS_PATH, weights_only=True)  # (300, 1152)
        return all_embeds[train_idx].to(device)

    # neural: run best encoder on training neural responses
    from data_utils.rust_loader import make_rust_loader
    from encoders import BottleneckMLP, TemporalLSTM, TemporalTransformer

    # Find best encoder checkpoint by val_2afc
    records = []
    for pt in sorted(CHECKPOINT_DIR.rglob("best.pt")):
        if "ip_adapter" in pt.parts:
            continue
        ckpt = torch.load(pt, weights_only=False, map_location="cpu")
        if "model_state" not in ckpt:
            continue
        records.append({**ckpt["args"], **ckpt.get("metrics", {}), "path": str(pt), "_ckpt": ckpt})
    if not records:
        raise RuntimeError(f"No encoder checkpoints found under {CHECKPOINT_DIR}")
    best = max(records, key=lambda r: r.get("val_2afc", 0))
    print(f"Neural encoder: {best['path']}  val_2afc={best['val_2afc']:.4f}")

    # Load full dataset to get neural responses in split order
    train_loader, _, _ = make_rust_loader(use_embeddings=True, verbose=True)
    # Collect all training samples in a single pass (loader is shuffled, so
    # we iterate until we have all N_TRAIN samples)
    all_neural = []
    all_targets = []
    for neural, tgt in train_loader:
        all_neural.append(neural)
        all_targets.append(tgt)
    # Stack — shape (N_TRAIN, neurons, time)
    neural_train = torch.cat(all_neural, dim=0)[:N_TRAIN]

    _, n_neurons, n_time = neural_train.shape
    enc_args = best
    if enc_args["model"] == "mlp":
        enc_model = BottleneckMLP(
            in_dim=n_neurons * n_time,
            bottleneck=enc_args["bottleneck"],
            out_dim=SIGLIP_DIM,
            dropout=enc_args["dropout"],
        )
    elif enc_args["model"] == "lstm":
        enc_model = TemporalLSTM(
            n_neurons=n_neurons,
            hidden=enc_args["hidden"],
            out_dim=SIGLIP_DIM,
            dropout=enc_args["dropout"],
        )
    else:
        enc_model = TemporalTransformer(
            n_neurons=n_neurons,
            d_model=enc_args["d_model"],
            n_heads=enc_args["n_heads"],
            n_layers=enc_args["n_layers"],
            out_dim=SIGLIP_DIM,
            dropout=enc_args["dropout"],
        )
    enc_model.load_state_dict(best["_ckpt"]["model_state"])
    enc_model.to(device).eval()

    with torch.no_grad():
        x = neural_train.to(device)
        if enc_args["model"] == "mlp":
            x = x.flatten(1)
        else:
            x = x.permute(0, 2, 1)
        embeds = enc_model(x)  # (N_TRAIN, 1152)

    print(f"Neural embeddings: {embeds.shape}")
    return embeds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    # --- Load FLUX + fresh IP-Adapter ---
    pipe, ip_adapter = load_pipeline(
        device=device,
        ip_adapter_hidden_dim=args.hidden,
        ip_adapter_n_tokens=args.n_tokens,
        ip_adapter_dropout=args.mlp_dropout,
    )
    ip_adapter.float()
    for p in ip_adapter.parameters():
        p.requires_grad_(True)
    print(f"IP-Adapter params: {sum(p.numel() for p in ip_adapter.parameters()):,}")

    # --- Split indices ---
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N_STIMULI)
    train_idx = torch.from_numpy(perm[:N_TRAIN]).long()
    val_idx   = torch.from_numpy(perm[N_TRAIN:N_TRAIN + N_VAL]).long()

    # --- Load images + embeddings ---
    all_images = load_rust_stimuli()  # (300, 3, 224, 224)
    train_embeds = load_train_embeddings(args, train_idx, device)
    val_embeds   = load_train_embeddings(args, val_idx,   device)
    print(f"Embeddings: train={train_embeds.shape}  val={val_embeds.shape}  source={args.embedding_source}")

    # --- VAE-encode and pack train + val latents once ---
    clean_latents,     latent_image_ids = prepare_latents(pipe, all_images[train_idx], args.image_size, device)
    clean_latents_val, _                = prepare_latents(pipe, all_images[val_idx],   args.image_size, device)
    print(f"clean_latents: {clean_latents.shape}  val: {clean_latents_val.shape}")

    # --- Loss mask ---
    loss_mask = make_packed_loss_mask(args.image_size, args.mask_center_frac, device)
    loss_mask = loss_mask.to(clean_latents.dtype)
    mask_sum = loss_mask.sum()
    print(f"loss_mask: {loss_mask.shape}  fraction active: {(mask_sum / loss_mask.numel()).item():.3f}")

    # Pooled CLIP embeds needed by transformer; text encoders can be offloaded after.
    with torch.no_grad():
        _, pooled_embeds, _ = pipe.encode_prompt(
            prompt="",
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
    assert pooled_embeds is not None
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    n_train = clean_latents.shape[0]
    micro_batch = args.micro_batch
    assert args.batch_size % micro_batch == 0, \
        f"batch_size ({args.batch_size}) must be divisible by micro_batch ({micro_batch})"
    accum_steps = args.batch_size // micro_batch
    bf16 = torch.bfloat16
    mask = loss_mask.float()

    # --- Optimizer ---
    optimizer = AdamW(ip_adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    print(f"Strength range: [{args.strength_min}, {args.strength_max}]")

    best_loss = float("inf")
    ip_ids = torch.zeros(ip_adapter.n_tokens, 3, device=device, dtype=bf16)

    t_lo = args.strength_min
    t_hi = args.strength_max

    for epoch in range(1, args.epochs + 1):
        ip_adapter.train()
        perm_e = torch.randperm(n_train)
        epoch_loss = 0.0
        n_micro = 0
        n_updates = 0

        optimizer.zero_grad()
        n_forwards = (n_train + micro_batch - 1) // micro_batch
        pbar = tqdm(range(n_forwards), desc=f"epoch {epoch:3d}/{args.epochs}", leave=False)
        for step in pbar:
            start = step * micro_batch
            end = min(start + micro_batch, n_train)
            idx = perm_e[start:end]
            b = idx.shape[0]

            x0  = clean_latents[idx].to(device, dtype=bf16)   # (b, seq, 64)
            emb = train_embeds[idx].clone()                    # (b, 1152)

            # --- Embedding augmentation (training only) ---
            # 1. Gaussian noise: simulates the gap between ground-truth SigLIP
            #    embeddings (train) and noisy neural-predicted embeddings (inference).
            if args.embed_noise_std > 0:
                emb = emb + args.embed_noise_std * torch.randn_like(emb)
            # 2. Embedding dropout: randomly zero the full embedding, forcing the
            #    MLP to learn a meaningful null condition.
            if args.embed_dropout > 0:
                drop_mask = (torch.rand(emb.shape[0], 1, device=emb.device) > args.embed_dropout)
                emb = emb * drop_mask

            # --- img2img timestep sampling ---
            # Restrict t to [strength_min, strength_max] so training matches the
            # partial-noise regime used by generate_img2img() at inference.
            # Within that range, use logit-normal to concentrate mass away from
            # the edges (same motivation as the full-range logit-normal in the
            # txt2img script).
            t_raw = torch.sigmoid(torch.randn(b, device=device)).to(bf16)  # (0, 1)
            t = t_lo + (t_hi - t_lo) * t_raw                              # [t_lo, t_hi]

            noise  = torch.randn_like(x0)
            xt     = (1 - t[:, None, None]) * x0 + t[:, None, None] * noise

            ip_tokens = ip_adapter(emb.float()).to(bf16)      # (b, n_tokens, 4096)
            guidance  = torch.full([b], args.guidance_scale, device=device, dtype=bf16)

            # pred is the velocity (noise - x0) as usual for FLUX
            pred = pipe.transformer(
                hidden_states=xt,
                timestep=t,
                guidance=guidance,
                encoder_hidden_states=ip_tokens,
                pooled_projections=pooled_embeds.expand(b, -1),
                txt_ids=ip_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            # x0-prediction loss with L1:
            # Convert predicted velocity → x0 estimate via the flow ODE inverse:
            #   xt = (1-t)*x0 + t*noise  →  x0 = (xt - t*pred) / (1-t)
            # At low t (our regime) (1-t) is well away from zero, so this is stable.
            # L1 on x0 directly optimises for clean-image similarity rather than
            # velocity-space agreement, giving sharper reconstructions.
            x0_pred = (xt - t[:, None, None] * pred.to(bf16)) / (1 - t[:, None, None])
            err   = (x0_pred.float() - x0.float()).abs()       # L1, (b, seq, 64)
            denom = mask.sum() * err.shape[-1]
            per_sample_loss = (err * mask).sum(dim=(1, 2)) / denom   # (b,)
            micro_loss = per_sample_loss.mean()
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

        # --- Validation ---
        ip_adapter.eval()
        val_loss = 0.0
        n_val = clean_latents_val.shape[0]
        with torch.no_grad():
            for v_start in range(0, n_val, micro_batch):
                v_end = min(v_start + micro_batch, n_val)
                v_idx = torch.arange(v_start, v_end)
                b = v_idx.shape[0]

                x0  = clean_latents_val[v_idx].to(device, dtype=bf16)
                emb = val_embeds[v_idx]

                t_raw = torch.sigmoid(torch.randn(b, device=device)).to(bf16)
                t     = t_lo + (t_hi - t_lo) * t_raw
                noise = torch.randn_like(x0)
                xt    = (1 - t[:, None, None]) * x0 + t[:, None, None] * noise

                ip_tokens = ip_adapter(emb.float()).to(bf16)
                guidance  = torch.full([b], args.guidance_scale, device=device, dtype=bf16)

                pred = pipe.transformer(
                    hidden_states=xt,
                    timestep=t,
                    guidance=guidance,
                    encoder_hidden_states=ip_tokens,
                    pooled_projections=pooled_embeds.expand(b, -1),
                    txt_ids=ip_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

                x0_pred = (xt - t[:, None, None] * pred.to(bf16)) / (1 - t[:, None, None])
                err     = (x0_pred.float() - x0.float()).abs()
                denom   = mask.sum() * err.shape[-1]
                val_loss += (err * mask).sum(dim=(1, 2)).sum().item() / denom.item()

        val_loss /= n_val
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d}/{args.epochs}"
            f"  train={epoch_loss:.5f}"
            f"  val={val_loss:.5f}"
            f"  lr={current_lr:.2e}"
        )

        ckpt = {
            "epoch": epoch,
            "ip_adapter_state": ip_adapter.state_dict(),
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "args": vars(args),
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(ckpt, ckpt_dir / "best.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-tokens",          type=int,   default=128)
    p.add_argument("--hidden",            type=int,   default=1024)
    p.add_argument("--image-size",        type=int,   default=512)
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--batch-size",        type=int,   default=32)
    p.add_argument("--micro-batch",       type=int,   default=2)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--guidance-scale",    type=float, default=3.5)
    p.add_argument("--mask-center-frac",  type=float, default=0.06)
    p.add_argument("--embedding-source",  type=str,   default="siglip",
                   choices=["siglip", "neural"],
                   help="siglip: ground-truth SigLIP embeddings; neural: best encoder predictions")
    p.add_argument("--strength-min",      type=float, default=0.5,
                   help="Lower bound of the partial-noise range for img2img training")
    p.add_argument("--strength-max",      type=float, default=0.9,
                   help="Upper bound of the partial-noise range for img2img training")
    p.add_argument("--mlp-dropout",       type=float, default=0.1,
                   help="Dropout rate inside the IP-Adapter MLP (between the two linear layers)")
    p.add_argument("--embed-dropout",     type=float, default=0.1,
                   help="Probability of zeroing the entire embedding for a sample (CFG-style)")
    p.add_argument("--embed-noise-std",   type=float, default=0.05,
                   help="Std of Gaussian noise added to embeddings during training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert 0.0 <= args.strength_min < args.strength_max <= 1.0, \
        "strength_min and strength_max must satisfy 0 <= min < max <= 1"
    train(args)
