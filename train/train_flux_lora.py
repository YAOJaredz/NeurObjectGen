"""Fine-tune the FLUX transformer backbone to the HVM stimulus distribution via LoRA.

Wraps Q/K/V/O of the main attention path, the joint-attention text-side
projections, and both feed-forward modules across all 57 FLUX transformer
blocks (19 double-stream + 38 single-stream). At rank 8 this gives ~30-40M
trainable parameters while the FLUX base weights, VAE, text encoders,
MLPProjModel, and SigLIP encoder all stay frozen. The InstantX IP-Adapter
remains loaded and runs at scale=1.0 throughout, providing the SigLIP
conditioning signal the LoRA learns to denoise against.

Stochastic IP-Adapter dropout (CFG-style) drops the SigLIP conditioning at
probability ``p_drop`` per sample, so the LoRA learns to behave both with
and without guidance.

Multi-GPU: each rank loads its own frozen FLUX, and we manually all-reduce
LoRA grads across ranks before each optimizer step. We do NOT call
``Accelerator.prepare`` on the model (DDP-on-empty-forward semantics caused
a 'no grad_fn' crash in earlier versions of this script). Launch with
``accelerate launch --num_processes N`` to scale effective batch size.

Usage:
    # single GPU
    python -m train.train_flux_lora --steps 2000 --rank 8 --alpha 16 --p-drop 0.1
    # multi-GPU on one node (e.g. 2 GPUs on ax11)
    accelerate launch --num_processes 2 -m train.train_flux_lora --steps 2000 ...
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append('.')

from config_const import (
    HVM_CATEGORIES, HVM_N_VAR, HVM_STIM_DIR,
    HVM_SIGLIP_EMBEDDINGS_PATH, SIGLIP_DIM,
    FLUX_LORA_DEFAULT_PATH, FLUX_LORA_DIR,
    FLUX_LORA_RANK_DEFAULT, FLUX_LORA_ALPHA_DEFAULT, FLUX_LORA_PDROP_DEFAULT,
)
from generation.flux_instantx import (
    encode_image, encode_text_embeds, load_pipeline,
    _siglip_to_image_emb,
)
from generation.flux_lora import add_flux_lora, save_flux_lora


def _allreduce_grads(params, world_size: int) -> None:
    """All-reduce LoRA grads across ranks. Equivalent to DDP's grad-bucket sync,
    but applied directly to the LoRA params — avoids the fragile DDP-on-empty-
    forward semantics that caused 'no grad_fn' on rank 1 in earlier versions.
    """
    if world_size <= 1:
        return
    for p in params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class HVMFullDataset(Dataset):
    """450 (PIL image, SigLIP embedding) pairs in (category, variation) order.

    Order matches ``HVM_CATEGORIES * HVM_N_VAR`` (apple/00, apple/01, ..., turtle/44),
    which is the same flat indexing used by HVM_SIGLIP_EMBEDDINGS_PATH (cached by
    ``scripts/cache_siglip.py``) and by ``data_utils.hvm_loader._load_hvm_neural``.
    """

    def __init__(self, image_size: int = 512):
        self.image_size = image_size
        self.paths: list[Path] = []
        for cat in HVM_CATEGORIES:
            for var in range(HVM_N_VAR):
                self.paths.append(HVM_STIM_DIR / cat / f"{var:02d}.png")
        assert len(self.paths) == len(HVM_CATEGORIES) * HVM_N_VAR == 450, (
            f"expected 450 HVM stimuli; got {len(self.paths)}"
        )
        for p in self.paths[:1] + self.paths[-1:]:
            assert p.exists(), f"missing stimulus: {p}"

        self.siglip = torch.load(HVM_SIGLIP_EMBEDDINGS_PATH, weights_only=True)  # (450, 1152)
        assert self.siglip.shape == (450, SIGLIP_DIM), (
            f"siglip cache shape {tuple(self.siglip.shape)} mismatches (450, {SIGLIP_DIM})"
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB").resize(
            (self.image_size, self.image_size), Image.LANCZOS
        )
        return img, self.siglip[idx]


def _collate(batch):
    """Custom collate that keeps PIL images as a list (encode_image expects PIL)."""
    images = [b[0] for b in batch]
    siglip = torch.stack([b[1] for b in batch], dim=0)
    return images, siglip


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def _logit_normal(shape: tuple[int, ...], device, dtype, generator) -> torch.Tensor:
    """Sample t ~ logit-normal(0, 1) — FLUX's default training timestep distribution."""
    u = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    t = torch.sigmoid(u)
    return t.to(dtype=dtype)


def training_step(
    pipe, image_proj, images, siglip_batch, *,
    p_drop: float, generator: torch.Generator, device, dtype,
    null_prompt_embeds: torch.Tensor, null_pooled_embeds: torch.Tensor,
    null_text_ids: torch.Tensor,
) -> torch.Tensor:
    """One flow-matching denoising step. Returns the (already reduced) MSE loss."""
    batch_size = len(images)

    # --- VAE encode (no grad) ---
    with torch.no_grad():
        latents_unpacked = torch.cat(
            [encode_image(pipe, im) for im in images], dim=0
        ).to(device=device, dtype=dtype)

    _, latent_channels, lat_h, lat_w = latents_unpacked.shape
    h, w = lat_h, lat_w  # encode_image already returns (B, 16, H/8, W/8)
    latents = pipe._pack_latents(latents_unpacked, batch_size, latent_channels, h, w)
    latent_image_ids = pipe._prepare_latent_image_ids(batch_size, h // 2, w // 2, device, dtype)

    # --- SigLIP conditioning, with stochastic dropout for CFG-style training ---
    drop_mask = torch.rand((batch_size,), generator=generator, device=device) < p_drop
    siglip_batch = siglip_batch.to(device=device, dtype=dtype)
    with torch.no_grad():
        image_emb = _siglip_to_image_emb(image_proj, siglip_batch, device, dtype)
    if drop_mask.any():
        zero_mask = drop_mask.view(-1, 1, 1).to(image_emb.dtype)
        image_emb = image_emb * (1.0 - zero_mask)

    # --- Sample timestep + noise; build noisy latents (rectified flow) ---
    t = _logit_normal((batch_size,), device, torch.float32, generator)        # (B,)
    noise = torch.randn(latents.shape, device=device, dtype=dtype, generator=generator)
    sigma = t.view(-1, 1, 1).to(dtype)                                         # broadcast to (B,1,1)
    noisy = (1.0 - sigma) * latents + sigma * noise
    target = noise - latents                                                    # rectified-flow velocity

    # --- Text conditioning: empty prompt, broadcast to batch ---
    prompt_embeds = null_prompt_embeds.expand(batch_size, -1, -1)
    pooled_embeds = null_pooled_embeds.expand(batch_size, -1)
    text_ids = null_text_ids

    # --- Forward through FLUX transformer (LoRA-wrapped backbone) ---
    timesteps_in = (t * 1000.0).to(dtype)  # FLUX expects timestep / 1000 inside transformer
    guidance = torch.full((batch_size,), 3.5, device=device, dtype=dtype)

    pred = pipe.transformer(
        hidden_states=noisy,
        timestep=timesteps_in / 1000,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        joint_attention_kwargs={
            "image_emb": image_emb,
            "object_image_emb": None,
            "object_mask": None,
            "object_scale": 1.0,
        },
        return_dict=False,
    )[0]

    # Cast to fp32 for the loss so LoRA grads are stable in bf16 base.
    loss = F.mse_loss(pred.float(), target.float())
    return loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--rank", type=int, default=FLUX_LORA_RANK_DEFAULT)
    p.add_argument("--alpha", type=int, default=FLUX_LORA_ALPHA_DEFAULT)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--p-drop", type=float, default=FLUX_LORA_PDROP_DEFAULT,
                   help="probability of dropping the SigLIP conditioning per sample (CFG-style)")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", type=Path, default=FLUX_LORA_DEFAULT_PATH)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--gradient-checkpointing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = torch.bfloat16

    # ── Accelerator (device + dataloader sharding + seed; no DDP wrap) ──────
    accelerator = Accelerator()
    set_seed(args.seed)
    device = accelerator.device
    world_size = accelerator.num_processes

    if accelerator.is_main_process:
        print(f"[flux_lora] world_size={world_size} device={device} dtype={dtype}")
        print(f"[flux_lora] rank={args.rank} alpha={args.alpha} per-rank-batch={args.batch_size} grad_accum={args.grad_accum}")
        eff_batch = args.batch_size * args.grad_accum * world_size
        print(f"[flux_lora] effective batch (samples per optim step) = {eff_batch}")

    # ── Pipeline + LoRA injection (each rank loads its own frozen FLUX) ─────
    pipe, image_proj = load_pipeline(device=device, dtype=dtype, default_scale=1.0)
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    lora_params = add_flux_lora(pipe, rank=args.rank, alpha=args.alpha, dropout=args.lora_dropout)
    n_trainable = sum(p.numel() for p in lora_params)
    if accelerator.is_main_process:
        print(f"[flux_lora] trainable params: {n_trainable:,} ({n_trainable / 1e6:.2f}M)")

    n_other = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad) - n_trainable
    assert n_other == 0, f"non-LoRA trainable params leaked: {n_other:,}"

    # ── Cache empty-prompt text embeddings once ──────────────────────────────
    pooled_null, t5_null = encode_text_embeds(pipe, [""])
    pooled_null = pooled_null.to(device=device, dtype=dtype)
    t5_null = t5_null.to(device=device, dtype=dtype)
    null_text_ids = torch.zeros(t5_null.shape[1], 3, device=device, dtype=dtype)

    # ── Dataset / loader (Accelerate handles per-rank sharding) ─────────────
    dataset = HVMFullDataset(image_size=args.image_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=_collate, drop_last=True, pin_memory=False,
    )

    # ── Optimizer + scheduler ────────────────────────────────────────────────
    optim = AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay,
                  betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = LambdaLR(optim, lr_lambda)

    # ── Shard the dataloader across ranks (no model wrap) ───────────────────
    loader = accelerator.prepare(loader)

    # ── Training loop ────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        FLUX_LORA_DIR.mkdir(parents=True, exist_ok=True)

    # Per-rank generator: offset by rank so dropout / noise differ across GPUs.
    generator = torch.Generator(device=device).manual_seed(args.seed + accelerator.process_index)
    step = 0
    accum = 0
    loss_window: list[float] = []
    t0 = time.time()

    pipe.transformer.train()
    optim.zero_grad(set_to_none=True)
    pbar = tqdm(total=args.steps, desc="flux_lora", dynamic_ncols=True,
                disable=not accelerator.is_main_process)
    data_iter = iter(loader)
    while step < args.steps:
        try:
            images, siglip_batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, siglip_batch = next(data_iter)

        loss = training_step(
            pipe, image_proj, images, siglip_batch,
            p_drop=args.p_drop, generator=generator, device=device, dtype=dtype,
            null_prompt_embeds=t5_null, null_pooled_embeds=pooled_null,
            null_text_ids=null_text_ids,
        )
        (loss / args.grad_accum).backward()
        accum += 1
        loss_window.append(loss.detach().float().item())

        if accum >= args.grad_accum:
            _allreduce_grads(lora_params, world_size)
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            if accelerator.is_main_process:
                pbar.update(1)

                if step % args.log_every == 0:
                    avg = sum(loss_window) / len(loss_window)
                    loss_window = []
                    lr_now = scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}")

                if step % args.save_every == 0 or step == args.steps:
                    ckpt_path = args.save_path.with_name(
                        args.save_path.stem + f"_step{step}" + args.save_path.suffix
                    )
                    save_flux_lora(pipe, ckpt_path)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_flux_lora(pipe, args.save_path)
        pbar.close()
        print(f"[flux_lora] done. wallclock={time.time() - t0:.1f}s. ckpt={args.save_path}")


if __name__ == "__main__":
    main()
