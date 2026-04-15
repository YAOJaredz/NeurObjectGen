"""FLUX.1-dev + pretrained InstantX IP-Adapter for neural image reconstruction.

This module vendors the two small pieces of InstantX/FLUX.1-dev-IP-Adapter
that diffusers' native ``pipe.load_ip_adapter`` cannot consume directly:

  * ``MLPProjModel``  — the 2-layer MLP projecting a (B, 1152) SigLIP
    pooler embedding into (B, 128, 4096) IP tokens.
  * ``IPAFluxAttnProcessor`` — a FluxAttention processor that adds a
    parallel IP key/value path using weights shipped in
    ``ip-adapter.bin`` at ``ip_adapter.{0..56}.to_{k,v}_ip.weight``. One
    instance is installed on every FLUX attention block (19 double +
    38 single).

Both are wired into a frozen ``FluxPipeline`` by ``load_pipeline`` and
driven from ``generate`` / ``generate_img2img``. The image embedding is
passed through ``joint_attention_kwargs={"image_emb": ...}`` so we can
reuse diffusers' stock ``FluxTransformer2DModel.forward`` without a fork.

Rust-dataset helpers preserved from the old custom-adapter wrapper:
  * ``encode_image`` — grayscale-luminance VAE encode matching training
  * ``decode_latent``
  * per-step aperture compositing inside ``generate_img2img``
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.normalization import RMSNorm
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipeline,
    calculate_shift,
    retrieve_timesteps,
)
from huggingface_hub import hf_hub_download
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from config_const import SIGLIP_DIM
from generation.aperture import load_packed_aperture_mask


# ---------------------------------------------------------------------------
# Constants — fixed by the InstantX checkpoint.
# ---------------------------------------------------------------------------

INSTANTX_REPO = "InstantX/FLUX.1-dev-IP-Adapter"
INSTANTX_WEIGHTS = "ip-adapter.bin"

FLUX_JOINT_DIM = 4096       # cross_attention_dim
FLUX_HIDDEN_DIM = 3072      # num_attention_heads * attention_head_dim
NUM_IP_TOKENS = 128         # from image_proj shape: 524288 = 128 * 4096


# ---------------------------------------------------------------------------
# Image projection — ported from InstantX infer_flux_ipa_siglip.py.
# ---------------------------------------------------------------------------

class MLPProjModel(nn.Module):
    """Maps a (B, id_embeddings_dim) SigLIP pooler embedding to (B, num_tokens, cross_attention_dim)."""

    def __init__(
        self,
        cross_attention_dim: int = FLUX_JOINT_DIM,
        id_embeddings_dim: int = SIGLIP_DIM,
        num_tokens: int = NUM_IP_TOKENS,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Attention processor — ported from InstantX attention_processor.py and
# adapted to diffusers 0.37.1 FluxAttention conventions.
# ---------------------------------------------------------------------------

class IPAFluxAttnProcessor(nn.Module):
    """Flux attention processor that adds a parallel IP key/value path.

    Identical math to InstantX's IPAFluxAttnProcessor2_0, with:
      * ``image_emb`` declared in ``__call__`` so diffusers forwards it
        via ``joint_attention_kwargs``.
      * Compatible return shape for both double-stream (2-tuple) and
        single-stream (tensor) FLUX blocks in diffusers 0.37.1.
      * No trainable params besides ``to_k_ip`` / ``to_v_ip`` — the
        ``norm_added_k`` is a non-parametric RMSNorm, matching InstantX.
    """

    def __init__(
        self,
        hidden_size: int = FLUX_HIDDEN_DIM,
        cross_attention_dim: int = FLUX_JOINT_DIM,
        num_tokens: int = NUM_IP_TOKENS,
        scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale

        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        self.norm_added_k = RMSNorm(128, eps=1e-5, elementwise_affine=False)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        image_emb: torch.Tensor | None = None,
    ):
        batch_size = (
            encoder_hidden_states.shape[0] if encoder_hidden_states is not None else hidden_states.shape[0]
        )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        ip_hidden_states = None
        if image_emb is not None:
            ip_k = self.to_k_ip(image_emb)
            ip_v = self.to_v_ip(image_emb)
            ip_k = ip_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_v = ip_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_k = self.norm_added_k(ip_k)

            ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_k, ip_v, dropout_p=0.0, is_causal=False
            )
            ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            ip_hidden_states = ip_hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            ctx_q = attn.add_q_proj(encoder_hidden_states)
            ctx_k = attn.add_k_proj(encoder_hidden_states)
            ctx_v = attn.add_v_proj(encoder_hidden_states)

            ctx_q = ctx_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ctx_k = ctx_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ctx_v = ctx_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                ctx_q = attn.norm_added_q(ctx_q)
            if attn.norm_added_k is not None:
                ctx_k = attn.norm_added_k(ctx_k)

            query = torch.cat([ctx_q, query], dim=2)
            key = torch.cat([ctx_k, key], dim=2)
            value = torch.cat([ctx_v, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            enc_len = encoder_hidden_states.shape[1]
            encoder_out, hidden_states = hidden_states[:, :enc_len], hidden_states[:, enc_len:]
            if ip_hidden_states is not None:
                hidden_states = hidden_states + self.scale * ip_hidden_states
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_out = attn.to_add_out(encoder_out)
            return hidden_states, encoder_out

        if ip_hidden_states is not None:
            hidden_states = hidden_states + self.scale * ip_hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# Pipeline loading.
# ---------------------------------------------------------------------------

def _install_ip_adapter(pipe: FluxPipeline, device: torch.device | str, dtype: torch.dtype) -> MLPProjModel:
    """Download the InstantX checkpoint, build the image projector, and
    monkey-patch every FLUX attention block with an IP-enabled processor.

    Returns the image projector (needs to be called once per generation to
    produce the ``image_emb`` tensor threaded through ``joint_attention_kwargs``).
    """
    ckpt_path = hf_hub_download(INSTANTX_REPO, INSTANTX_WEIGHTS)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    image_proj_sd = state["image_proj"]
    ip_sd = state["ip_adapter"]

    # Image projector (SigLIP 1152 -> 128 * 4096 tokens).
    image_proj = MLPProjModel()
    image_proj.load_state_dict(image_proj_sd, strict=True)
    image_proj.to(device=device, dtype=dtype).eval()
    for p in image_proj.parameters():
        p.requires_grad_(False)

    # Attention processors — 19 double + 38 single, installed in the order
    # FluxTransformer2DModel exposes them via .attn_processors. The InstantX
    # state dict is keyed 0..56 in that same order.
    transformer = pipe.transformer
    existing = transformer.attn_processors
    proc_names = list(existing.keys())
    assert len(proc_names) == 57, f"expected 57 FLUX attention blocks, got {len(proc_names)}"

    new_procs = {}
    for i, name in enumerate(proc_names):
        proc = IPAFluxAttnProcessor()
        proc.to_k_ip.weight.data.copy_(ip_sd[f"{i}.to_k_ip.weight"])
        proc.to_v_ip.weight.data.copy_(ip_sd[f"{i}.to_v_ip.weight"])
        proc.to(device=device, dtype=dtype).eval()
        for p in proc.parameters():
            p.requires_grad_(False)
        new_procs[name] = proc

    transformer.set_attn_processor(new_procs)
    return image_proj


def load_pipeline(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    default_scale: float = 1.0,
) -> tuple[FluxPipeline, MLPProjModel]:
    """Load FLUX.1-dev with the InstantX IP-Adapter installed on every block.

    Args:
        device:        Target device.
        dtype:         Working dtype (bf16 recommended).
        default_scale: Initial value for every IP attention processor's ``scale``.

    Returns:
        (pipe, image_proj) — pipe is a frozen FluxPipeline with IP-enabled
        attention processors; image_proj is the InstantX MLP that turns a
        (B, 1152) SigLIP embedding into (B, 128, 4096) IP tokens.
    """
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=dtype,
    )
    pipe.to(device)

    for p in pipe.transformer.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder_2.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    image_proj = _install_ip_adapter(pipe, device=device, dtype=dtype)
    set_ip_adapter_scale(pipe, default_scale)

    return pipe, image_proj


def set_ip_adapter_scale(pipe: FluxPipeline, scale: float) -> None:
    """Set the ``scale`` attribute on every IP-enabled attention processor."""
    for proc in pipe.transformer.attn_processors.values():
        if isinstance(proc, IPAFluxAttnProcessor):
            proc.scale = scale


# ---------------------------------------------------------------------------
# VAE helpers — Rust-dataset-specific grayscale-luminance encode/decode.
# ---------------------------------------------------------------------------

def encode_image(pipe: FluxPipeline, pil_image: Image.Image) -> torch.Tensor:
    """VAE-encode a PIL image after converting to grayscale-replicated-3ch.

    The encoder was trained against luminance replicated across the three
    RGB channels; inference must match that distribution.
    """
    x = T.ToTensor()(pil_image.convert("RGB"))
    gray = 0.2989 * x[0:1] + 0.5870 * x[1:2] + 0.1140 * x[2:3]
    x = gray.expand(3, -1, -1).contiguous()
    x = x * 2 - 1
    x = x.unsqueeze(0).to(pipe.vae.device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(x).latent_dist.sample()
        latent = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return latent


def decode_latent(pipe: FluxPipeline, latent: torch.Tensor) -> Image.Image:
    """Decode an unpacked FLUX latent to a PIL image."""
    latent = latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.no_grad():
        image = pipe.vae.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return T.ToPILImage()(image[0].cpu().float())


# ---------------------------------------------------------------------------
# Conditioning helpers.
# ---------------------------------------------------------------------------

def _siglip_to_image_emb(
    image_proj: MLPProjModel,
    siglip_embedding: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Project a cached / neural-decoded SigLIP pooler embedding into IP tokens.

    Accepts shape (D,), (1, D), or (B, D). Returns (B, NUM_IP_TOKENS, FLUX_JOINT_DIM).
    """
    if siglip_embedding.dim() == 1:
        siglip_embedding = siglip_embedding.unsqueeze(0)
    assert siglip_embedding.dim() == 2 and siglip_embedding.shape[-1] == SIGLIP_DIM, (
        f"expected (B, {SIGLIP_DIM}); got {tuple(siglip_embedding.shape)}"
    )
    with torch.no_grad():
        emb = image_proj(siglip_embedding.to(device=device, dtype=dtype))
    return emb


def encode_text_embeds(
    pipe: FluxPipeline,
    prompts: list[str],
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of prompts into FLUX text embeddings using the loaded pipeline.

    Runs the CLIP and T5 encoders that are already resident in ``pipe`` — no
    extra model loading required.  Both encoders are called on CPU-offloaded
    weights if the pipeline is in that mode, so VRAM usage is minimal.

    Args:
        pipe:                The loaded FluxPipeline (must have text_encoder and
                             text_encoder_2 available).
        prompts:             List of N prompt strings.
        max_sequence_length: T5 sequence length (default 512, matching training).

    Returns:
        clip_embeds: (N, 768)        CLIP pooled embeddings, float32, on CPU.
        t5_embeds:   (N, seq, 4096)  T5 sequence embeddings, float32, on CPU.
    """
    device = pipe.device
    clip_list, t5_list = [], []

    for prompt in prompts:
        pe, pooled, _ = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        t5_list.append(pe.cpu().float())        # (1, seq, 4096)
        clip_list.append(pooled.cpu().float())  # (1, 768)

    return torch.cat(clip_list, dim=0), torch.cat(t5_list, dim=0)


# ---------------------------------------------------------------------------
# Generation.
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    pipe: FluxPipeline,
    image_proj: MLPProjModel,
    siglip_embedding: torch.Tensor,
    *,
    prompt: str | None = None,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float = 3.5,
    ip_adapter_scale: float = 1.0,
    seed: int = 0,
    show_progress: bool = True,
) -> Image.Image:
    """Pure-noise txt2img-style generation conditioned on a SigLIP embedding.

    Args:
        pipe:             Loaded FluxPipeline with IP processors installed.
        image_proj:       InstantX MLP that maps SigLIP -> IP tokens.
        siglip_embedding: (D,) or (1, D) SigLIP pooler embedding.
        prompt:           Optional text prompt; None for empty-string neural-only.
        ip_adapter_scale: Scalar gain on the IP attention contribution.
    """
    device = pipe.device
    dtype = torch.bfloat16
    set_ip_adapter_scale(pipe, ip_adapter_scale)

    image_emb = _siglip_to_image_emb(image_proj, siglip_embedding, device, dtype)

    prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt or "",
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )

    latent_channels = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=latent_channels,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=torch.Generator(device).manual_seed(seed),
    )

    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, mu=mu)
    assert timesteps is not None

    guidance = torch.full([1], guidance_scale, device=device, dtype=dtype)

    iterator = tqdm(enumerate(timesteps), total=len(timesteps)) if show_progress else enumerate(timesteps)
    for _, t in iterator:
        t_input = torch.as_tensor(t, device=device).expand(latents.shape[0]).to(latents.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=t_input / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={"image_emb": image_emb},
            return_dict=False,
        )[0]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    return decode_latent(pipe, latents)


@torch.no_grad()
def generate_img2img(
    pipe: FluxPipeline,
    image_proj: MLPProjModel,
    init_image: Image.Image,
    siglip_embedding: torch.Tensor,
    *,
    strength: float = 0.75,
    prompt: str | None = None,
    prompt_embeds: torch.Tensor | None = None,
    pooled_prompt_embeds: torch.Tensor | None = None,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float = 3.5,
    ip_adapter_scale: float = 1.0,
    seed: int = 0,
    aperture_composite: bool = True,
    show_progress: bool = True,
) -> Image.Image:
    """Img2img generation with per-step aperture compositing.

    Starts from the VAE-encoded ``init_image`` noised to ``strength``,
    conditions every FLUX attention block on the SigLIP embedding through
    the installed InstantX IP-Adapter, and (if ``aperture_composite``)
    pastes a freshly re-noised version of the original init latent into
    the untrained ring outside the Rust aperture at every step.

    Text conditioning accepts either a ``prompt`` string (encoded on the fly)
    or pre-computed ``prompt_embeds`` (T5, shape (1, seq, 4096)) and
    ``pooled_prompt_embeds`` (CLIP, shape (1, 768)) from ``encode_text_embeds``.
    Pre-computed embeds take precedence over ``prompt`` when both are supplied.
    """
    device = pipe.device
    dtype = torch.bfloat16
    generator = torch.Generator(device).manual_seed(seed)
    set_ip_adapter_scale(pipe, ip_adapter_scale)

    image_emb = _siglip_to_image_emb(image_proj, siglip_embedding, device, dtype)

    if prompt_embeds is not None:
        # Pre-computed path: move to device/dtype, derive text_ids from seq length.
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
    else:
        prompt_embeds, pooled_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt or "",
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )

    init_resized = init_image.resize((width, height), Image.LANCZOS)
    init_latent = encode_image(pipe, init_resized).to(device=device, dtype=dtype)

    h = 2 * (height // (pipe.vae_scale_factor * 2))
    w = 2 * (width // (pipe.vae_scale_factor * 2))
    latent_channels = init_latent.shape[1]
    init_latent_packed = pipe._pack_latents(init_latent, 1, latent_channels, h, w)
    latent_image_ids = pipe._prepare_latent_image_ids(1, h // 2, w // 2, device, dtype)

    image_seq_len = init_latent_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, mu=mu)

    start_step = int(num_inference_steps * (1.0 - strength))
    timesteps = timesteps[start_step:]
    pipe.scheduler.set_begin_index(start_step)

    noise = torch.randn(init_latent_packed.shape, dtype=dtype, device=device, generator=generator)
    t_start = timesteps[0:1].to(dtype)
    latents = pipe.scheduler.scale_noise(init_latent_packed, t_start, noise)

    aperture_mask = None
    if aperture_composite:
        aperture_mask = load_packed_aperture_mask(image_size=height, device=device, dtype=dtype)
        assert aperture_mask.shape[1] == init_latent_packed.shape[1], (
            f"mask seq {aperture_mask.shape[1]} != latent seq {init_latent_packed.shape[1]}"
        )

    guidance = torch.full([1], guidance_scale, device=device, dtype=dtype)

    iterator = tqdm(enumerate(timesteps), total=len(timesteps)) if show_progress else enumerate(timesteps)
    for _, t in iterator:
        t_input = torch.as_tensor(t, device=device).expand(latents.shape[0]).to(latents.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=t_input / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={"image_emb": image_emb},
            return_dict=False,
        )[0]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if aperture_mask is not None:
            step_idx = (timesteps == t).nonzero(as_tuple=True)[0].item()
            if step_idx + 1 < len(timesteps):
                t_next = timesteps[step_idx + 1 : step_idx + 2].to(dtype)
                init_noisy = pipe.scheduler.scale_noise(init_latent_packed, t_next, noise)
            else:
                init_noisy = init_latent_packed
            latents = aperture_mask * latents + (1.0 - aperture_mask) * init_noisy

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    return decode_latent(pipe, latents)
