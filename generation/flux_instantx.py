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

from config_const import (
    SIGLIP_DIM,
    INSTANTX_REPO, INSTANTX_WEIGHTS,
    FLUX_JOINT_DIM, FLUX_HIDDEN_DIM, NUM_IP_TOKENS,
)
from generation.aperture import load_packed_aperture_mask


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
def invert_image(
    pipe: FluxPipeline,
    init_latent_packed: torch.Tensor,
    latent_image_ids: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_embeds: torch.Tensor,
    text_ids: torch.Tensor,
    *,
    num_inversion_steps: int = 28,
    gamma: float = 0.5,
    guidance_scale: float = 3.5,
    seed: int = 0,
) -> torch.Tensor:
    """Forward ODE inversion (RF-Inversion): maps init_latent -> inverted_latent.

    Follows diffusers' `pipeline_flux_rf_inversion.py`:

        u_hat = u_transformer + gamma * (u_cond_to_noise - u_transformer)
        u_cond_to_noise = (y_1 - Y_t) / (1 - t_i)          # deterministic pull toward y_1
        Y_t_next = Y_t + u_hat * (sigmas[i] - sigmas[i+1])

    At i=0, Y_t = y_0 (clean image latent) and sigmas[0] ≈ 1.
    After N-1 steps, Y_t ≈ y_1 (structured noise at sigma ≈ 1/N).

    Args:
        init_latent_packed: (1, seq, C) packed VAE latent on device.
        gamma: Balance between editability (0.0, pure transformer ODE) and
            faithfulness (1.0, deterministic linear interp to y_1).
    """
    device = init_latent_packed.device
    dtype = init_latent_packed.dtype
    guidance = torch.full([1], guidance_scale, device=device, dtype=dtype)

    image_seq_len = init_latent_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    retrieve_timesteps(pipe.scheduler, num_inversion_steps, device, mu=mu)
    # scheduler.sigmas: length N+1, decreasing from ~1 down to 0 (generation order).
    sigmas = pipe.scheduler.sigmas.to(device=device, dtype=dtype)

    generator = torch.Generator(device).manual_seed(seed)
    y_1 = torch.randn(init_latent_packed.shape, dtype=dtype, device=device, generator=generator)

    Y_t = init_latent_packed.clone()
    N = num_inversion_steps

    for i in range(N):
        t_i = torch.tensor(i / N, device=device, dtype=dtype)
        t_input = t_i.expand(Y_t.shape[0])

        u_t = pipe.transformer(
            hidden_states=Y_t,
            timestep=t_input,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

        if gamma > 0.0:
            u_t_cond = (y_1 - Y_t) / (1.0 - t_i + 1e-8)
            u_hat = u_t + gamma * (u_t_cond - u_t)
        else:
            u_hat = u_t

        # sigmas[i] > sigmas[i+1]; step is positive, pushing Y_t toward noise.
        Y_t = Y_t + u_hat * (sigmas[i] - sigmas[i + 1])

    return Y_t


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
    ip_adapter_schedule: dict[str, float] | None = None,
    use_rf_inversion: bool = False,
    rf_gamma: float = 0.0,
    rf_eta: float = 0.15,
    rf_inversion_steps: int | None = None,
    rf_noise_blend: float = 0.0,
    seed: int = 0,
    aperture_composite: bool = True,
    aperture_mask: torch.Tensor | None = None,
    init_latent: torch.Tensor | None = None,
    guidance_latent: torch.Tensor | None = None,
    guidance_eta: float = 0.15,
    guidance_mask: torch.Tensor | None = None,
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

    Args:
        ip_adapter_schedule: Optional dict with keys "high"/"mid"/"low" mapping
            timestep fractions (t/1000 > 0.7 / 0.4–0.7 / ≤0.4) to IP scale
            values. When set, overrides ``ip_adapter_scale`` dynamically each
            step. ``ip_adapter_scale`` is used as a fallback for missing keys.
        use_rf_inversion: If True, replace simple scale_noise init with
            RF-Inversion (forward ODE) for a lossless structured noise latent.
        rf_gamma: Inversion gamma — 0.5 balances faithfulness vs editability.
        rf_eta: Generation eta — controls image-faithful correction strength.
        rf_inversion_steps: Steps for the inversion ODE (defaults to
            num_inference_steps).
        rf_noise_blend: Linear blend of the inverted latent with fresh Gaussian
            noise before denoising. 0.0 = pure inverted (tries to reconstruct
            original), 1.0 = pure noise (RF structure discarded). Intermediate
            values keep the structural skeleton while giving the IP-adapter
            room to repaint content instead of fighting a deterministic ODE.
        guidance_latent: Optional packed or unpacked latent to use as a
            per-step RF correction target (``y_0``). When provided, every
            denoising step is steered toward this latent with strength
            ``guidance_eta``, independently of ``use_rf_inversion``. Pass the
            neural-predicted latent here to guide generation without using it
            as the noise init. Must match the spatial resolution implied by
            ``height``/``width`` (i.e. shape ``(1, 16, H/8, W/8)``).
        guidance_eta: Strength of the per-step RF correction toward
            ``guidance_latent``. 0.0 = no guidance, 1.0 = full correction.
        guidance_mask: Optional packed-latent-resolution mask ``(1, seq_len, 1)``
            that spatially gates the RF correction. When provided, the correction
            is applied only at positions where the mask is non-zero (i.e. inside
            the projected object region). Outside positions receive zero correction
            and evolve freely. Shape must match the packed latent sequence length.
            Build with ``generation.aperture.build_object_region_mask``.
    """
    device = pipe.device
    dtype = torch.bfloat16
    generator = torch.Generator(device).manual_seed(seed)

    if ip_adapter_schedule is None:
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

    if init_latent is not None:
        init_latent = init_latent.to(device=device, dtype=dtype)
    else:
        init_resized = init_image.resize((width, height), Image.LANCZOS)
        init_latent = encode_image(pipe, init_resized).to(device=device, dtype=dtype)

    h = 2 * (height // (pipe.vae_scale_factor * 2))
    w = 2 * (width // (pipe.vae_scale_factor * 2))
    latent_channels = init_latent.shape[1]
    init_latent_packed = pipe._pack_latents(init_latent, 1, latent_channels, h, w)
    latent_image_ids = pipe._prepare_latent_image_ids(1, h // 2, w // 2, device, dtype)

    if guidance_latent is not None:
        guidance_latent = guidance_latent.to(device=device, dtype=dtype)
        guidance_latent_packed = pipe._pack_latents(guidance_latent, 1, guidance_latent.shape[1], h, w)

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

    # --- Init latent preparation ---
    if use_rf_inversion:
        latents = invert_image(
            pipe, init_latent_packed, latent_image_ids,
            prompt_embeds, pooled_embeds, text_ids,
            num_inversion_steps=rf_inversion_steps or num_inference_steps,
            gamma=rf_gamma,
            guidance_scale=guidance_scale,
        )
        if rf_noise_blend > 0.0:
            blend_noise = torch.randn(
                latents.shape, dtype=dtype, device=device, generator=generator,
            )
            latents = (1.0 - rf_noise_blend) * latents + rf_noise_blend * blend_noise
        image_latents = init_latent_packed  # y_0 for RF correction term
        # invert_image calls retrieve_timesteps internally, resetting pipe.scheduler
        # sigmas and begin_index — restore generation schedule state.
        retrieve_timesteps(pipe.scheduler, num_inference_steps, device, mu=mu)
        pipe.scheduler.set_begin_index(start_step)
    else:
        noise = torch.randn(init_latent_packed.shape, dtype=dtype, device=device, generator=generator)
        t_start = timesteps[0:1].to(dtype)
        latents = pipe.scheduler.scale_noise(init_latent_packed, t_start, noise)
        image_latents = None

    if aperture_mask is not None:
        aperture_mask = aperture_mask.to(device=device, dtype=dtype)
    elif aperture_composite:
        aperture_mask = load_packed_aperture_mask(image_size=height, device=device, dtype=dtype)
    if aperture_mask is not None:
        assert aperture_mask.shape[1] == init_latent_packed.shape[1], (
            f"mask seq {aperture_mask.shape[1]} != latent seq {init_latent_packed.shape[1]}"
        )

    # Aperture compositing re-noises the init latent at each step; RF path needs
    # a fresh noise tensor since scale_noise is not called during init.
    if use_rf_inversion:
        noise = torch.randn(init_latent_packed.shape, dtype=dtype, device=device, generator=generator)

    guidance = torch.full([1], guidance_scale, device=device, dtype=dtype)

    iterator = tqdm(enumerate(timesteps), total=len(timesteps)) if show_progress else enumerate(timesteps)
    for step_idx, t in iterator:
        # Timestep-aware IP scale
        if ip_adapter_schedule is not None:
            t_frac = t.item() / 1000.0
            if t_frac > 0.7:
                scale = ip_adapter_schedule.get("high", ip_adapter_scale)
            elif t_frac > 0.4:
                scale = ip_adapter_schedule.get("mid", ip_adapter_scale)
            else:
                scale = ip_adapter_schedule.get("low", ip_adapter_scale)
            set_ip_adapter_scale(pipe, scale)

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

        sigma_curr = t.to(dtype) / 1000.0
        sigma_next = (timesteps[step_idx + 1].to(dtype) / 1000.0
                      if step_idx + 1 < len(timesteps) else torch.zeros(1, device=device, dtype=dtype))

        if use_rf_inversion or guidance_latent is not None:
            # RF-style Euler step. The correction target is:
            #   - guidance_latent (neural prediction) when provided — steers toward it
            #   - image_latents (GT init) when only use_rf_inversion is set — reconstructs original
            # When both are set, RF-inversion seeds the structured noise init and guidance_latent
            # provides the per-step correction target, combining structured init with neural steering.
            v_t = -noise_pred
            if guidance_latent is not None:
                correction_target = guidance_latent_packed
                eta = guidance_eta
            else:
                correction_target = image_latents
                eta = rf_eta
            v_t_cond = (correction_target - latents) / (sigma_curr + 1e-3)
            correction = eta * (v_t_cond - v_t)
            if guidance_mask is not None:
                gm = guidance_mask.to(device=latents.device, dtype=latents.dtype)
                correction = gm * correction
            v_hat = v_t + correction
            latents = latents + v_hat * (sigma_curr - sigma_next)
        else:
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if aperture_mask is not None:
            if step_idx + 1 < len(timesteps):
                t_next = timesteps[step_idx + 1 : step_idx + 2].to(dtype)
                init_noisy = pipe.scheduler.scale_noise(init_latent_packed, t_next, noise)
            else:
                init_noisy = init_latent_packed
            latents = aperture_mask * latents + (1.0 - aperture_mask) * init_noisy

    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    return decode_latent(pipe, latents)
