"""FLUX.1-dev + IP-Adapter pipeline for neural image reconstruction.

The IP-Adapter projects a SigLIP/neural embedding into a small set of extra
"text" tokens and prepends them to the T5 encoder_hidden_states before the
FLUX transformer forward pass. This lets FLUX's native joint attention attend
to the image conditioning signal without any modification to the attention
processor. Only the projection MLP is trainable; all FLUX weights are frozen.
"""

import torch
import torch.nn as nn
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from config_const import SIGLIP_DIM

# FLUX T5 encoder_hidden_states width — the dim of tokens fed to the transformer.
FLUX_TEXT_DIM = 4096


# ---------------------------------------------------------------------------
# IP-Adapter: project an image embedding into N extra text tokens that get
# prepended to the FLUX T5 encoder_hidden_states.
# ---------------------------------------------------------------------------

class IPAdapterProjection(nn.Module):
    """MLP that maps an image embedding to ``n_tokens`` extra text tokens.

    The output tokens are prepended to the T5 encoder_hidden_states inside
    ``generate()``, so FLUX's existing joint attention attends to them as if
    they were part of the text prompt.

    Args:
        image_dim:  Dimensionality of the SigLIP (or neural) embedding.
        hidden_dim: Internal width of the projection MLP.
        n_tokens:   Number of extra tokens emitted per sample.
        text_dim:   Target token dim — must match FLUX's T5 width (4096).
    """

    def __init__(
        self,
        image_dim: int,
        hidden_dim: int,
        n_tokens: int = 4,
        text_dim: int = FLUX_TEXT_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.text_dim = text_dim
        self.proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_tokens * text_dim),
        )

    def forward(self, image_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embed: (B, image_dim) embedding.

        Returns:
            (B, n_tokens, text_dim) tensor of extra text tokens.
        """
        out = self.proj(image_embed)
        return out.reshape(image_embed.shape[0], self.n_tokens, self.text_dim)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(
    device: str = "cuda",
    ip_adapter_hidden_dim: int = 1024,
    ip_adapter_n_tokens: int = 4,
    ip_adapter_dropout: float = 0.1,
    ip_adapter_checkpoint: str | None = None,
) -> tuple[FluxPipeline, IPAdapterProjection]:
    """Load FLUX.1-dev and an IP-Adapter projection head.

    The FLUX weights are frozen; only the IPAdapterProjection is trainable.

    Args:
        device:                  Target device.
        ip_adapter_hidden_dim:   Hidden width of the IP-Adapter MLP.
        ip_adapter_n_tokens:     Number of extra text tokens to emit.
        ip_adapter_checkpoint:   Optional path to a saved IPAdapterProjection
                                 state dict to resume fine-tuning.

    Returns:
        (pipe, ip_adapter) where pipe is a loaded FluxPipeline (all weights
        frozen) and ip_adapter is the trainable projection module on device.
    """
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)

    # Freeze all FLUX parameters — only IP-Adapter trains.
    for p in pipe.transformer.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)
    for p in pipe.text_encoder_2.parameters():
        p.requires_grad_(False)
    for p in pipe.vae.parameters():
        p.requires_grad_(False)

    ip_adapter = IPAdapterProjection(
        image_dim=SIGLIP_DIM,
        hidden_dim=ip_adapter_hidden_dim,
        n_tokens=ip_adapter_n_tokens,
        dropout=ip_adapter_dropout,
    ).to(device)

    if ip_adapter_checkpoint is not None:
        state = torch.load(ip_adapter_checkpoint, map_location=device, weights_only=True)
        if "ip_adapter_state" in state:
            state = state["ip_adapter_state"]
        ip_adapter.load_state_dict(state)
        print(f"Loaded IP-Adapter from {ip_adapter_checkpoint}")

    return pipe, ip_adapter


# ---------------------------------------------------------------------------
# VAE helpers (from notebook Cell 5)
# ---------------------------------------------------------------------------

def _encode_image(pipe: FluxPipeline, pil_image: Image.Image) -> torch.Tensor:
    transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])
    x = torch.as_tensor(transform(pil_image))
    x = x.unsqueeze(0).to(pipe.vae.device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(x).latent_dist.sample()
        latent = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return latent


def decode_latent(pipe: FluxPipeline, latent: torch.Tensor) -> Image.Image:
    """Decode a packed FLUX latent to a PIL image."""
    latent = latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.no_grad():
        image = pipe.vae.decode(latent).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return T.ToPILImage()(image[0].cpu().float())


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    pipe: FluxPipeline,
    ip_adapter: IPAdapterProjection,
    image_embedding: torch.Tensor,
    prompt: str | None = None,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float = 3.5,
    seed: int = 0,
    ip_adapter_scale: float = 1.0,
    show_progress: bool = True,
) -> Image.Image:
    """Generate an image conditioned on a (neural or SigLIP) image embedding.

    The IP-Adapter projects the image embedding into ``ip_adapter.n_tokens``
    extra text tokens that are prepended to the T5 encoder_hidden_states, so
    FLUX's joint attention attends to them natively.  A text prompt can
    optionally supplement the neural signal.

    Args:
        pipe:               Loaded FluxPipeline (weights frozen).
        ip_adapter:         Trained IPAdapterProjection.
        image_embedding:    (1, D) or (D,) SigLIP or neural embedding tensor.
        prompt:             Optional text prompt.  Pass None for neural-only.
        height:             Output image height in pixels.
        width:              Output image width in pixels.
        num_inference_steps: Number of denoising steps.
        guidance_scale:     FLUX distilled guidance scale (embedded in forward).
        seed:               RNG seed for reproducibility.
        ip_adapter_scale:   Multiplier on the injected IP-Adapter tokens.
        show_progress:      Show tqdm progress bar.

    Returns:
        Reconstructed PIL image.
    """
    device = pipe.device
    dtype = torch.bfloat16

    # Normalise embedding shape to (1, D)
    if image_embedding.dim() == 1:
        image_embedding = image_embedding.unsqueeze(0)
    image_embedding = image_embedding.to(device, dtype=dtype)

    # --- 4. IP-Adapter: project image embedding into extra text tokens ------
    ip_adapter.eval()
    ip_dtype = next(ip_adapter.parameters()).dtype
    ip_tokens = ip_adapter(image_embedding.to(ip_dtype)).to(dtype) * ip_adapter_scale
    # (1, n_tokens, 4096)
    ip_ids = torch.zeros(ip_tokens.shape[1], 3, device=device, dtype=dtype)

    # --- 1. Text encoding ---------------------------------------------------
    # When prompt is None we match the training regime exactly: only IP tokens
    # are passed as encoder_hidden_states, with no T5 sequence.  The model was
    # trained without T5 tokens, so injecting an empty-prompt T5 sequence at
    # inference would shift the conditioning distribution.
    if prompt is None:
        encoder_hidden_states = ip_tokens          # (1, n_tokens, 4096)
        text_ids = ip_ids                          # (n_tokens, 3)
        _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt="",
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
    else:
        prompt_embeds, pooled_prompt_embeds, t5_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        # Prepend IP tokens to T5 tokens.
        encoder_hidden_states = torch.cat([ip_tokens, prompt_embeds], dim=1)
        text_ids = torch.cat([ip_ids, t5_ids], dim=0)

    # --- 2. Prepare latents --------------------------------------------------
    latent_channels = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=latent_channels,
        height=height,
        width=width,
        dtype=encoder_hidden_states.dtype,
        device=device,
        generator=torch.Generator(device).manual_seed(seed),
    )

    # --- 3. Timestep schedule ------------------------------------------------
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, mu=mu
    )
    assert timesteps is not None

    guidance = torch.full([1], guidance_scale, device=device, dtype=dtype)

    # --- 5. Denoising loop ---------------------------------------------------
    iterator = tqdm(enumerate(timesteps), total=len(timesteps)) if show_progress else enumerate(timesteps)
    for _, t in iterator:
        t_input = torch.as_tensor(t, device=device).expand(latents.shape[0]).to(latents.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=t_input / 1000,
            guidance=guidance,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # --- 6. Decode -----------------------------------------------------------
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    return decode_latent(pipe, latents)


# ---------------------------------------------------------------------------
# img2img: denoise from a noised stimulus latent
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_img2img(
    pipe: FluxPipeline,
    ip_adapter: IPAdapterProjection,
    init_image: Image.Image,
    image_embedding: torch.Tensor,
    strength: float = 0.75,
    prompt: str | None = None,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float = 3.5,
    cfg_scale: float = 1.0,
    seed: int = 0,
    ip_adapter_scale: float = 1.0,
    show_progress: bool = True,
) -> Image.Image:
    """img2img generation: start from a noised stimulus latent instead of pure noise.

    Encodes ``init_image`` with the VAE, packs the latent into FLUX's sequence
    format, adds noise up to the timestep corresponding to ``strength``, then
    runs the remaining denoising steps conditioned on ``image_embedding``.

    Two-pass classifier-free guidance is applied over the IP-Adapter tokens:
    each denoising step runs one unconditional forward (null embedding) and one
    conditional forward, then combines them as:
        pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
    Set ``cfg_scale=1.0`` to disable (conditional only, one pass per step).

    Args:
        pipe:                Loaded FluxPipeline (weights frozen).
        ip_adapter:          Trained IPAdapterProjection.
        init_image:          PIL image used as the starting latent (resized to
                             height × width before encoding).
        image_embedding:     (1, D) or (D,) SigLIP or neural embedding tensor.
        strength:            Noise strength in [0, 1].  1.0 = pure noise (same
                             as generate()); 0.0 = no denoising (VAE round-trip).
        prompt:              Optional text prompt.
        height:              Output image height in pixels.
        width:               Output image width in pixels.
        num_inference_steps: Total denoising steps in the full schedule.
        guidance_scale:      FLUX distilled guidance scale (embedded in forward).
        cfg_scale:           IP-Adapter CFG scale.  Values > 1 strengthen the
                             embedding signal; 1.0 = conditional pass only.
        seed:                RNG seed.
        ip_adapter_scale:    Multiplier on the injected IP-Adapter tokens.
        show_progress:       Show tqdm progress bar.

    Returns:
        Reconstructed PIL image.
    """
    device = pipe.device
    dtype  = torch.bfloat16
    generator = torch.Generator(device).manual_seed(seed)

    # Normalise embedding shape to (1, D)
    if image_embedding.dim() == 1:
        image_embedding = image_embedding.unsqueeze(0)
    image_embedding = image_embedding.to(device, dtype=dtype)

    # --- IP-Adapter tokens --------------------------------------------------
    ip_adapter.eval()
    ip_dtype = next(ip_adapter.parameters()).dtype

    cond_tokens = ip_adapter(image_embedding.to(ip_dtype)).to(dtype) * ip_adapter_scale
    # (1, n_tokens, 4096)

    # Unconditional tokens: project the null (zero) embedding, matching the
    # training-time dropout null condition.
    null_embed   = torch.zeros_like(image_embedding.to(ip_dtype))
    uncond_tokens = ip_adapter(null_embed).to(dtype) * ip_adapter_scale
    # (1, n_tokens, 4096)

    ip_ids = torch.zeros(cond_tokens.shape[1], 3, device=device, dtype=dtype)

    # --- Text encoding -------------------------------------------------------
    if prompt is None:
        cond_hidden   = cond_tokens    # (1, n_tokens, 4096)
        uncond_hidden = uncond_tokens
        text_ids      = ip_ids
        _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt="", prompt_2=None, device=device,
            num_images_per_prompt=1, max_sequence_length=512,
        )
    else:
        prompt_embeds, pooled_prompt_embeds, t5_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, device=device,
            num_images_per_prompt=1, max_sequence_length=512,
        )
        cond_hidden   = torch.cat([cond_tokens,   prompt_embeds], dim=1)
        uncond_hidden = torch.cat([uncond_tokens,  prompt_embeds], dim=1)
        text_ids      = torch.cat([ip_ids, t5_ids], dim=0)

    # --- Encode init image into VAE latent space ----------------------------
    init_resized = init_image.resize((width, height), Image.LANCZOS)
    init_latent  = _encode_image(pipe, init_resized).to(device, dtype=dtype)

    # prepare_latents skips packing when given a pre-encoded latent, so we
    # must pack manually and compute latent_image_ids with the same arithmetic
    # the txt2img path uses: height/width after vae_scale_factor*2 reduction.
    h = 2 * (height // (pipe.vae_scale_factor * 2))
    w = 2 * (width  // (pipe.vae_scale_factor * 2))
    latent_channels = init_latent.shape[1]
    init_latent_packed = pipe._pack_latents(init_latent, 1, latent_channels, h, w)
    latent_image_ids   = pipe._prepare_latent_image_ids(1, h // 2, w // 2, device, dtype)

    # --- Timestep schedule --------------------------------------------------
    image_seq_len = init_latent_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.base_image_seq_len,
        pipe.scheduler.config.max_image_seq_len,
        pipe.scheduler.config.base_shift,
        pipe.scheduler.config.max_shift,
    )
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, mu=mu)

    # Determine the start index based on strength
    start_step = int(num_inference_steps * (1.0 - strength))
    timesteps  = timesteps[start_step:]

    # Tell the scheduler which step we're beginning from so scale_noise can
    # look up the correct sigma index in its internal schedule.
    pipe.scheduler.set_begin_index(start_step)

    # Add noise to the init latent at the first active timestep.
    # scale_noise expects the raw timestep value (not divided by 1000).
    noise   = torch.randn(init_latent_packed.shape, dtype=dtype, device=device, generator=generator)
    t_start = timesteps[0:1].to(dtype)
    latents = pipe.scheduler.scale_noise(init_latent_packed, t_start, noise)

    guidance = torch.full([1], guidance_scale, device=device, dtype=dtype)

    # --- Denoising loop ------------------------------------------------------
    do_cfg = cfg_scale != 1.0
    iterator = tqdm(enumerate(timesteps), total=len(timesteps)) if show_progress else enumerate(timesteps)
    for _, t in iterator:
        t_input = torch.as_tensor(t, device=device).expand(latents.shape[0]).to(latents.dtype)

        # Conditional forward pass
        pred_cond = pipe.transformer(
            hidden_states=latents,
            timestep=t_input / 1000,
            guidance=guidance,
            encoder_hidden_states=cond_hidden,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if do_cfg:
            # Unconditional forward pass with null IP tokens
            pred_uncond = pipe.transformer(
                hidden_states=latents,
                timestep=t_input / 1000,
                guidance=guidance,
                encoder_hidden_states=uncond_hidden,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            noise_pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        else:
            noise_pred = pred_cond

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # --- Decode --------------------------------------------------------------
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    return decode_latent(pipe, latents)


# ---------------------------------------------------------------------------
# Batch generation helper
# ---------------------------------------------------------------------------

def generate_batch(
    pipe: FluxPipeline,
    ip_adapter: IPAdapterProjection,
    image_embeddings: torch.Tensor,
    prompts: list[str | None] | None = None,
    seeds: list[int] | None = None,
    **generate_kwargs,
) -> list[Image.Image]:
    """Generate one image per embedding in a batch.

    Args:
        pipe:             Loaded FluxPipeline.
        ip_adapter:       Trained IPAdapterProjection.
        image_embeddings: (N, D) tensor of embeddings.
        prompts:          Optional list of N prompts.  None for neural-only.
        seeds:            Optional list of N integer seeds.

    Returns:
        List of N PIL images.
    """
    n = image_embeddings.shape[0]
    resolved_prompts: list[str | None] = prompts if prompts is not None else [None] * n
    resolved_seeds: list[int] = seeds if seeds is not None else list(range(n))

    images = []
    for i in range(n):
        img = generate(
            pipe,
            ip_adapter,
            image_embeddings[i],
            prompt=resolved_prompts[i],
            seed=resolved_seeds[i],
            show_progress=(i == 0),
            **generate_kwargs,
        )
        images.append(img)
    return images
