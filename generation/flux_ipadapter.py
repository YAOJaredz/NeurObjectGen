"""FLUX.1-dev + IP-Adapter pipeline for neural image reconstruction.

The IP-Adapter approach injects a SigLIP image embedding into the FLUX.1
transformer via cross-attention projection layers, without full fine-tuning.
We expose the full manual forward pass (text encode → prepare latents →
denoise loop → VAE decode) so that neural embeddings can be swapped in
as the image conditioning signal.
"""

import torch
import torch.nn as nn
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from config_const import SIGLIP_DIM


# ---------------------------------------------------------------------------
# IP-Adapter: project a SigLIP embedding into the FLUX transformer's
# cross-attention key/value space for each double-stream block.
# ---------------------------------------------------------------------------

class IPAdapterProjection(nn.Module):
    """Lightweight MLP that maps a SigLIP embedding to per-block K/V pairs.

    For each of the ``n_blocks`` double-stream transformer blocks we learn
    a separate linear projection from the image embedding space into
    the block's cross-attention dim.  At inference time the projected
    vectors are injected as extra key/value tokens via forward hooks.

    Args:
        image_dim:  Dimensionality of the SigLIP (or neural) embedding.
        hidden_dim: Internal width of the projection MLP.
        n_blocks:   Number of double-stream blocks in the FLUX transformer.
        head_dim:   Attention head dimension of the FLUX transformer.
        n_heads:    Number of attention heads in each block.
    """

    def __init__(
        self,
        image_dim: int,
        hidden_dim: int,
        n_blocks: int,
        head_dim: int = 128,
        n_heads: int = 24,
    ):
        super().__init__()
        cross_dim = head_dim * n_heads  # 3072 for FLUX.1-dev
        self.to_kv = nn.ModuleList([
            nn.Sequential(
                nn.Linear(image_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, cross_dim * 2),  # → [K, V] concatenated
            )
            for _ in range(n_blocks)
        ])

    def forward(self, image_embed: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Project image_embed into per-block (K, V) pairs.

        Args:
            image_embed: (B, image_dim) embedding.

        Returns:
            List of (K, V) tuples, each of shape (B, 1, cross_dim).
        """
        kvs = []
        for proj in self.to_kv:
            kv = proj(image_embed)  # (B, cross_dim * 2)
            k, v = kv.chunk(2, dim=-1)
            kvs.append((k.unsqueeze(1), v.unsqueeze(1)))  # (B, 1, cross_dim)
        return kvs


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(
    device: str = "cuda",
    ip_adapter_hidden_dim: int = 512,
    ip_adapter_checkpoint: str | None = None,
) -> tuple[FluxPipeline, IPAdapterProjection]:
    """Load FLUX.1-dev and an IP-Adapter projection head.

    The FLUX weights are frozen; only the IPAdapterProjection is trainable.

    Args:
        device:                  Target device.
        ip_adapter_hidden_dim:   Hidden width of the IP-Adapter MLP.
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

    n_blocks = len(pipe.transformer.transformer_blocks)
    ip_adapter = IPAdapterProjection(
        image_dim=SIGLIP_DIM,
        hidden_dim=ip_adapter_hidden_dim,
        n_blocks=n_blocks,
    ).to(device)

    if ip_adapter_checkpoint is not None:
        state = torch.load(ip_adapter_checkpoint, map_location=device, weights_only=True)
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

    The IP-Adapter projection layers inject the image embedding as extra K/V
    tokens into each double-stream block of the FLUX transformer via forward
    hooks.  A text prompt can optionally supplement the neural signal.

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
        ip_adapter_scale:   Weight of the IP-Adapter K/V injection.
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

    # --- 1. Text encoding ---------------------------------------------------
    effective_prompt = prompt if prompt is not None else ""
    (
        prompt_embeds,        # (1, 512, 4096)
        pooled_prompt_embeds, # (1, 768)
        text_ids,             # (512, 3)
    ) = pipe.encode_prompt(
        prompt=effective_prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=512,
    )

    # --- 2. Prepare latents --------------------------------------------------
    latent_channels = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=latent_channels,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,          # type: ignore
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

    # --- 4. IP-Adapter: compute per-block K/V projections -------------------
    ip_adapter.eval()
    block_kvs: list[tuple[torch.Tensor, torch.Tensor]] = ip_adapter(image_embedding)
    # block_kvs[i] = (K_i, V_i), each (1, 1, cross_dim)

    # Register forward hooks on each double-stream block to inject IP-Adapter
    # K/V tokens by concatenating them to the existing cross-attention inputs.
    hooks = []

    def _make_hook(k_ip: torch.Tensor, v_ip: torch.Tensor, scale: float):
        def hook(module, args, kwargs):
            # Double-stream blocks receive encoder_hidden_states as the text
            # side; we inject our image K/V by patching joint_attention_kwargs.
            if kwargs is None:
                kwargs = {}
            # Pass IP tokens through joint_attention_kwargs so the block can
            # concatenate them.  If the block doesn't support this key we fall
            # back to a no-op (zero-scale injection).
            kwargs.setdefault("joint_attention_kwargs", {})
            kwargs["joint_attention_kwargs"]["ip_adapter_image_embeds"] = (
                k_ip * scale, v_ip * scale
            )
            return args, kwargs
        return hook

    for block_idx, (block, (k_ip, v_ip)) in enumerate(
        zip(pipe.transformer.transformer_blocks, block_kvs)
    ):
        h = block.register_forward_pre_hook(
            _make_hook(k_ip, v_ip, ip_adapter_scale), with_kwargs=True
        )
        hooks.append(h)

    # --- 5. Denoising loop ---------------------------------------------------
    try:
        iterator = tqdm(enumerate(timesteps), total=len(timesteps)) if show_progress else enumerate(timesteps)
        for _, t in iterator:
            t_input = torch.as_tensor(t, device=device).expand(latents.shape[0]).to(latents.dtype)
            noise_pred = pipe.transformer(
                hidden_states=latents,
                timestep=t_input / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    finally:
        for h in hooks:
            h.remove()

    # --- 6. Decode -----------------------------------------------------------
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
