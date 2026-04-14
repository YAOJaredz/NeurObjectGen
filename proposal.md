# Neural Image Reconstruction from Primate IT Cortex via Conditioned Generative Models

**Jared Yao, Columbia University**

---

## 1. Background

The primate ventral visual stream transforms retinal input into a rich representation in inferotemporal (IT) cortex that supports robust object recognition [Rust and DiCarlo, 2010]. A powerful test of how well we understand this representation is whether we can invert it—reconstructing the perceived image directly from neural activity. This also poses challenges modern generative models, requiring them to produce faithful reconstructions when conditioned on noisy, low-dimensional signals far removed from their training distribution of text and image embeddings.

Recent work has achieved striking reconstructions by aligning brain responses with the latent spaces of pretrained generative models, first from fMRI [Takagi and Nishimoto, 2023, Scotti et al., 2024] and more recently from intracortical electrophysiology [Ciferri et al., 2026]. However, these successes rely on massive datasets (e.g., 22,000 images, 1024 channels) atypical of most primate neurophysiology and sidestep the question of whether reconstruction is possible in the data-limited regime.

We address this using Neuropixels 1.0 recordings (384 channels per session) from primate IT cortex, collected across multiple sessions during viewing of the 300 naturalistic image stimulus set introduced by Rust and DiCarlo [2010]. Pooling across sessions yields a large pseudo-population, enabling high-dimensional decoding despite the modest stimulus count.

---

## 2. Problem Formulation

We collect neural responses with Neuropixels 1.0 probes across multiple sessions from primate IT cortex during passive viewing of 300 naturalistic images [Rust and DiCarlo, 2010], and construct a pseudo-population by pooling across sessions. The goal is to reconstruct the perceived image from the population activity vector.

Formally, we seek a mapping **f : r → z**, where **r ∈ R^(N×T)** is the neural response (N channels from the pooled pseudo-population, with 384 channels per session and T time points after stimulus presentation) and **z** is a latent representation that conditions a pretrained image generation model to produce a reconstruction **x̂** of the original stimulus **x**.

---

## 3. Approach

### Stage 1 — Neural Encoder

We train a regularized encoder that maps pseudo-population firing rate vectors to the SigLIP image embedding space. Given the ratio of stimuli (300) to channels (potentially thousands), we compare three architectures:

- **Ridge regression** — baseline, no temporal structure
- **Low-rank bottleneck MLP** — flattens the (N × T) response and projects through a small bottleneck
- **Temporal LSTM / Transformer** — operates over the T time bins, treating each T-step as a token over N-dimensional neural activity

All encoders are trained with an **InfoNCE contrastive loss** that aligns predicted embeddings with frozen SigLIP embeddings of the presented stimuli, improving data efficiency over direct regression. The best architecture is selected by 2AFC identification accuracy on a held-out validation set.

### Stage 2 — Pretrained IP-Adapter (InstantX/FLUX.1-dev-IP-Adapter)

The predicted SigLIP embedding conditions a pretrained **FLUX.1-dev** model via the **pretrained InstantX/FLUX.1-dev-IP-Adapter**, which already ships a SigLIP-SO400M image encoder matching our cached embeddings. This replaces an earlier plan to train a custom projection module from scratch: with only 200 training stimuli, training a bespoke adapter that has never seen neural-decoded inputs was a poor use of data relative to adopting a community-validated adapter that already implements a real cross-attention path via a learned resampler and per-block IP key/value projections on all 57 FLUX transformer blocks.

Because diffusers 0.37.1's generic `load_ip_adapter` loader does not accept InstantX's checkpoint layout for FLUX, we vendor the minimal adapter bits directly: an `MLPProjModel` (SigLIP 1152 → 128 × 4096 image tokens) and an `IPAFluxAttnProcessor` installed on all 57 attention blocks (19 double-stream + 38 single-stream). Image conditioning is threaded through `pipe(..., joint_attention_kwargs={"image_emb": image_emb, "scale": ip_adapter_scale})` so each block's attention output receives an additive `scale · ip_attn` contribution in parallel with standard joint attention. **No adapter training is performed** — Stage 2 is now a pure inference component, and the only trained module in the pipeline is the Stage 1 neural encoder.

Rust-dataset-specific loop logic is preserved around the pretrained adapter rather than baked into it:

- **Grayscale-luminance VAE encoding** of Rust stimuli, matching the achromatic presentation distribution
- **Per-step aperture compositing** during img2img denoising: at every scheduler step, latents inside the circular stimulus aperture are updated by the adapter-conditioned transformer while latents outside the aperture (and inside the central fixation square) are pinned to a re-noised copy of the original init latent, using a soft mask downsampled from pixel space to FLUX's packed-latent grid (each position covers a 16×16 pixel block)
- **Init-latent conditioning** via flow-matching `scale_noise(strength)`, so at inference we can trade off how much the adapter repaints versus how much of the stimulus structure leaks through from the init

This preserves everything the earlier custom-adapter plan used the aperture mask for (keeping generations anchored to the trained visual region), while moving the conditioning mechanism itself onto pretrained weights.

### Inference Pipeline

At test time the single trained stage is chained into the pretrained generator:

```
neural r (N×T) ──encoder──▶ ẑ ∈ R^1152 ──MLPProj──▶ image_emb (128×4096)
                                                              │
                          optional text prompt ──T5/CLIP──▶ encoder_hidden_states
                                                              │
                                                              ▼
                                     FLUX.1-dev + per-block IPAFluxAttnProcessor
                                                              │
                                                              ▼
                                                        reconstruction x̂
```

The neural encoder predicts a SigLIP-aligned embedding from held-out neural responses; the vendored InstantX `MLPProjModel` expands it into 128 image tokens at FLUX's cross-attention dimension; and each transformer block mixes those image tokens with standard T5/CLIP text conditioning through parallel IP key/value attention. A starting Rust stimulus (or a gray plate) is VAE-encoded, noised to a configurable `strength`, and denoised under adapter conditioning with per-step aperture compositing, so the untrained ring of the canvas stays pinned to the original while the trained aperture is repainted from the neural signal.

### Text-Conditioned Hybrid

Because 300 images may under-constrain a purely neural decoder, we implement a hybrid condition where short text captions (from BLIP-2 auto-captioning) supplement the neural embedding. The InstantX adapter makes this natural: FLUX's standard T5/CLIP text path and the IP image path are additive inside each attention block, so switching conditions amounts to toggling the `prompt` argument and the `ip_adapter_scale` knob without changing code paths. We systematically compare **neural-only** (empty prompt, ip_scale>0), **text-only** (caption, ip_scale=0), and **neural+text** (caption, ip_scale>0) to isolate the contribution of neural signals beyond what text provides.

### Sanity Ceiling

Before evaluating neural-decoded embeddings, we measure an **upper-bound recovery test**: feed the pipeline the ground-truth SigLIP embedding and a noised version of the ground-truth VAE latent, then sweep `strength` and `ip_adapter_scale` and measure luminance MSE inside the trained aperture. This isolates adapter+compositing behavior from neural-decoder quality — if this ceiling is low, any downstream failure lives in Stage 1 rather than the generator.

---

## 4. Deliverables & Evaluation

**Deliverables:**
1. A trained neural encoder mapping IT pseudo-population activity to SigLIP space
2. An inference wrapper around the pretrained InstantX/FLUX.1-dev-IP-Adapter with Rust-specific grayscale VAE encoding and per-step aperture compositing
3. A conditioned image generation pipeline producing reconstructions from held-out neural responses
4. A sanity-ceiling recovery analysis using ground-truth embeddings + init latents to bound adapter-side error
5. A systematic comparison across decoding conditions (neural-only, text-only, neural+text) with ablations

**Evaluation:** We will use a fixed 200/50/50 train/val/test split over the 300 image stimulus set. Metrics include:
- **Semantic similarity** — embedding cosine similarity
- **Perceptual quality** — SSIM
- **Identification accuracy** — 2AFC on held-out trials

**Success criteria:**
- (a) Neural-only reconstructions achieve above-chance 2AFC identification
- (b) Neural+text significantly outperforms text-only, demonstrating that neural signals carry stimulus-specific information beyond category labels

---

## 5. Compute Plan

**Stage 1 (neural encoder):** Lightweight — ridge regression, MLP, or LSTM on neural time series. Runs in minutes on a local machine or Colab. Hyperparameter sweeps run via SLURM on the Issa Lab cluster inside an Apptainer container.

**Stage 2 (IP-Adapter):** Inference only — no gradient updates. Requires GPU memory for the frozen FLUX.1-dev transformer (~24 GB bf16) plus the pretrained InstantX adapter weights (small, negligible). Switching from a custom-trained adapter to pretrained InstantX eliminates a full training loop and its hyperparameter sweeps from the compute budget.

**Inference:** Generating 300 images through the full pipeline takes on the order of a few GPU-hours. Total compute requirements remain modest relative to the available resources.

---

## References

- Y. Takagi and S. Nishimoto. High-resolution image reconstruction with latent diffusion models from human brain activity. In *CVPR*, 2023.
- P. Scotti, M. Triparity, C. K. T. Villanueva, et al. MindEye2: Shared-subject models enable fMRI-to-image with 1 hour of data. *arXiv:2403.11207*, 2024.
- M. Ciferri, M. Ferrante, and N. Toschi. Simple models, rich representations: Visual decoding from primate intracortical neural signals. *arXiv:2601.11108*, 2026.
- N. C. Rust and J. J. DiCarlo. Selectivity and tolerance ("invariance") both increase as visual information propagates from cortical area V4 to IT. *Journal of Neuroscience*, 30(39):12978–12995, 2010.
- H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang. IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv:2308.06721*, 2023.
