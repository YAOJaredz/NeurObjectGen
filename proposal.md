# Neural Image Reconstruction from Primate IT Cortex via Conditioned Generative Models

**Jared Yao, Columbia University**

---

## 1. Background

The primate ventral visual stream transforms retinal input into a rich representation in inferotemporal (IT) cortex that supports robust object recognition [Rust and DiCarlo, 2010]. A powerful test of how well we understand this representation is whether we can invert it—reconstructing the perceived image directly from neural activity. This also poses challenges for modern generative models, requiring them to produce faithful reconstructions when conditioned on noisy, low-dimensional signals far removed from their training distribution of text and image embeddings.

Recent work has achieved striking reconstructions by aligning brain responses with the latent spaces of pretrained generative models, first from fMRI [Takagi and Nishimoto, 2023, Scotti et al., 2024] and more recently from intracortical electrophysiology [Ciferri et al., 2026]. However, these successes rely on massive datasets (e.g., 22,000 images, 1024 channels) atypical of most primate neurophysiology and sidestep the question of whether reconstruction is possible in the data-limited regime.

We address this using Neuropixels 1.0 recordings from primate IT cortex during passive viewing of the High-variation (HVM) object dataset [DiCarlo et al., 2012], which comprises 450 naturalistic images spanning 10 object categories. Pooling neural responses across sessions yields a large pseudo-population, enabling high-dimensional decoding despite the modest stimulus count.

---

## 2. Problem Formulation

We collect neural responses with Neuropixels 1.0 probes across multiple sessions from primate IT cortex during passive viewing of the 450-image HVM stimulus set (10 object categories × 45 variations each). Neural responses are pooled across sessions to form a pseudo-population. The goal is to reconstruct the perceived image from population activity.

Formally, we train two decoders **f_global** and **f_obj**, each mapping the same neural response **r ∈ R^(N×T)** to dual embedding targets. **f_global** targets full-image embeddings: **ẑ_sig_global ∈ R^1152** (SigLIP-SO400M of the full stimulus) and **ẑ_clip ∈ R^768** (CLIP-short). **f_obj** targets object-crop embeddings: **ẑ_sig_obj ∈ R^1152** (SigLIP-SO400M of the GDINO-cropped object region). Both SigLIP predictions condition a pretrained image generation model — the global embedding shapes overall scene coherence and the object embedding steers the object region specifically — to produce a reconstruction **x̂** of the original stimulus **x**.

Stimuli are split by category-stratified sampling into train/validation/test sets (270/90/90), ensuring each category is proportionally represented across splits.

---

## 3. Approach

### Stage 1 — Neural Encoders (MultiHeadTransformer × 2)

Two **MultiHeadTransformer** models are trained independently on the same neural data but with different SigLIP targets. The **global model** targets full-image SigLIP and CLIP-short embeddings; the **object model** targets SigLIP embeddings extracted from the GDINO-cropped object region (512 × 512 PIL crop, SigLIP-SO400M encoded). Both share the same architecture:

- **Input projection:** LayerNorm over neurons, then a linear projection to a d_model-dimensional token embedding for each time step
- **CLS token + positional embeddings** prepended to the time-step sequence
- **Pre-LN transformer encoder:** configurable depth (n_layers) and width (d_model), with GELU activations and multi-head self-attention
- **Attention pooling:** a learned scalar attention weight over all tokens (including CLS) produces a single pooled vector
- **Shared projection:** a linear layer from d_model → shared_dim followed by GELU, producing a shared latent
- **Category conditioning:** a learned embedding for each of the 10 HVM categories is added to the shared latent at training and inference time
- **Dual heads:** two linear layers from shared_dim → 1152 (SigLIP) and shared_dim → 768 (CLIP), each L2-normalised

**Training objective.** The loss combines a per-head weighted sum of InfoNCE (symmetric contrastive loss, CLIP-style) and cosine regression, plus a uniformity loss on the shared latent:

```
L = w_sig * head_loss(ẑ_sig, z_sig) + w_clip * head_loss(ẑ_clip, z_clip) + w_unif * L_unif(shared)
head_loss = nce_weight * L_InfoNCE + (1 − nce_weight) * L_cos
```

Hyperparameters (d_model, n_layers, shared_dim, nce_weight, and others) are swept on the Issa Lab SLURM cluster; the best configuration for each model is selected by mean validation cosine similarity `(cos_siglip + cos_clip) / 2` and saved for inference. Because object-crop SigLIP embeddings are more view- and scale-sensitive than full-image embeddings, the object model is encouraged to capture stimulus-specific pose and appearance rather than global scene statistics.

### Stage 2 — Pretrained IP-Adapter (InstantX/FLUX.1-dev-IP-Adapter)

The predicted SigLIP and CLIP-short embeddings condition a pretrained **FLUX.1-dev** model via the **pretrained InstantX/FLUX.1-dev-IP-Adapter**. The adapter ships a SigLIP-SO400M encoder matching our cached embeddings and implements a real cross-attention path via a learned resampler (`MLPProjModel`: SigLIP 1152 → 128 × 4096 image tokens) and per-block IP key/value projections on all 57 FLUX transformer blocks.

Because diffusers' generic loader does not accept InstantX's checkpoint layout for FLUX, we vendor the minimal adapter components directly: the `MLPProjModel` and an `IPAFluxAttnProcessor` installed on all 57 attention blocks (19 double-stream + 38 single-stream). Image conditioning is threaded via `pipe(..., joint_attention_kwargs={"image_emb": image_emb, "scale": ip_adapter_scale})`. **No adapter training is performed** — Stage 2 is a pure inference component.

HVM-specific generation details:

- **Init latent:** the ground-truth stimulus image at 512 × 512 is VAE-encoded and noised to `strength=0.65`, so the generator begins from a perturbed version of the original stimulus
- **Per-step aperture compositing:** at every scheduler step, latents inside the circular HVM stimulus aperture are updated by the adapter-conditioned transformer while latents outside are pinned to a re-noised copy of the original init latent, using a soft mask downsampled to FLUX's packed-latent grid (each position covers a 16 × 16 pixel block)
- **Sequential two-pass generation** (`generate_obj_sequential`): reconstruction proceeds in two img2img passes. **Pass 1** crops the GDINO bbox region, upscales it to 512 × 512, and runs img2img conditioned on ẑ_sig_obj (`ip_adapter_scale=1.0`, no aperture compositing), producing an object-specific reconstruction. The generated crop is pasted back onto the original at pixel level and VAE-encoded into a *spliced latent*. **Pass 2** starts from the original init latent (not the spliced one) conditioned on ẑ_sig_global, while the spliced latent serves as a per-step bbox guidance reference: at each denoising step, bbox tokens are blended toward the re-noised spliced latent with a weight that decays via cosine schedule from `bbox_preserve_start=0.6` at t=1 to 0 at t=0. This anchors object structure early in diffusion and releases it late, allowing global coherence to emerge without leaving a hard boundary at the bbox edge.
- **Object-crop generation:** a separate img2img pass operates directly on the GDINO bbox crop resized to 512 × 512, with `aperture_composite=False` and `strength=0.65`. The generator refines object appearance conditioned on ẑ_sig_obj at full scale (`ip_adapter_scale=1.0`), providing an independent, background-free view of the reconstructed object
- **Text conditioning:** T5 embeddings are zeroed out. The CLIP pooled embedding is predicted by the object model (ẑ_clip), providing stimulus-specific global context beyond the category name

### Comparison Conditions

Four conditions are compared on the 90 held-out test stimuli to isolate the contribution of each signal:

| # | Condition | T5 | CLIP pooled | Pass 1 — obj SigLIP | Pass 2 — global SigLIP |
|---|-----------|-----|-------------|---------------------|------------------------|
| 1 | **Control** | null | null | disabled | disabled |
| 2 | **Text (cat CLIP + T5)** | category T5 | category-name CLIP | disabled | disabled |
| 3 | **Neural pred (sequential)** | zeroed | ẑ_clip (obj model) | ẑ_sig_obj (obj model) | ẑ_sig_global (global model) |
| 4 | **GT emb (sequential ↑)** | zeroed | category-name CLIP | GT SigLIP (crop) | GT SigLIP (full) |

The **control** condition establishes the img2img baseline with no semantic conditioning. **Text (cat CLIP + T5)** adds category-level semantic information via both the T5 text encoder and the CLIP pooled embedding of the category name. **Neural pred** uses the sequential two-pass approach: the object model's ẑ_sig_obj drives pass 1 (object recovery), ẑ_sig_global from the global model drives pass 2 (global coherence), and ẑ_clip provides stimulus-specific CLIP conditioning. **GT emb** substitutes ground-truth embeddings in both passes and serves as the upper bound on reconstruction quality.

### Inference Pipeline

```
                ┌─ global model ─┬── SigLIP head ──▶ ẑ_sig_global ─────────────────────────────────────────┐
neural r (N×T) ─┤                └── CLIP head ─────▶ (unused)                                             │
                │                                                                                          │
                └─ object model ─┬── SigLIP head ──▶ ẑ_sig_obj                                             │
                                 └── CLIP head ─────▶ ẑ_clip ──────────────────────────────▶ CLIP pooled   │
                                                          │                                                │
                                                          ▼                                               │
                                              ┌─── Pass 1 (obj recovery) ───┐                            │
                                              │  init: bbox crop of orig    │                            │
                                              │  SigLIP: ẑ_sig_obj          │                            │
                                              │  scale: 1.0, no aperture    │                            │
                                              └──────────┬──────────────────┘                            │
                                                         │ VAE-encode composite → spliced latent         │
                                                         ▼                                               ▼
                                              ┌─── Pass 2 (global recovery) ────────────────────────────┐
                                              │  init: original latent (not spliced)                    │
                                              │  SigLIP: ẑ_sig_global, scale: 1.0                       │
                                              │  bbox guidance: spliced latent, preserve 0.6→0 cosine   │
                                              │  aperture compositing: HVM circular mask                │
                                              └──────────┬──────────────────────────────────────────────┘
                                                         │
                                          ┌──────────────┴──────────────┐
                                          ▼                             ▼
                              full reconstruction x̂        obj-crop reconstruction x̂_obj
                                                           (separate pass, ẑ_sig_obj, no aperture)
```

---

## 4. Deliverables & Evaluation

**Deliverables:**
1. Two trained **MultiHeadTransformer** models: a global model targeting full-image SigLIP + CLIP-short, and an object model targeting bbox-crop SigLIP + CLIP-short
2. An inference wrapper around the pretrained InstantX/FLUX.1-dev-IP-Adapter with HVM-specific aperture compositing, sequential two-pass SigLIP conditioning, and object-crop generation
3. Per-stimulus reconstruction outputs across all four conditions for the 90 held-out test stimuli — both full-image and object-crop views saved as labeled PNG composites
4. A systematic comparison of all four conditions on quantitative metrics

**Evaluation:** Fixed category-stratified 270/90/90 train/val/test split. Metrics:
- **Embedding cosine similarity** — mean cosine similarity between predicted and GT embeddings (SigLIP and CLIP-short separately)
- **Identification accuracy** — 2-AFC forced-choice identification on the test set for both embedding heads

**Success criteria:**
- (a) Neural-only reconstructions achieve above-chance 2-AFC identification
- (b) Neural pred significantly outperforms text (cat CLIP), demonstrating that predicted SigLIP embeddings carry stimulus-specific visual information beyond what the category name provides

---

## 5. Compute Plan

**Stage 1 (MultiHeadTransformer × 2):** Hyperparameter sweep over d_model, n_layers, shared_dim, nce_weight, and regularisation coefficients, run via SLURM on the Issa Lab cluster inside an Apptainer container. Individual training runs are fast (a few minutes each on GPU). The global model is exported to `cache/best_hvm_multihead_config.json` and the object model to `cache/best_hvm_multihead_obj_config.json`.

**Stage 2 (IP-Adapter):** Inference only — no gradient updates. Requires GPU memory for the frozen FLUX.1-dev transformer (~24 GB bf16) plus the pretrained InstantX adapter weights. Generating 90 test stimuli × 4 conditions takes on the order of a few GPU-hours.

---

## References

- Y. Takagi and S. Nishimoto. High-resolution image reconstruction with latent diffusion models from human brain activity. In *CVPR*, 2023.
- P. Scotti, M. Triparity, C. K. T. Villanueva, et al. MindEye2: Shared-subject models enable fMRI-to-image with 1 hour of data. *arXiv:2403.11207*, 2024.
- M. Ciferri, M. Ferrante, and N. Toschi. Simple models, rich representations: Visual decoding from primate intracortical neural signals. *arXiv:2601.11108*, 2026.
- J. J. DiCarlo, D. Zoccolan, and N. C. Rust. How does the brain solve visual object recognition? *Neuron*, 73(3):415–434, 2012.
- N. C. Rust and J. J. DiCarlo. Selectivity and tolerance ("invariance") both increase as visual information propagates from cortical area V4 to IT. *Journal of Neuroscience*, 30(39):12978–12995, 2010.
- H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang. IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv:2308.06721*, 2023.
