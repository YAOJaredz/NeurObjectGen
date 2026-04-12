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

### Stage 2 — IP-Adapter Training

The predicted SigLIP embedding conditions a pretrained **FLUX.1-dev** model via a custom IP-Adapter projection module. Rather than modifying the attention processor (which FLUX's native `FluxAttnProcessor` does not support), we project the image embedding into **N extra text tokens** that are prepended to the T5 `encoder_hidden_states` before each transformer forward pass. FLUX's existing joint attention then attends to these tokens naturally alongside the text prompt.

The IP-Adapter projection MLP (image_dim → hidden → N × 4096) is trained separately from the neural encoder, using **ground-truth SigLIP embeddings** of the 200 training stimuli as conditioning input. The training objective is a **flow-matching reconstruction loss** on FLUX's packed VAE latents:

- For each training step, sample t ~ U[0,1], form the noisy latent x_t = (1−t) x_0 + t ε, and minimize MSE between the transformer's predicted velocity and the target ε − x_0
- The loss is computed **only within the circular stimulus aperture**, excluding the central fixation square, via a soft mask downsampled from pixel space to the packed-latent grid (each position covers a 16×16 pixel block)
- Only the IP-Adapter MLP is updated; all FLUX weights remain frozen

### Inference Pipeline

At test time the two stages are chained:

```
neural r (N×T)  ──encoder──▶  ẑ ∈ R^1152  ──IP-Adapter──▶  ip_tokens  ──FLUX──▶  x̂
```

The neural encoder predicts a SigLIP-aligned embedding from held-out neural responses; the IP-Adapter projects it into FLUX text tokens; and FLUX decodes the conditioned tokens into a reconstructed image.

### Text-Conditioned Hybrid

Because 300 images may under-constrain a purely neural decoder, we implement a hybrid condition where short text captions (from BLIP-2 auto-captioning) supplement the neural embedding. We systematically compare **neural-only**, **text-only**, and **neural+text** conditions to isolate the contribution of neural signals beyond what text provides. In the text-only baseline, the IP-Adapter scale is set to zero so all three conditions share the same generation code path.

---

## 4. Deliverables & Evaluation

**Deliverables:**
1. A trained neural encoder mapping IT pseudo-population activity to SigLIP space
2. A trained IP-Adapter projection mapping SigLIP embeddings to FLUX conditioning tokens
3. A conditioned image generation pipeline producing reconstructions from held-out neural responses
4. A systematic comparison across decoding conditions (neural-only, text-only, neural+text) with ablations

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

**Stage 2 (IP-Adapter):** Requires GPU memory for the frozen FLUX.1-dev transformer (~24 GB). Training runs at batch size 1 with only the small IP-Adapter MLP updating gradients. Each epoch over 200 stimuli takes on the order of minutes; 50 epochs is feasible within available GPU resources at Columbia's Zuckerman Institute.

**Inference:** Generating 300 images through the full pipeline takes on the order of a few GPU-hours. Total compute requirements remain modest relative to the available resources.

---

## References

- Y. Takagi and S. Nishimoto. High-resolution image reconstruction with latent diffusion models from human brain activity. In *CVPR*, 2023.
- P. Scotti, M. Triparity, C. K. T. Villanueva, et al. MindEye2: Shared-subject models enable fMRI-to-image with 1 hour of data. *arXiv:2403.11207*, 2024.
- M. Ciferri, M. Ferrante, and N. Toschi. Simple models, rich representations: Visual decoding from primate intracortical neural signals. *arXiv:2601.11108*, 2026.
- N. C. Rust and J. J. DiCarlo. Selectivity and tolerance ("invariance") both increase as visual information propagates from cortical area V4 to IT. *Journal of Neuroscience*, 30(39):12978–12995, 2010.
- H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang. IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv:2308.06721*, 2023.
