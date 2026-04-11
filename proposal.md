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

### Neural Encoder

We will train a regularized encoder that maps pseudo-population firing rate vectors to the SigLIP image embedding space. Given the ratio of stimuli (300) to channels (potentially thousands), we will use **ridge regression** as a baseline and compare it against sequence-aware architectures such as a **low-rank bottleneck MLP** or an **LSTM** that operates over the temporal dimension of the neural response. Training will use a contrastive objective: aligning neural embeddings with SigLIP embeddings of the presented images, to improve data efficiency over direct regression.

### Image Generation

The predicted SigLIP embedding will condition a pretrained **FLUX.1-dev** model via its **IP-Adapter** [Ye et al., 2023], which injects image embeddings into the transformer's attention layers without full fine-tuning. FLUX.1 is a state-of-the-art open-weight generative model that offers substantially improved image quality over earlier diffusion architectures. We will explore light fine-tuning of the IP-Adapter projection layers on our dataset.

### Text-Conditioned Hybrid

Because 300 images may under-constrain a purely neural decoder, we will implement a hybrid condition where short text captions (e.g., from BLIP-2 auto-captioning) supplement the neural embedding. We will systematically compare **neural-only**, **text-only**, and **neural+text** conditions to isolate the contribution of neural signals beyond what text provides.

---

## 4. Deliverables & Evaluation

**Deliverables:**
1. A trained neural encoder mapping IT pseudo-population activity to SigLIP space
2. A conditioned image generation pipeline producing reconstructions from held-out neural responses
3. A systematic comparison across decoding conditions (neural-only, text-only, neural+text) with ablations

**Evaluation:** We will use 5-fold cross-validation over the 300 image stimulus set. Metrics include:
- **Semantic similarity** — embedding cosine similarity
- **Perceptual quality** — SSIM
- **Identification accuracy** — 2AFC on held-out trials

**Success criteria:**
- (a) Neural-only reconstructions achieve above-chance 2AFC identification
- (b) Neural+text significantly outperforms text-only, demonstrating that neural signals carry stimulus-specific information beyond category labels

---

## 5. Compute Plan

Neural encoder training is lightweight (ridge regression, MLP, or LSTM on neural time series) and can run on a local machine or Google Colab. Image generation with FLUX.1-dev + IP-Adapter requires GPU inference; we will use the compute nodes of the Issa Lab at Columbia's Zuckerman Institute for parallelized generation and fine-tuning.

Total compute requirements are modest: encoder training takes minutes, and generating 300 images through the diffusion pipeline takes on the order of a few GPU-hours. The project is feasible within available resources.

---

## References

- Y. Takagi and S. Nishimoto. High-resolution image reconstruction with latent diffusion models from human brain activity. In *CVPR*, 2023.
- P. Scotti, M. Tripathy, C. K. T. Villanueva, et al. MindEye2: Shared-subject models enable fMRI-to-image with 1 hour of data. *arXiv:2403.11207*, 2024.
- M. Ciferri, M. Ferrante, and N. Toschi. Simple models, rich representations: Visual decoding from primate intracortical neural signals. *arXiv:2601.11108*, 2026.
- N. C. Rust and J. J. DiCarlo. Selectivity and tolerance ("invariance") both increase as visual information propagates from cortical area V4 to IT. *Journal of Neuroscience*, 30(39):12978–12995, 2010.
- H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang. IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv:2308.06721*, 2023.
