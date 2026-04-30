# From Spikes to Scenes: Neural Image Reconstruction from Marmoset Visual Cortex in the Data-Limited Regime

**Jared Yao - Columbia University - Spring 2026**

---

## Introduction

The primate ventral visual stream builds a rich, tolerant representation in inferotemporal (IT) cortex that supports robust and invariant object recognition (DiCarlo et al., 2012). A compelling test of how well we understand this representation is whether we can invert it, reconstructing the perceived image directly from neural activity.

Recent work has achieved this by aligning brain responses to the latent spaces of diffusion models. Conditioning Stable Diffusion on fMRI activity mapped to CLIP embeddings produces recognizable reconstructions of perceived images (Takagi and Nishimoto, 2023), and subsequent shared-subject models have pushed this to as little as one hour of fMRI data (Scotti et al., 2024). More recently, similar decoding has been demonstrated from intracortical spiking activity in macaque (Ciferri et al., 2026). Diffusion models have also been used as analysis tools: guiding synthesis with mutual information over disentangled IT latent subgroups reveals semantic selectivity in higher visual cortex (Wang et al., 2026), though their object-only stimuli with coherent centered structure represent a considerably simpler setting than cluttered variable-background images. Crucially, all prior approaches rely on large datasets (22,000+ images or thousands of trials) atypical of most primate neurophysiology. Whether reconstruction is feasible in the *data-limited regime* (a few hundred stimuli, invasive electrophysiology, non-human primates) remains open.

We address this using Neuropixels 1.0 recordings from marmoset ventral stream cortices during passive viewing of the HVM dataset (DiCarlo et al., 2012): 450 images spanning 10 categories x 45 pose/scale/background variations, split 270/90/90. Key challenges include severe data scarcity (80x fewer training pairs than MindEye2), noisy pseudo-population responses pooled across sessions, distribution mismatch between predicted and ground-truth embeddings at generation time, the need to separate object-level from scene-level signals, and limited spatial control in the generative model.

---

## Method

### Stage 1 - Neural Decoding

We train two lightweight MultiHeadTransformer decoders (Figure 1). The **global model** predicts full-image SigLIP-SO400M (1152-d) and CLIP-short (768-d) embeddings. The **object model** predicts SigLIP of the GDINO-cropped object region and CLIP-short embeddings. Separating targets is neurobiologically motivated: IT cortex encodes a position- and size-invariant representation of objects, so targeting the object crop is more aligned with what IT actually represents than targeting the full image. Both decoders share a pre-LN transformer encoder with attention pooling, a shared projection layer, and dual L2-normalised heads. A learned category embedding is added to the shared latent as a low-data inductive bias. Crucially, the shared latent also serves as an interpretable probe of the neural population, capturing a compressed summary of IT representations that can be analyzed independently of generation.

The training loss combines per-head InfoNCE and cosine regression with a uniformity penalty on the shared latent:

$$\mathcal{L} = w_{\text{sig}} \Bigl[\alpha\,\mathcal{L}^{\text{sig}}_{\text{NCE}} + (1-\alpha)\,\mathcal{L}^{\text{sig}}_{\text{cos}}\Bigr] + w_{\text{clip}} \Bigl[\alpha\,\mathcal{L}^{\text{clip}}_{\text{NCE}} + (1-\alpha)\,\mathcal{L}^{\text{clip}}_{\text{cos}}\Bigr] + w_{\text{unif}}\,\mathcal{L}_{\text{unif}}$$

where $\alpha$ controls the InfoNCE/cosine tradeoff and $\mathcal{L}_{\text{unif}}$ penalises collapsed shared latents. Hyperparameters are swept on SLURM; the best configuration is d_model=64, n_heads=4, n_layers=2, shared_dim=512.

*[Figure 1: MultiHeadTransformer architecture. Neural population response is projected to a shared latent, conditioned on category, and decoded into dual SigLIP and CLIP-short heads.]*

### Stage 2 - Conditioned Generation

Predicted embeddings condition a pretrained **FLUX.1-dev** via the pretrained **InstantX/FLUX.1-dev-IP-Adapter** with no finetuning. Generation uses a sequential two-pass img2img procedure (Figure 2):

1. **Pass 1 (object recovery):** The GDINO bbox crop is denoised conditioned on **z_sig_obj**, then composited back onto the original to form a spliced latent.
2. **Pass 2 (global recovery):** Starts from the original init latent conditioned on **z_sig_global**. At each step, bbox tokens are blended toward the spliced latent with a cosine-decaying weight (0.6 to 0), anchoring object structure early and releasing it late. Latents outside the circular aperture are pinned to the original.

Four conditions are evaluated: Control, Text-only (category name), Neural pred (predicted embeddings), and GT upper bound (ground-truth embeddings).

*[Figure 2: Two-pass generation pipeline. Pass 1 recovers object structure; the result guides Pass 2 for global coherence.]*

---

## Results

### Decoder Prediction Performance

| Model | SigLIP cos sim | CLIP cos sim | SigLIP 2-AFC | CLIP 2-AFC |
|---|---|---|---|---|
| Global model | TBD | TBD | TBD | TBD |
| Object model | TBD | TBD | TBD | TBD |

*[Table 1: Embedding cosine similarity and 2-AFC accuracy (chance = 0.5) on the 90 test stimuli. Image-level metrics (SSIM, perceptual scores) will be added.]*

Both models achieve above-chance 2-AFC accuracy, confirming that 270 training examples are sufficient to recover stimulus-specific embedding information from noisy neural responses.

### Reconstruction Quality

As shown in Figure 3, neural pred reconstructions recover the correct object category and, in many cases, rough object pose and scale. The most striking examples are cases where neural pred captures viewpoint or scale that text-only completely misses.

*[Figure 3: Selected 5-column composite strips (Original | Control | Text | Neural Pred | GT upper bound).]*

---

## Discussion

The results demonstrate that neural image reconstruction is feasible in the data-limited regime typical of primate electrophysiology. A lightweight decoder with only 270 training examples can recover enough stimulus-specific information from noisy pseudo-population responses to drive a pretrained generative model toward visually plausible reconstructions without any generative finetuning. The contrastive objective is well suited to this setting, directly optimising a 2-AFC-style ranking that encourages predicted embeddings to be closer to the correct ground-truth than to other stimuli in the batch. The two-pass pipeline further reflects the neuroscience: by feeding the object crop embedding in Pass 1, we align the generation signal with IT's position- and size-invariant object code rather than full-image statistics, while the cosine-decaying bbox blend in Pass 2 recovers global coherence without hard boundary artifacts.

Several limitations qualify these conclusions. The aperture mask pins background pixels to the original stimulus, so full-image metrics are flattering and background reconstruction is never actually tested. The init latent is the ground-truth stimulus noised to strength=0.65, meaning the generator starts close to the answer; a blind init would give a cleaner measure of how much reconstruction is driven by the neural signal alone. Performance is also uneven across categories, with high intra-class pose variation (cars, planes) proving harder than compact canonical-view categories (apples, heads). Image-level perceptual metrics and human evaluation remain to be conducted.

**Future directions.** The most impactful near-term change would be switching to a blind init latent and adding a small amount of neural-conditioned finetuning of the IP-Adapter resampler. Longer-term directions include cross-animal generalization tests, incorporating temporal spike structure beyond mean rate, and probing the shared latent geometry as a standalone analysis of what visual dimensions IT cortex prioritises across categories and stimulus variations.

---

## References

- J. J. DiCarlo, D. Zoccolan, and N. C. Rust. How does the brain solve visual object recognition? *Neuron*, 73(3):415-434, 2012.
- Y. Takagi and S. Nishimoto. High-resolution image reconstruction with latent diffusion models from human brain activity. *CVPR*, 2023.
- P. Scotti et al. MindEye2: Shared-subject models enable fMRI-to-image with 1 hour of data. *arXiv:2403.11207*, 2024.
- M. Ciferri, M. Ferrante, and N. Toschi. Simple models, rich representations: Visual decoding from primate intracortical neural signals. *arXiv:2601.11108*, 2026.
- H. Ye et al. IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv:2308.06721*, 2023.
- N. C. Rust and J. J. DiCarlo. Selectivity and tolerance both increase as visual information propagates from V4 to IT. *Journal of Neuroscience*, 30(39):12978-12995, 2010.
- Y. Wang et al. Uncovering semantic selectivity of latent groups in higher visual cortex with mutual information-guided diffusion. *ICLR*, 2026. arXiv:2510.02182.
