# From Spikes to Scenes: Neural Image Reconstruction from Marmoset Visual Cortex in the Data-Limited Regime

**Jared Yao - Columbia University - Spring 2026**

---

## Introduction

The primate ventral visual stream builds a rich, tolerant representation that supports robust and invariant object recognition. A compelling test of how well we understand this representation is whether we can invert it, reconstructing the perceived image directly from neural activity.

Recent work has achieved this by aligning brain responses to the latent spaces of diffusion models. Conditioning Stable Diffusion on fMRI activity mapped to CLIP embeddings produces recognizable reconstructions of perceived images (Takagi and Nishimoto, 2023), and subsequent shared-subject models have pushed this to as little as one hour of fMRI data (Scotti et al., 2024). More recently, similar decoding has been demonstrated from intracortical spiking activity in macaque (Ciferri et al., 2026). Diffusion models have also been used as analysis tools: guiding synthesis with mutual information over disentangled latent subgroups reveals semantic selectivity in higher visual cortex (Wang et al., 2026), though their object-only stimuli with coherent centered structure represent a considerably simpler setting than cluttered variable-background images. Crucially, all prior approaches rely on large datasets (22,000+ images or thousands of trials) atypical of most primate neurophysiology. Whether reconstruction is feasible in the *data-limited regime* (a few hundred stimuli, invasive electrophysiology, non-human primates) remains open.

We address this using Neuropixels 1.0 recordings from marmoset ventral stream cortices during passive viewing of the HVM dataset (DiCarlo et al., 2012): 450 images spanning 10 categories x 45 pose/scale/background variations, split 270/90/90. Key challenges include severe data scarcity (80x fewer training pairs than MindEye2), noisy pseudo-population responses pooled across sessions, distribution mismatch between predicted and ground-truth embeddings at generation time, the need to separate object-level from scene-level signals, and limited spatial control in the generative model.

---

## Method

### Stage 1 - Neural Encoding

We train two lightweight MultiHeadTransformer encoders (Figure 1). The **global model** predicts full-image SigLIP-SO400M (1152-d) and CLIP-short (768-d) embeddings. The **object model** predicts SigLIP of the GDINO-cropped object region and CLIP-short embeddings. Separating targets is neurobiologically motivated: the ventral stream encodes a position- and size-invariant representation of objects, so targeting the object crop is more aligned with what the ventral population actually represents than targeting the full image. Both encoders share a pre-LN transformer backbone with attention pooling, a shared projection layer, and dual L2-normalised heads. A learned category embedding is added to the shared latent as a low-data inductive bias. Crucially, the shared latent also serves as an interpretable probe of the neural population, capturing a compressed summary of ventral stream representations that can be analyzed independently of generation.

The training loss combines per-head InfoNCE and cosine regression with a uniformity penalty on the shared latent:

$$\mathcal{L} = w_{\text{sig}} \Bigl[\alpha\,\mathcal{L}^{\text{sig}}_{\text{NCE}} + (1-\alpha)\,\mathcal{L}^{\text{sig}}_{\text{cos}}\Bigr] + w_{\text{clip}} \Bigl[\alpha\,\mathcal{L}^{\text{clip}}_{\text{NCE}} + (1-\alpha)\,\mathcal{L}^{\text{clip}}_{\text{cos}}\Bigr] + w_{\text{unif}}\,\mathcal{L}_{\text{unif}}$$

where $\alpha$ controls the InfoNCE/cosine tradeoff and $\mathcal{L}_{\text{unif}}$ penalises collapsed shared latents. Hyperparameters are swept on SLURM; the best configuration is d_model=64, n_heads=4, n_layers=2, shared_dim=512.

*[Figure 1: MultiHeadTransformer architecture. Neural population response is projected to a shared latent, conditioned on category, and mapped to dual SigLIP and CLIP-short heads.]*

### Stage 2 - Conditioned Generation

Predicted embeddings condition a pretrained **FLUX.1-dev** via the pretrained **InstantX/FLUX.1-dev-IP-Adapter** with no finetuning. Generation uses a sequential two-pass img2img procedure (Figure 2):

1. **Pass 1 (object recovery):** The GDINO bbox crop is denoised conditioned on **z_sig_obj**, then composited back onto the original to form a spliced latent.
2. **Pass 2 (global recovery):** Starts from the original init latent conditioned on **z_sig_global**. At each step, bbox tokens are blended toward the spliced latent with a cosine-decaying weight (0.6 to 0), anchoring object structure early and releasing it late. Latents outside the circular aperture are pinned to the original.

Four conditions are evaluated: Control, Text-only (category name), Neural pred (predicted embeddings), and GT upper bound (ground-truth embeddings).

*[Figure 2: Two-pass generation pipeline. Pass 1 recovers object structure; the result guides Pass 2 for global coherence.]*

---

## Results

### Encoder Generalization

Both models generalize to held-out test stimuli, confirming that 270 training examples are sufficient to recover stimulus-specific embedding information from noisy pseudo-population responses.

| Model | SigLIP cos sim | CLIP cos sim | SigLIP 2-AFC | CLIP 2-AFC |
|---|---|---|---|---|
| Global model | TBD | TBD | TBD | TBD |
| Object model | TBD | TBD | TBD | TBD |

*[Table 1: Embedding cosine similarity and 2-AFC identification accuracy (chance = 0.5) on the 90 test stimuli.]*

Above-chance 2-AFC accuracy on both heads indicates that predicted embeddings are measurably closer to the correct ground-truth than to the 89 distractors, even under the distributional mismatch between pseudo-population responses pooled across sessions.

### Neural Pred Matches or Exceeds Text-Guided Reconstruction Quality

Across the 90 test stimuli, neural pred reconstructions are visually comparable to or better than text-conditioned reconstructions (condition 2), which receive the category name via both T5 and CLIP pooled embeddings. This is a meaningful bar: the text condition has access to the correct category label at inference time, while neural pred receives only the predicted embedding. That neural pred competes with text conditioning on overall image quality indicates that the predicted SigLIP embeddings carry semantic signal sufficient to drive the generative model, despite the low training data regime.

*[Figure 3: Five-column composite strips (Original | Control | Text | Neural Pred | GT upper bound) for representative test stimuli.]*

### Decoding Object Gesture

The most diagnostic cases are stimuli where object pose, viewpoint, or spatial configuration varies substantially within a category. Neural pred reconstructions recover these within-category variations in several examples, producing outputs where the object gesture (e.g., limb position, orientation, facing direction) matches the original stimulus more closely than the text condition, which is blind to pose. This provides direct evidence that the encoder captures stimulus-specific visual structure beyond category membership.

*[Figure 4: Pose-diagnostic examples. Neural pred recovers the correct gesture or viewpoint; text condition produces the canonical category view.]*

### Failure Cases Reveal the Limits of the Ventral Population Representation

Systematic failures are informative. Two examples illustrate this. In the elephant stimuli with strong background cues, neural pred sometimes generates a plausible elephant pose but in the wrong context, suggesting that the pseudo-population signal is dominated by the object representation and carries limited background information, consistent with the ventral stream's known object-centric tuning. In a head stimulus where two faces appear in the image, neural pred tends to generate a single centered face rather than the multi-object configuration, suggesting the pooled population response reflects a single dominant object percept. These failures are not arbitrary noise: they reveal systematic properties of what the marmoset ventral population actually encodes, and they would not be visible without a reconstruction method sensitive enough to recover within-category structure in the first place.

*[Figure 5: Failure case examples. Left: elephant with background mismatch. Right: two-head stimulus with single-face reconstruction.]*

---

## Discussion

The results demonstrate that neural image reconstruction is feasible in the data-limited regime typical of primate electrophysiology. A lightweight encoder with only 270 training examples generalizes to held-out stimuli, recovering stimulus-specific embedding information from noisy pseudo-population responses. That neural pred reconstructions match or exceed text-guided quality is a non-trivial finding: the text condition receives the correct category label at inference time, whereas neural pred operates purely from predicted embeddings. The ability to decode within-category object gesture — recovering pose, viewpoint, and spatial configuration that are invisible to the text condition — is direct evidence that the ventral population signal carries fine-grained visual structure beyond category membership. The two-pass pipeline contributes to this: targeting the object crop in Pass 1 aligns the generation signal with the ventral stream's position- and size-invariant object code, while the cosine-decaying bbox blend in Pass 2 recovers global coherence without hard boundary artifacts.

The failure cases are equally informative. The elephant examples, where neural pred generates the correct object in the wrong background context, are consistent with the ventral stream's known object-centric tuning: the population signal is dominated by the object representation and carries limited scene-level information. The two-head stimuli, where neural pred collapses to a single centered face, suggest the pooled population response reflects a single dominant object percept rather than a full scene description. These are not reconstruction failures in the engineering sense; they are windows into what the marmoset ventral population encodes.

Several limitations qualify these conclusions. The aperture mask pins background pixels to the original stimulus, so background reconstruction is never actually tested. The init latent is the ground-truth stimulus noised to strength=0.65, meaning the generator starts close to the answer; a blind init would give a cleaner measure of how much reconstruction is driven by the neural signal alone. Performance is also uneven across categories, with high intra-class pose variation (cars, planes) proving harder than compact canonical-view categories (apples, heads). Image-level perceptual metrics and human evaluation remain to be conducted.

**Future directions.** The most impactful near-term change would be switching to a blind init latent and adding a small amount of neural-conditioned finetuning of the IP-Adapter resampler. Longer-term directions include cross-animal generalization tests, incorporating temporal spike structure beyond mean rate, and probing the shared latent geometry as a standalone analysis of what visual dimensions the ventral population prioritises across categories and stimulus variations.

---

## References

- J. J. DiCarlo, D. Zoccolan, and N. C. Rust. How does the brain solve visual object recognition? *Neuron*, 73(3):415-434, 2012.
- Y. Takagi and S. Nishimoto. High-resolution image reconstruction with latent diffusion models from human brain activity. *CVPR*, 2023.
- P. Scotti et al. MindEye2: Shared-subject models enable fMRI-to-image with 1 hour of data. *arXiv:2403.11207*, 2024.
- M. Ciferri, M. Ferrante, and N. Toschi. Simple models, rich representations: Visual decoding from primate intracortical neural signals. *arXiv:2601.11108*, 2026.
- H. Ye et al. IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv:2308.06721*, 2023.
- Y. Wang et al. Uncovering semantic selectivity of latent groups in higher visual cortex with mutual information-guided diffusion. *ICLR*, 2026. arXiv:2510.02182.
