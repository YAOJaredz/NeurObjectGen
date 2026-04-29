# NeurObjectGen

Neural image reconstruction from primate visual cortex neural activity via conditioned generative models.

We train lightweight decoders on Neuropixels recordings from marmoset visual ventral stream cortices during passive viewing of the HVM object dataset (450 images, 10 categories), then condition a pretrained FLUX.1-dev + InstantX IP-Adapter on the decoded embeddings to reconstruct the perceived image. The decoder has 270 training examples — the data-limited regime typical of primate neurophysiology.

---

## Pipeline

```
Neural response r (N neurons × 250 ms)
        │
        ├─ global model ──► ẑ_sig_global (1152-d)   ──────────────────────────────┐
        │                   ẑ_clip       (768-d, unused in generation)            │
        │                                                                         │
        └─ object model ──► ẑ_sig_obj   (1152-d)                                  │
                            ẑ_clip       (768-d) ──► CLIP pooled conditioning     │
                                │                                                 │
                         Pass 1 (object recovery)                                 │
                         init: GDINO bbox crop                                    │
                         SigLIP: ẑ_sig_obj, scale 1.0                             │
                                │                                                 │
                         splice crop back → spliced latent                        │
                                │                                                 ▼
                         Pass 2 (global recovery)  ◄──────────────── SigLIP: ẑ_sig_global
                         init: original latent                                 
                         bbox guidance: spliced latent, preserve 0.6→0 (cosine)  
                         aperture compositing: circular HVM mask               
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
     full reconstruction x̂        obj-crop reconstruction x̂_obj
```

**Stage 1 — MultiHeadTransformer (×2).** Each model maps neural population responses to dual embedding targets. The *global model* targets full-image SigLIP-SO400M (1152-d) + CLIP-short (768-d). The *object model* targets bbox-crop SigLIP + CLIP-short. Both use a pre-LN transformer encoder with attention pooling, a shared projection to `shared_dim`, optional category conditioning (learned embedding added to shared latent), and two L2-normalised head projections. Loss: `w_sig · head_loss(ẑ_sig, z_sig) + w_clip · head_loss(ẑ_clip, z_clip) + w_unif · L_unif`, where `head_loss = nce_weight · InfoNCE + (1 − nce_weight) · cosine`.

**Stage 2 — FLUX.1-dev + InstantX IP-Adapter (inference only).** No adapter training. The pretrained `MLPProjModel` (SigLIP 1152 → 128 × 4096 image tokens) conditions all 57 FLUX transformer blocks via installed `IPAFluxAttnProcessor` key/value projections. Generation is a sequential two-pass img2img:

1. **Pass 1 (object):** GDINO bbox crop → 512×512, img2img conditioned on ẑ_sig_obj. Generated crop is spliced back onto the original to form a spliced latent.
2. **Pass 2 (global):** starts from the original init latent conditioned on ẑ_sig_global. At each denoising step, bbox tokens are blended toward the re-noised spliced latent with weight cosine-decaying from 0.6 → 0, anchoring object structure early and releasing it late. Latents outside the circular aperture are pinned to the original at every step.

---

## Dataset

**HVM (High-Variation Object) dataset.** 450 naturalistic images: 10 object categories (apple, bear, car, chair, dog, elephant, head, plane, table, turtle) × 45 variations (pose, scale, background). Neural responses are mean-trial spike rates recorded with Neuropixels from marmoset visual ventral stream cortices and pooled across sessions to form a pseudo-population.

| Split | Size | Note                |
| ----- | ---- | ------------------- |
| Train | 270  | category-stratified |
| Val   | 90   | category-stratified |
| Test  | 90   | category-stratified |

Stimuli are 512 × 512 with a circular aperture (radius = 49% of image width) on a gray background.

---

## Setup

```bash
conda env create -f environment.yml
conda activate objGen
```

All scripts are run from the repo root. Key paths are defined in `config_const.py`. HVM stimulus images are expected at `stimuli/hvm_nofixation/`. Pre-computed caches (`cache/*.pt`, `cache/*.json`) are required for training and generation — see `scripts/` for the scripts that build them.

---

## Training

```bash
# Train global model (full-image SigLIP + CLIP)
python train/train_multihead.py --dataset hvm --use-category

# Train object model (bbox-crop SigLIP + CLIP)
python train/train_multihead_obj.py --use-category
```

Best configurations are saved to `cache/best_hvm_multihead_config.json` and `cache/best_hvm_multihead_obj_config.json`. Best model from the sweep: `d_model=64, n_heads=4, n_layers=2, shared_dim=512`.

---

## Generation

```bash
# Generate reconstructions for all 90 test stimuli (4 conditions × full + crop views)
python scripts/generate_hvm_obj.py

# Limit to first N stimuli
python scripts/generate_hvm_obj.py --n 10
```

Outputs are saved to `outputs/generate_hvm_obj/full/` and `outputs/generate_hvm_obj/crop/` as labeled 5-column PNG composites:

| Column | Condition            | T5          | CLIP                | SigLIP                                   |
| ------ | -------------------- | ----------- | ------------------- | ---------------------------------------- |
| 1      | Original             | —          | —                  | —                                       |
| 2      | Control              | null        | null                | disabled                                 |
| 3      | Text (cat CLIP + T5) | category T5 | category CLIP       | disabled                                 |
| 4      | Neural pred          | zeroed      | ẑ_clip (obj model) | ẑ_sig_obj → ẑ_sig_global (sequential) |
| 5      | GT emb (upper bound) | zeroed      | category CLIP       | GT SigLIP crop → GT SigLIP full         |

---

## Evaluation Metrics

- **Cosine similarity** — mean cosine between predicted and ground-truth embeddings
- **2-AFC identification** — pairwise forced-choice accuracy (chance = 0.5): for each test sample, check whether the predicted embedding is closer to its own ground-truth than to all other ground-truths
- **Top-k retrieval** — fraction of test samples whose correct target ranks in the top k by cosine similarity
- **SSIM** — structural similarity on generated images

Success criteria: (a) neural-only 2-AFC > chance, (b) neural pred 2-AFC > text condition.

---

## References

- N. C. Rust and J. J. DiCarlo. Selectivity and tolerance ("invariance") both increase as visual information propagates from cortical area V4 to IT. *Journal of Neuroscience*, 30(39):12978–12995, 2010.
- J. J. DiCarlo, D. Zoccolan, and N. C. Rust. How does the brain solve visual object recognition? *Neuron*, 73(3):415–434, 2012.
- M. Ciferri, M. Ferrante, and N. Toschi. Simple models, rich representations: Visual decoding from primate intracortical neural signals. *arXiv:2601.11108*, 2026.
