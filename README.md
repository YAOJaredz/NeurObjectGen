# From Spikes to Scenes: Neural Image Reconstruction from Marmoset Visual Cortex in the Data-Limited Regime

Reconstructing perceived images from neural activity tests how well we understand the ventral visual stream, but published approaches have required far more data than a typical primate electrophysiology session yields. We ask whether reconstruction is feasible in the data-limited regime, using Neuropixels recordings from marmoset ventral stream cortices during passive viewing of the 450-image HVM object set, with a 2,330-neuron pseudo-population pooled across sessions and **270 training examples**. A lightweight transformer decodes population responses into SigLIP and CLIP embeddings, which condition a pretrained FLUX.1-dev IP-Adapter through a two-pass procedure that recovers the object before rebuilding the scene around it. A rank-16 LoRA adapts the FLUX backbone to the stimulus domain. On 90 held-out stimuli the global SigLIP decoder reaches 0.96 2-AFC identification (chance 0.5), and reconstructions conditioned on decoded activity exceed those conditioned on the ground-truth category label in both SSIM and pixel correlation, indicating that the embeddings carry stimulus-specific information a category name does not supply.

Probing the representation bounds that claim. Decoded embeddings retain image identity beyond category, reaching 0.71 within-category 2-AFC where a category-centroid baseline sits at 0.50 by construction, but the brain-decodable subspace is approximately 10-dimensional and dominated by between-category structure, with pose, size, and position largely unrecovered. Reconstruction in this regime is therefore feasible but rests on a narrow, category-weighted code. Encoder metrics come from the unified three-head model, while the reconstructions were generated from the earlier separate global and object encoders.

![Reconstruction composite strips](assets/fig3_reconstructions.png)

*Original | Text-guided | Neural-guided | GT-guided, for held-out test stimuli.*

---

## How it works

Two lightweight transformer encoders map a 2,330-neuron pseudo-population to SigLIP-SO400M (1152-d) and CLIP (768-d) embeddings. A frozen InstantX IP-Adapter then conditions FLUX.1-dev on those embeddings through a two-pass img2img procedure: pass 1 recovers the object inside its GDINO bounding box, and pass 2 rebuilds the full scene around it.

![Two-pass generation pipeline](assets/fig2_pipeline.png)

1. **Pass 1 (object recovery).** The GDINO bbox crop is denoised conditioned on `ẑ_sig_obj`, then composited back onto the original to form a spliced latent.
2. **Pass 2 (global recovery).** Starting from the original init latent conditioned on `ẑ_sig_global`, bbox tokens are blended toward the spliced latent with a cosine-decaying weight (0.6 → 0), anchoring object structure early and releasing it late. Latents outside the circular aperture are pinned to the original.

The IP-Adapter itself is used as pretrained. To close the domain gap between its natural-photograph training distribution and HVM stimuli (controlled lighting, plain backgrounds, circular aperture), a rank-16 LoRA is fine-tuned on the 270 HVM training images over the FLUX transformer's attention and feed-forward layers, with the base weights, VAE, text encoders, SigLIP encoder, and IP-Adapter all frozen. That LoRA is loaded by default at inference.

---

## Install and usage

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. All commands run from the repo root.

```bash
uv sync

# HexPred is reached by path, not as a dependency
printf '%s\n' /home/yy3658 /home/yy3658/HexPred /home/yy3658/helpers \
  > .venv/lib/python3.12/site-packages/extra_paths.pth

uv run python train/train_multihead_v2.py                   # unified three-head encoder
uv run python train/train_multihead.py --dataset hvm --use-category   # global model, for generation
uv run python train/train_multihead_obj.py --use-category             # object model, for generation
uv run python scripts/generate_hvm_obj.py                   # 90 test stimuli
```

Paths live in `config_const.py`, stimuli in `stimuli/hvm_nofixation/`, and `cache/` is built by the scripts in `scripts/`. Reconstructions land in `outputs/generate_hvm_obj/{full,crop}/`. Best sweep config: `d_model=64, n_heads=4, n_layers=2, shared_dim=512`.

## Repository layout

| Path | Contents |
| --- | --- |
| `encoders/` | MultiHeadTransformer encoder definitions |
| `train/` | Training entry points for the global and object models |
| `generation/` | FLUX + IP-Adapter pipeline and attention processors |
| `eval/metrics.py` | Cosine, SSIM, LPIPS, PixCorr, 2-AFC, retrieval |
| `scripts/` | Cache builders, generation driver, standalone analyses |
| `analysis/` | Analysis notebooks |
| `data_utils/` | HVM loaders and stratified splits |
| `config_const.py` | Paths and dataset constants |

---

## Data

**Stimuli.** The HVM object set (Majaj et al., 2015) comprises 450 naturalistic images spanning 10 categories (apple, bear, car, chair, dog, elephant, head, plane, table, turtle) × 45 variations in pose, scale, and background. Stimuli are 512 × 512 pixels with a circular aperture of radius 49% of image width on a gray surround.

**Neural data.** Neuropixels 1.0 recordings were collected in the Issa Lab at Columbia University's Zuckerman Institute from marmoset ventral stream cortices during passive viewing of the HVM set. Neural responses are mean-trial spike rates pooled across sessions into a 2,330-neuron pseudo-population.

| Split | Size | Note |
| ----- | ---- | ---- |
| Train | 270  | category-stratified |
| Val   | 90   | category-stratified |
| Test  | 90   | category-stratified |

---

## Model details

A shared pre-LN transformer backbone pools the neural response over time by learned attention, projects it to a `shared_dim` latent, and reads out three L2-normalised heads: full-image SigLIP, object-crop SigLIP, and CLIP. The two SigLIP targets are split because the ventral stream encodes a position- and size-invariant representation that aligns more closely with the object crop than with the full image.

This unified V2 encoder (`encoders/multihead_v2.py`, trained by `train/train_multihead_v2.py`) is what the analyses below probe. The generation pipeline still runs the earlier design, where the same backbone and loss are trained twice as separate global and object models with one SigLIP head each.

A learned category embedding is added to the shared latent as a low-data inductive bias. The label is either supplied (`given`) or predicted from the pre-addition latent by a small classifier (`predict`). All results here use `given`, since the classifier reaches only 0.22 test accuracy. The latent before that addition doubles as an interpretable probe of the population, analyzable independently of generation.

![MultiHeadTransformer architecture](assets/fig1_architecture.png)

The loss combines per-head InfoNCE and cosine regression with a uniformity penalty on the shared latent:

$$\mathcal{L} = \sum_{h \in \{\text{sig-g},\, \text{sig-o},\, \text{clip}\}} w_h \left[\alpha \mathcal{L}^h_\text{NCE} + (1-\alpha)\mathcal{L}^h_\text{cos}\right] + w_\text{unif}\mathcal{L}_\text{unif}$$

where $\alpha$ controls the InfoNCE/cosine tradeoff and $\mathcal{L}_\text{unif}$ penalises collapsed shared latents.

The IP-Adapter's `MLPProjModel` maps SigLIP 1152-d to 128 × 4096 image tokens, conditioning all 57 FLUX transformer blocks via installed `IPAFluxAttnProcessor` key/value projections.

---

## Results

**Conditions.** Every reconstruction is generated under four conditions, so neural conditioning can be read against both a floor and a ceiling.

| Condition | T5 | CLIP | SigLIP |
| --- | --- | --- | --- |
| Control | null | null | disabled |
| Text (cat CLIP + T5) | category T5 | category CLIP | disabled |
| Neural pred | zeroed | ẑ_clip (obj model) | ẑ_sig_obj → ẑ_sig_global |
| GT emb (upper bound) | zeroed | category CLIP | GT SigLIP crop → GT SigLIP full |

**The encoders generalize.** All three heads reach well above chance on held-out stimuli, confirming that 270 training examples suffice to recover stimulus-specific embedding information from noisy pseudo-population responses.

| Head | cos sim | 2-AFC | top-1 | top-5 |
| --- | --- | --- | --- | --- |
| SigLIP global | 0.8526 | 0.9649 | 0.222 | 0.711 |
| SigLIP object | 0.8744 | 0.9596 | 0.178 | 0.644 |
| CLIP | 0.9845 | — | 0.078 | 0.511 |

*Unified V2 encoder (`cat_mode=given`), test split, 90 stimuli. Chance is 0.5 for 2-AFC and 0.011 for top-1. The reconstructions below were generated from the two separate global and object encoders.*

**Neural conditioning matches or beats text conditioning.** Across the 90 test stimuli, neural reconstructions surpass text-conditioned ones on pixel and perceptual fidelity, even though the text condition receives the ground-truth category label through both T5 and CLIP. The neural pathway must recover that semantic content from a small, noisy population and still comes out ahead.

| Full image | Control | Text | Neural | GT (upper bound) |
| --- | --- | --- | --- | --- |
| SSIM ↑ | 0.7589 | 0.7599 | **0.7835** | 0.7809 |
| LPIPS ↓ | 0.3238 | 0.3223 | **0.3196** | 0.3131 |
| PixCorr ↑ | 0.9279 | 0.9284 | **0.9377** | 0.9370 |
| SigLIP cos ↑ | 0.7919 | 0.8226 | 0.8275 | 0.8504 |

*Neural vs text, paired over 90 stimuli: SSIM +0.024 (p = 1.3e-36), PixCorr +0.009 (p = 2.8e-20). Full breakdown including the object-crop view in `analysis/quantitative_conditions.ipynb`.*

**Within-category gesture is recovered.** On stimuli where pose, viewpoint, or spatial configuration vary within a category, neural reconstructions track the object's gesture and surface texture more closely than the text condition. The text condition is blind to pose and can misplace the foreground object onto scene background. These reconstructions capture stimulus-specific visual structure that goes beyond category membership.

**Failures localize what the population misses.** The gap between neural and GT conditioning directly reveals what is present in the image but absent from the predicted embeddings. Two patterns recur: on faces, GT conditioning recovers fine detail while neural conditioning produces a face-shaped object without recognizable features, and on elephants, both text and GT conditioning succeed while neural conditioning consistently struggles, suggesting that this category is particularly difficult for the population to encode.

## Probing the decoded representation

The analysis notebooks ask what the decoded embeddings actually contain, beyond whether the reconstructions look right. Two findings stand out.

**Identity survives the category-centroid control.** A baseline that predicts each image's category centroid scores 0.93 overall 2-AFC, beating the decoder, because getting the category right settles most pairwise comparisons. Within-category 2-AFC removes that advantage, placing centroids at exactly 0.500 by construction. The global SigLIP decoder reaches **0.712** there, confirming that it carries genuine image identity beyond category. The object and CLIP heads fall off sharply (0.599, 0.500).

**The usable subspace is only ~10-dimensional.** Against a permutation null, 10 components of the predicted-to-ground-truth alignment survive, capturing 87.6% of ground-truth embedding variance. That subspace is category-structured: between-category axes (animate, face, vehicle) are 93 to 97% aligned with it, while within-category pose axes (rotation, size, translation) are only 3 to 18% aligned and fall mostly in the discarded remainder. Linear probes corroborate this pattern, recovering category from the shared latent (0.607 vs 0.100 chance) but not pose, scale, or translation.

The reconstructions therefore rest on a low-dimensional, category-dominated code that still carries measurable within-category identity. The gesture recovery visible in Figure 3A is real but sits at the edge of what the population supports, which explains why it appears in reconstructions more clearly than in pose probes.

## Limitations

The FLUX LoRA is trained on the same 270 images the encoders use, so the generator has seen the stimulus domain even though it never sees the held-out test images. Reconstruction quality therefore reflects domain adaptation alongside the decoded neural signal.

The encoder's usable signal is narrower than the reconstructions suggest, as the subspace and probe results show. Recovering pose reliably would likely require a richer neural signal rather than a better readout.

Evaluation covers the 90 held-out stimuli, not the full 450.

## References

- N. J. Majaj, H. Hong, E. A. Solomon, and J. J. DiCarlo. Simple learned weighted sums of inferior temporal neuronal firing rates accurately predict human core object recognition performance. *Journal of Neuroscience*, 35(39):13402–13418, 2015.
- Y. Takagi and S. Nishimoto. High-resolution image reconstruction with latent diffusion models from human brain activity. *CVPR*, 2023.
- P. Scotti et al. MindEye2: Shared-subject models enable fMRI-to-image with 1 hour of data. *arXiv:2403.11207*, 2024.
- M. Ciferri, M. Ferrante, and N. Toschi. Simple models, rich representations: Visual decoding from primate intracortical neural signals. *arXiv:2601.11108*, 2026.
- H. Ye et al. IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv:2308.06721*, 2023.
