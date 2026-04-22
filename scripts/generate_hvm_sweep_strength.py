"""
Generate HVM condition comparisons swept over img2img strength values.

Saves one PNG per (stim_idx, strength) with five side-by-side columns:
  Original | Control (null) | Text (cat CLIP) | Neural pred | GT emb (↑)

Output layout:
  outputs/generate_hvm_strength/s{strength:.2f}/stim{idx:04d}.png
"""
import sys, json, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_const import (
    CACHE_DIR, SEED,
    HVM_STIM_DIR, HVM_N_STIMULI, HVM_N_VAR, HVM_N_CAT, HVM_CATEGORIES,
    HVM_SIGLIP_EMBEDDINGS_PATH, HVM_CLIP_EMBEDS_PATH, HVM_BLIP2_CAPTIONS_PATH,
    SIGLIP_DIM,
)
from data_utils.hvm_loader import _category_stratified_split, _load_hvm_neural
from encoders import MultiHeadTransformer
from generation.flux_instantx import load_pipeline, generate_img2img, encode_text_embeds
from generation.aperture import load_hvm_packed_aperture_mask
from get_device import get_device

STRENGTHS    = [0.35, 0.45, 0.55, 0.65, 0.75]
NUM_STEPS    = 20
IMAGE_SIZE   = 512
IP_SCALE     = 1.0
GUIDANCE     = 3.5

SAVE_LABELS  = ['Original', 'Control\n(null)', 'Text\n(cat CLIP)',
                 'Neural pred\n(pred CLIP+SigLIP)', 'GT emb (↑)']
LABEL_H      = 36

try:
    _font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
except Exception:
    _font = ImageFont.load_default()


def add_labels(images, labels, label_h=LABEL_H):
    w, h = images[0].size
    canvas = Image.new('RGB', (w * len(images), h + label_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    for j, (img, lbl) in enumerate(zip(images, labels)):
        canvas.paste(img, (w * j, label_h))
        lines = lbl.split('\n')
        line_h = label_h // max(len(lines), 1)
        for li, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=_font)
            tw = bbox[2] - bbox[0]
            draw.text((w * j + (w - tw) // 2, li * line_h + 2), line,
                      font=_font, fill=(255, 255, 255))
    return canvas


def load_stimulus(global_idx: int, size: int = IMAGE_SIZE) -> Image.Image:
    cat = HVM_CATEGORIES[global_idx // HVM_N_VAR]
    var = global_idx % HVM_N_VAR
    return Image.open(HVM_STIM_DIR / cat / f"{var:02d}.png").convert("RGB").resize(
        (size, size), Image.LANCZOS
    )


def main(strengths, stim_limit):
    device = get_device()
    torch.cuda.empty_cache()

    # ── neural encoder ────────────────────────────────────────────────────────
    cfg_path = CACHE_DIR / 'best_hvm_multihead_config.json'
    cfg = json.loads(cfg_path.read_text())
    rsp = _load_hvm_neural()
    _, n_neurons, n_time = rsp.shape

    model = MultiHeadTransformer(
        n_neurons=n_neurons,
        d_model=int(cfg['d_model']),
        n_heads=int(cfg['n_heads']),
        n_layers=int(cfg['n_layers']),
        shared_dim=int(cfg['shared_dim']),
        dropout=float(cfg['dropout']),
        n_categories=HVM_N_CAT,
    )
    ckpt = torch.load(cfg['checkpoint'], weights_only=False, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    neural_tensor = torch.from_numpy(rsp).float()
    cat_indices   = torch.arange(HVM_N_STIMULI) // HVM_N_VAR

    @torch.no_grad()
    def predict_all():
        sigs, clips = [], []
        for i in range(HVM_N_STIMULI):
            x   = neural_tensor[i:i+1].permute(0, 2, 1)
            cat = cat_indices[i:i+1]
            out = model(x, cat)
            sigs.append(out['siglip'].cpu())
            clips.append(out['clip'].cpu())
        return torch.cat(sigs), torch.cat(clips)

    neural_pred_sig, neural_pred_clip = predict_all()

    # ── GT embeddings ─────────────────────────────────────────────────────────
    siglip_gt = torch.load(HVM_SIGLIP_EMBEDDINGS_PATH, weights_only=True)
    clip_gt   = F.normalize(torch.load(HVM_CLIP_EMBEDS_PATH, weights_only=True), dim=-1)

    _, _, test_idx = _category_stratified_split(SEED)
    if stim_limit:
        test_idx = test_idx[:stim_limit]

    # ── FLUX pipeline ─────────────────────────────────────────────────────────
    pipe, image_proj = load_pipeline(device=device, default_scale=1.0)

    cat_clip_embeds, cat_t5_embeds = encode_text_embeds(pipe, list(HVM_CATEGORIES))
    null_clip, null_t5 = encode_text_embeds(pipe, [''])
    zero_siglip = torch.zeros(SIGLIP_DIM)
    zero_t5     = null_t5

    hvm_aperture = load_hvm_packed_aperture_mask(
        image_size=IMAGE_SIZE, device='cpu', dtype=torch.bfloat16
    )

    def cat_text_embeds(stim_idx):
        cat_i = int(stim_idx) // HVM_N_VAR
        return cat_t5_embeds[cat_i:cat_i+1], cat_clip_embeds[cat_i:cat_i+1]

    # ── sweep ─────────────────────────────────────────────────────────────────
    base_out = Path('outputs/generate_hvm_strength')

    for strength in strengths:
        out_dir = base_out / f's{strength:.2f}'
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, stim_idx in enumerate(tqdm(
            (int(s) for s in test_idx),
            desc=f'strength={strength:.2f}',
            total=len(test_idx),
        )):
            orig = load_stimulus(stim_idx)
            _, clip_stim = cat_text_embeds(stim_idx)

            shared = dict(
                height=IMAGE_SIZE, width=IMAGE_SIZE,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE,
                strength=strength,
                seed=i,
                aperture_mask=hvm_aperture,
                show_progress=False,
            )

            gen_control = generate_img2img(
                pipe, image_proj, orig, zero_siglip,
                ip_adapter_scale=0.0, prompt='', **shared)

            gen_text = generate_img2img(
                pipe, image_proj, orig, zero_siglip,
                ip_adapter_scale=0.0,
                prompt_embeds=zero_t5,
                pooled_prompt_embeds=clip_stim, **shared)

            gen_neural = generate_img2img(
                pipe, image_proj, orig, neural_pred_sig[stim_idx],
                ip_adapter_scale=IP_SCALE,
                prompt_embeds=zero_t5,
                pooled_prompt_embeds=neural_pred_clip[stim_idx].unsqueeze(0), **shared)

            gen_gt = generate_img2img(
                pipe, image_proj, orig, siglip_gt[stim_idx],
                ip_adapter_scale=IP_SCALE,
                prompt_embeds=zero_t5,
                pooled_prompt_embeds=clip_stim, **shared)

            combined = add_labels(
                [orig, gen_control, gen_text, gen_neural, gen_gt],
                SAVE_LABELS,
            )
            combined.save(out_dir / f'stim{stim_idx:04d}.png')

        print(f'Saved {len(test_idx)} images → {out_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--strengths', nargs='+', type=float, default=STRENGTHS,
        help='img2img strength values to sweep (default: %(default)s)',
    )
    parser.add_argument(
        '--n', type=int, default=None,
        help='limit to first N test stimuli (default: all)',
    )
    args = parser.parse_args()
    main(args.strengths, args.n)
