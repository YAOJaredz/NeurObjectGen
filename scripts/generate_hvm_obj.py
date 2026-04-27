"""
Generate HVM object-specific reconstructions for all test stimuli.

For each stimulus, saves one PNG with 10 side-by-side columns:
  Full image: Original | Control | Text (cat CLIP) | Neural pred (global+obj) | GT emb (↑)
  Obj crop:   Crop     | Control | Text (cat CLIP) | Neural pred (obj)        | GT emb (↑)

Global SigLIP comes from the full-image multihead model; obj-SigLIP comes from
the obj multihead model (trained on bbox-crop embeddings) and is injected into
the bbox-masked object slot. CLIP pooled embeddings come from the obj model.

Output: outputs/generate_hvm_obj/stim{idx:04d}.png
"""
import sys, json, argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config_const import (
    CACHE_DIR, SEED,
    HVM_STIM_DIR, HVM_N_STIMULI, HVM_N_VAR, HVM_N_CAT, HVM_CATEGORIES,
    HVM_SIGLIP_EMBEDDINGS_PATH, HVM_OBJ_SIGLIP_EMBEDDINGS_PATH,
    HVM_CLIP_EMBEDS_PATH, SIGLIP_DIM,
)
from data_utils.hvm_loader import _category_stratified_split, _load_hvm_neural
from encoders import MultiHeadTransformer
from generation.flux_instantx import load_pipeline, generate_img2img, encode_text_embeds
from generation.aperture import build_object_region_mask, load_hvm_packed_aperture_mask
from generation.project_hvm import load_hvm_bboxes
from get_device import get_device

STRENGTH     = 0.55
OBJ_STRENGTH = 0.75
IP_SCALE     = 1.0
OBJ_SCALE    = 0.75
NUM_STEPS    = 20
IMAGE_SIZE   = 512
BBOX_SRC     = 276

LABEL_H = 36

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


def get_object_mask(stim_idx: int, bboxes, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    b = bboxes[stim_idx]
    scale = image_size / BBOX_SRC
    half_px = b['half'] * scale + 5
    return build_object_region_mask(
        image_size=image_size,
        cx_px=b['cx'] * scale, cy_px=b['cy'] * scale,
        r_px=0, device='cpu', dtype=torch.bfloat16,
        bbox_frac=half_px * 2 / image_size,
    )


def get_crop(stim_idx: int, orig: Image.Image, bboxes, image_size: int = IMAGE_SIZE) -> Image.Image:
    b = bboxes[stim_idx]
    scale = image_size / BBOX_SRC
    pad = 5
    x0 = max(0, int((b['cx'] - b['half']) * scale - pad))
    y0 = max(0, int((b['cy'] - b['half']) * scale - pad))
    x1 = min(image_size, int((b['cx'] + b['half']) * scale + pad))
    y1 = min(image_size, int((b['cy'] + b['half']) * scale + pad))
    return orig.crop((x0, y0, x1, y1)).resize((image_size, image_size), Image.LANCZOS)


def make_model(cfg, n_neurons):
    m = MultiHeadTransformer(
        n_neurons=n_neurons,
        d_model=int(cfg['d_model']), n_heads=int(cfg['n_heads']),
        n_layers=int(cfg['n_layers']), shared_dim=int(cfg['shared_dim']),
        dropout=float(cfg['dropout']), n_categories=HVM_N_CAT,
    )
    ckpt = torch.load(cfg['checkpoint'], weights_only=False, map_location='cpu')
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    return m


def predict_all(model, neural_tensor, cat_indices):
    sigs, clips = [], []
    with torch.no_grad():
        for i in range(HVM_N_STIMULI):
            x   = neural_tensor[i:i+1].permute(0, 2, 1)
            cat = cat_indices[i:i+1]
            out = model(x, cat)
            sigs.append(out['siglip'].cpu())
            clips.append(out['clip'].cpu())
    return torch.cat(sigs), torch.cat(clips)


def main(stim_limit):
    device = get_device()
    torch.cuda.empty_cache()

    # ── neural encoders ───────────────────────────────────────────────────────
    cfg_global = json.loads((CACHE_DIR / 'best_hvm_multihead_config.json').read_text())
    cfg_obj    = json.loads((CACHE_DIR / 'best_hvm_multihead_obj_config.json').read_text())

    rsp = _load_hvm_neural()
    _, n_neurons, _ = rsp.shape
    neural_tensor = torch.from_numpy(rsp).float()
    cat_indices   = torch.arange(HVM_N_STIMULI) // HVM_N_VAR

    model_global = make_model(cfg_global, n_neurons)
    model_obj    = make_model(cfg_obj,    n_neurons)

    print("Predicting embeddings...")
    neural_pred_sig,     _                = predict_all(model_global, neural_tensor, cat_indices)
    neural_pred_sig_obj, neural_pred_clip = predict_all(model_obj,    neural_tensor, cat_indices)

    # ── GT embeddings ─────────────────────────────────────────────────────────
    siglip_gt     = torch.load(HVM_SIGLIP_EMBEDDINGS_PATH,     weights_only=True)
    siglip_obj_gt = torch.load(HVM_OBJ_SIGLIP_EMBEDDINGS_PATH, weights_only=True)
    clip_gt       = torch.load(HVM_CLIP_EMBEDS_PATH,            weights_only=True)

    bboxes = load_hvm_bboxes()
    _, _, test_idx = _category_stratified_split(SEED)
    if stim_limit:
        test_idx = test_idx[:stim_limit]

    # ── FLUX pipeline ─────────────────────────────────────────────────────────
    pipe, image_proj = load_pipeline(device=device, default_scale=1.0)
    hvm_aperture = load_hvm_packed_aperture_mask(image_size=IMAGE_SIZE, device='cpu', dtype=torch.bfloat16)

    cat_clip_embeds, cat_t5_embeds = encode_text_embeds(pipe, list(HVM_CATEGORIES))
    _, null_t5 = encode_text_embeds(pipe, [''])
    zero_siglip = torch.zeros(SIGLIP_DIM)
    zero_t5     = null_t5

    def cat_text_embeds(stim_idx):
        cat_i = int(stim_idx) // HVM_N_VAR
        return cat_t5_embeds[cat_i:cat_i+1], cat_clip_embeds[cat_i:cat_i+1]

    # ── generate ──────────────────────────────────────────────────────────────
    full_dir = Path('outputs/generate_hvm_obj/full')
    crop_dir = Path('outputs/generate_hvm_obj/crop')
    full_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    full_labels = ['Original', 'Control\n(null)', 'Text\n(cat CLIP)', 'Neural pred\n(global+obj)', 'GT emb\n(global+obj ↑)']
    crop_labels = ['Crop',     'Control\n(null)', 'Text\n(cat CLIP)', 'Neural pred\n(obj)',         'GT emb\n(obj ↑)']

    for i, stim_idx in enumerate(tqdm(
        (int(s) for s in test_idx), desc='Saving', total=len(test_idx)
    )):
        orig     = load_stimulus(stim_idx)
        obj_mask = get_object_mask(stim_idx, bboxes)
        crop_pil = get_crop(stim_idx, orig, bboxes)
        _, clip_cat = cat_text_embeds(stim_idx)

        full_base = dict(
            height=IMAGE_SIZE, width=IMAGE_SIZE, num_inference_steps=NUM_STEPS,
            guidance_scale=3.5, strength=STRENGTH, seed=i,
            aperture_mask=hvm_aperture, show_progress=False,
        )
        crop_base = dict(
            height=IMAGE_SIZE, width=IMAGE_SIZE, num_inference_steps=NUM_STEPS,
            guidance_scale=3.5, strength=OBJ_STRENGTH, seed=i,
            aperture_composite=False, show_progress=False,
        )

        # full-image conditions
        f_control = generate_img2img(
            pipe, image_proj, orig, zero_siglip,
            ip_adapter_scale=0.0, prompt='', **full_base)
        f_text = generate_img2img(
            pipe, image_proj, orig, zero_siglip,
            ip_adapter_scale=0.0,
            prompt_embeds=zero_t5, pooled_prompt_embeds=clip_cat, **full_base)
        f_neural = generate_img2img(
            pipe, image_proj, orig, neural_pred_sig[stim_idx],
            ip_adapter_scale=0.25,
            object_siglip_embedding=neural_pred_sig_obj[stim_idx],
            object_ip_scale=OBJ_SCALE, object_mask=obj_mask,
            prompt_embeds=zero_t5,
            pooled_prompt_embeds=neural_pred_clip[stim_idx].unsqueeze(0), **full_base)
        f_gt = generate_img2img(
            pipe, image_proj, orig, siglip_gt[stim_idx],
            ip_adapter_scale=0.25,
            object_siglip_embedding=siglip_obj_gt[stim_idx],
            object_ip_scale=OBJ_SCALE, object_mask=obj_mask,
            prompt_embeds=zero_t5, pooled_prompt_embeds=clip_cat, **full_base)

        # obj-crop conditions
        c_control = generate_img2img(
            pipe, image_proj, crop_pil, zero_siglip,
            ip_adapter_scale=0.0, prompt='', **crop_base)
        c_text = generate_img2img(
            pipe, image_proj, crop_pil, zero_siglip,
            ip_adapter_scale=0.0,
            prompt_embeds=zero_t5, pooled_prompt_embeds=clip_cat, **crop_base)
        c_neural = generate_img2img(
            pipe, image_proj, crop_pil, neural_pred_sig_obj[stim_idx],
            ip_adapter_scale=IP_SCALE,
            prompt_embeds=zero_t5,
            pooled_prompt_embeds=neural_pred_clip[stim_idx].unsqueeze(0), **crop_base)
        c_gt = generate_img2img(
            pipe, image_proj, crop_pil, siglip_obj_gt[stim_idx],
            ip_adapter_scale=IP_SCALE,
            prompt_embeds=zero_t5,
            pooled_prompt_embeds=clip_gt[stim_idx].unsqueeze(0), **crop_base)

        add_labels([orig, f_control, f_text, f_neural, f_gt], full_labels).save(
            full_dir / f'stim{stim_idx:04d}.png')
        add_labels([crop_pil, c_control, c_text, c_neural, c_gt], crop_labels).save(
            crop_dir / f'stim{stim_idx:04d}.png')

    print(f'Saved {len(test_idx)} images to {full_dir} and {crop_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=None,
                        help='limit to first N test stimuli (default: all)')
    args = parser.parse_args()
    main(args.n)
