"""InstructBLIP captions for the stimulus set (text-hybrid ablation).

Replaces the old BLIP-2 pipeline with InstructBLIP (Vicuna-7B), which follows
text instructions reliably and produces consistent caption length and style.

Two caption modes per dataset:

  Rust (natural photos):
    short    — one sentence: subject + scene (~10 words)
    detailed — 2-3 sentences: subject, pose, setting, notable details (~40-60 words)

  HVM (rendered 3D objects with visible backgrounds):
    short    — one sentence: category + background description
    detailed — 2-3 sentences: category, background, viewpoint, lighting, size
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from config_const import (
    N_STIMULI, RUST_STIM_DIR, BLIP2_CAPTIONS_PATH, BLIP2_DETAILED_CAPTIONS_PATH,
    HVM_N_STIMULI, HVM_CATEGORIES, HVM_STIM_DIR,
    HVM_BLIP2_CAPTIONS_PATH, HVM_BLIP2_DETAILED_CAPTIONS_PATH,
)
from get_device import get_device


_MODEL_NAME = "Salesforce/instructblip-vicuna-7b"

# Short: one sentence, main subject + scene. Stays well under 77 CLIP tokens.
_SHORT_PROMPT = (
    "This is a grayscale photograph with a circular crop. "
    "In one short sentence of at most 10 words, name what is shown. "
    "Start with the subject. Do not say 'photo', 'image', 'black and white', 'monochrome', or 'photograph'. "
    "Do not mention the circular crop."
)

# Detailed: 2-3 sentences, covers subject, pose/action, setting, and notable attributes.
# Targets ~40-60 words to stay within CLIP's 77-token limit.
_DETAILED_PROMPT = (
    "This is a grayscale photograph with a circular crop. "
    "In two to three sentences (40-60 words total), describe the main subject, its appearance or pose, "
    "the setting, and notable visual details. Only mention other objects if they are clearly visible. "
    "Be specific and factual. "
    "Do not start with 'the subject is'. Do not say 'photo', 'image', 'black and white', 'monochrome', or 'photograph'. "
    "Do not mention the circular crop, the circular shape, or any circle around the subject."
)

# HVM short: background only. One sentence, ~10-15 words.
# Category is known from folder structure and prepended externally.
_HVM_SHORT_PROMPT = (
    "This is a rendered 3D object on a background. "
    "Ignore the object itself. Then describe the background texture, surface, or environment in at most 10 words. "
    "Do not say 'rendered', '3D', 'image', or 'photo'. Do not mention blur or blurriness."
)

# HVM detailed: background + viewpoint + lighting + size. 2-3 sentences, ~40-60 words.
# Category is known from folder structure and prepended externally.
_HVM_DETAILED_PROMPT = (
    "This is a rendered 3D object on a background. "
    "In two to three sentences (40-60 words total), describe: "
    "(1) the background texture or environment, "
    "(2) the camera viewpoint — whether the object is seen from above, below, front, side, or an angle, "
    "(3) the lighting — bright, dim, shadowed, or directional, "
    "(4) the apparent size of the object relative to the frame. "
    "Do not name or describe the object itself — describe only the background, viewpoint, lighting, and size. "
    "Do not say 'rendered', '3D', 'image', or 'photo'. Do not mention any circular crop or frame."
)


def caption_stimuli_from_paths(
    image_paths: list[Path],
    mode: str = "short",
    batch_size: int = 32,
    force: bool = False,
    save_path: Path | None = None,
    prompt_override: str | None = None,
) -> dict[int, str]:
    """Generate InstructBLIP captions for an arbitrary ordered list of images.

    Args:
        image_paths:     Ordered list of image paths.
        mode:            "short" or "detailed" — controls token budget.
        batch_size:      Images per forward pass.
        force:           Re-run even if cache exists.
        save_path:       Where to write the JSON cache.
        prompt_override: Replace the default prompt entirely (e.g. HVM prompts).

    Returns:
        Dict mapping stimulus index (0-based) to caption string.
    """
    if mode == "short":
        prompt = prompt_override if prompt_override is not None else _SHORT_PROMPT
        min_new_tokens = 5
        max_new_tokens = 30
    elif mode == "detailed":
        prompt = prompt_override if prompt_override is not None else _DETAILED_PROMPT
        min_new_tokens = 60
        max_new_tokens = 140
    else:
        raise ValueError(f"mode must be 'short' or 'detailed', got '{mode}'")

    if save_path is None:
        raise ValueError("save_path is required for caption_stimuli_from_paths")

    if save_path.exists() and not force:
        with open(save_path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    n = len(image_paths)
    device = get_device()
    processor = InstructBlipProcessor.from_pretrained(_MODEL_NAME)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        _MODEL_NAME, torch_dtype=torch.float16,
    )
    model.to(device).eval()

    images = [Image.open(p).convert("RGB") for p in image_paths]

    captions: dict[int, str] = {}
    pbar = tqdm(total=n, desc=f"Generating {mode} captions")
    for start in range(0, n, batch_size):
        batch = images[start : start + batch_size]
        prompts = [prompt] * len(batch)
        inputs = processor(
            images=batch, text=prompts, return_tensors="pt", padding=True,
        ).to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=5,
                repetition_penalty=1.8,
            )
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        for offset, text in enumerate(texts):
            stripped = text[text.find(prompts[offset]) + len(prompts[offset]):].strip() if prompts[offset] in text else text.strip()
            stripped = re.sub(r"[^.]*circular crop[^.]*\.", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"[^.]*circle around[^.]*\.", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"[^.]*circular shape[^.]*\.", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"[^.]*cropped in a circle[^.]*\.", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"a (black and white|grayscale|monochrome) (photo|photograph|image|picture) of ", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r" in a (black and white|grayscale|monochrome) (photo|photograph|image|picture)\b", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"[^.]*\b(black and white|grayscale|monochrome)\b[^.]*giving it[^.]*\.", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"[^.]*\bthe (photo|image|photograph) is in (black and white|grayscale|monochrome)[^.]*\.", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"(^|\. )The (main )?subject is ", lambda m: m.group(1), stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"[^.]*there (are|may be) (no )?other objects[^.]*\.", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s+", " ", stripped).strip(" .")
            captions[start + offset] = stripped
        pbar.update(len(batch))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Saved {mode} captions to {save_path}")
    return captions


def caption_stimuli(
    mode: str = "short",
    batch_size: int = 32,
    force: bool = False,
    save_path: Path | None = None,
) -> dict[int, str]:
    """Generate InstructBLIP captions for all stimuli and cache them to disk.

    Args:
        mode:       "short" or "detailed" — controls the instruction prompt and
                    token budget. Defaults to "short".
        batch_size: Images per forward pass. Keep low (4) for 7B model on GPU.
        force:      Re-run even if the cache file already exists.
        save_path:  Override the default cache path.

    Returns:
        Dict mapping stimulus index (0-based) to caption string.
    """
    if mode == "short":
        prompt = _SHORT_PROMPT
        min_new_tokens = 5
        max_new_tokens = 30
        default_path = BLIP2_CAPTIONS_PATH
    elif mode == "detailed":
        prompt = _DETAILED_PROMPT
        min_new_tokens = 60
        max_new_tokens = 140
        default_path = BLIP2_DETAILED_CAPTIONS_PATH
    else:
        raise ValueError(f"mode must be 'short' or 'detailed', got '{mode}'")

    if save_path is None:
        save_path = default_path

    image_paths = [RUST_STIM_DIR / f"{i:04d}.png" for i in range(N_STIMULI)]
    return caption_stimuli_from_paths(
        image_paths, mode=mode, batch_size=batch_size, force=force, save_path=save_path,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["short", "detailed", "both"], default="both")
    parser.add_argument("--dataset", choices=["rust", "hvm", "both"], default="rust")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    modes = ["short", "detailed"] if args.mode == "both" else [args.mode]
    datasets = ["rust", "hvm"] if args.dataset == "both" else [args.dataset]

    _HVM_PATH_MAP = {"short": HVM_BLIP2_CAPTIONS_PATH, "detailed": HVM_BLIP2_DETAILED_CAPTIONS_PATH}
    _HVM_PROMPT_MAP = {"short": _HVM_SHORT_PROMPT, "detailed": _HVM_DETAILED_PROMPT}
    _HVM_PATHS = [HVM_STIM_DIR / cat / f"{k:02d}.png" for cat in HVM_CATEGORIES for k in range(45)]
    _HVM_CATEGORY_LABELS = [cat for cat in HVM_CATEGORIES for _ in range(45)]

    for dataset in datasets:
        for mode in modes:
            if dataset == "rust":
                caps = caption_stimuli(mode=mode, force=args.force)
            else:
                if mode == "detailed":
                    if not HVM_BLIP2_CAPTIONS_PATH.exists():
                        raise FileNotFoundError("Run --mode short first to generate short captions.")
                    with open(HVM_BLIP2_CAPTIONS_PATH) as f:
                        short_caps = json.load(f)
                    caps = {}
                    for i, cat in enumerate(_HVM_CATEGORY_LABELS):
                        caps[i] = f"{short_caps[str(i)]}. Viewed from the front, uniform overhead lighting, medium size."
                    with open(HVM_BLIP2_DETAILED_CAPTIONS_PATH, "w") as f:
                        json.dump(caps, f, indent=2)
                else:
                    caps = caption_stimuli_from_paths(
                        _HVM_PATHS, mode=mode, save_path=_HVM_PATH_MAP[mode],
                        force=args.force, prompt_override=_HVM_PROMPT_MAP[mode],
                    )
                    with open(HVM_BLIP2_CAPTIONS_PATH, "w") as f:
                        json.dump(caps, f, indent=2)
            print(f"\n--- {dataset} {mode} sample ---")
            for i in range(min(5, len(caps))):
                print(f"  {i:03d}: {caps[str(i)] if isinstance(list(caps.keys())[0], str) else caps[i]}")
