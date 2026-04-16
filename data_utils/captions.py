"""InstructBLIP captions for the stimulus set (text-hybrid ablation).

Replaces the old BLIP-2 pipeline with InstructBLIP (Vicuna-7B), which follows
text instructions reliably and produces consistent caption length and style.

Two caption modes:
  short    — one compact sentence naming the main subject and scene (~10-15 words),
             designed to fit within CLIP's 77-token limit.
  detailed — two to three sentences covering subject, setting, visual attributes,
             and notable details (~40-60 words), still within CLIP's 77-token limit.

Stimuli properties the prompts account for:
  - Grayscale (no colour information available)
  - Circular crop with black corners (background is masked, not informative)
  - Single central object or scene in focus
  - Wide variety: everyday objects, animals, people, signs, outdoor scenes
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

from config_const import N_STIMULI, RUST_STIM_DIR, BLIP2_CAPTIONS_PATH, BLIP2_DETAILED_CAPTIONS_PATH
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

    if save_path.exists() and not force:
        with open(save_path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    device = get_device()
    processor = InstructBlipProcessor.from_pretrained(_MODEL_NAME)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        _MODEL_NAME, torch_dtype=torch.float16,
    )
    model.to(device).eval()

    images: list[Image.Image] = []
    for i in range(N_STIMULI):
        path = RUST_STIM_DIR / f"{i:04d}.png"
        images.append(Image.open(path).convert("RGB"))

    captions: dict[int, str] = {}
    pbar = tqdm(total=N_STIMULI, desc=f"Generating {mode} captions")
    for start in range(0, N_STIMULI, batch_size):
        batch = images[start : start + batch_size]
        prompts = [prompt] * len(batch)
        inputs = processor(
            images=batch,
            text=prompts,
            return_tensors="pt",
            padding=True,
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
            # InstructBLIP echoes the prompt; strip it by finding where the
            # prompt ends. The prompt text appears verbatim at the start.
            stripped = text[text.find(prompts[offset]) + len(prompts[offset]):].strip() if prompts[offset] in text else text.strip()
            # The model ignores negative constraints; remove forbidden phrases in post-processing.
            # Remove circular crop / circle references (whole sentence)
            stripped = re.sub(r"[^.]*circular crop[^.]*\.", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"[^.]*circle around[^.]*\.", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"[^.]*circular shape[^.]*\.", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"[^.]*cropped in a circle[^.]*\.", "", stripped, flags=re.IGNORECASE)
            # Remove "black and white / grayscale / monochrome" photo references (inline)
            stripped = re.sub(r"a (black and white|grayscale|monochrome) (photo|photograph|image|picture) of ", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r" in a (black and white|grayscale|monochrome) (photo|photograph|image|picture)\b", "", stripped, flags=re.IGNORECASE)
            # Remove whole sentences that are just about the image being black and white
            stripped = re.sub(r"[^.]*\b(black and white|grayscale|monochrome)\b[^.]*giving it[^.]*\.", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"[^.]*\bthe (photo|image|photograph) is in (black and white|grayscale|monochrome)[^.]*\.", "", stripped, flags=re.IGNORECASE)
            # Remove "the (main) subject is" at sentence boundaries
            stripped = re.sub(r"(^|\. )The (main )?subject is ", lambda m: m.group(1), stripped, flags=re.IGNORECASE)
            # Remove filler "there are/may be no other objects" sentences
            stripped = re.sub(r"[^.]*there (are|may be) (no )?other objects[^.]*\.", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s+", " ", stripped).strip(" .")
            captions[start + offset] = stripped
        pbar.update(len(batch))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Saved {mode} captions to {save_path}")

    return captions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["short", "detailed", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    modes = ["short", "detailed"] if args.mode == "both" else [args.mode]
    for mode in modes:
        caps = caption_stimuli(mode=mode, force=args.force)
        print(f"\n--- {mode} sample ---")
        for i in range(min(5, len(caps))):
            print(f"  {i:04d}: {caps[i]}")
