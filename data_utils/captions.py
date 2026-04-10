"""BLIP-2 auto-captions for the stimulus set (text-hybrid ablation)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from config_const import N_STIMULI, RUST_STIM_DIR, BLIP2_CAPTIONS_PATH, BLIP2_DETAILED_CAPTIONS_PATH
from get_device import get_device


_PROMPT = "a color photograph of"


def caption_stimuli(
    model_name: str = "Salesforce/blip2-opt-2.7b",
    batch_size: int = 32,
    min_new_tokens: int = 0,
    max_new_tokens: int = 100,
    force: bool = False,
    save_path: Path = BLIP2_CAPTIONS_PATH,
) -> dict[int, str]:
    """Generate BLIP-2 captions for all stimuli and cache them to disk.

    Args:
        model_name: HuggingFace model ID for BLIP-2.
        batch_size: Number of images to process per forward pass.
        max_new_tokens: Maximum number of tokens to generate per caption.
        force: Re-run even if the cache file already exists.

    Returns:
        Dict mapping stimulus index (0-based) to caption string.
    """
    if save_path.exists() and not force:
        with open(save_path) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    device = get_device()
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name)
    model.to(device).eval()

    images: list[Image.Image] = []
    for i in range(N_STIMULI):
        path = RUST_STIM_DIR / f"{i:04d}.png"
        images.append(Image.open(path).convert("RGB"))

    captions: dict[int, str] = {}
    pbar = tqdm(total=N_STIMULI, desc="Generating captions")
    for start in range(0, N_STIMULI, batch_size):
        batch = images[start : start + batch_size]
        prompts = [_PROMPT] * len(batch)
        inputs = processor(images=batch, text=prompts, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                repetition_penalty=1.3,
            )
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        for offset, text in enumerate(texts):
            captions[start + offset] = text.strip()
        pbar.update(len(batch))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"saved captions to {save_path}")

    return captions


if __name__ == "__main__":
    caps = caption_stimuli(force=True)
    detailed_caps = caption_stimuli(
        force=True, min_new_tokens=40, save_path=BLIP2_DETAILED_CAPTIONS_PATH
    )
    for i in range(min(5, len(caps))):
        print(f"{i:04d}: {caps[i]}")
        print(f"      {detailed_caps[i]}")
