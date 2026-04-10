"""Project-wide config constants: seeds, paths, dataset sizes."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / "cache"
RUST_STIM_DIR = REPO_ROOT / "stimuli" / "rust_cropped"
SIGLIP_EMBEDDINGS_PATH = CACHE_DIR / "siglip_embeddings.pt"
BLIP2_CAPTIONS_PATH = CACHE_DIR / "blip2_captions.json"
BLIP2_DETAILED_CAPTIONS_PATH = CACHE_DIR / "blip2_detailed_captions.json"

SEED = 42

N_STIMULI = 300
N_TRAIN = 200
N_VAL = 50
N_TEST = 50

RUST_TIME_WINDOW = (0, 250)
