"""Project-wide config constants: seeds, paths, dataset sizes."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / "cache"
STIMULI_ROOT = REPO_ROOT / "stimuli"
RUST_SRC_NAME = "20231025_Rust_NaturalImages300_300ms"
RUST_DST_NAME = "rust_cropped"
RUST_STIM_DIR = STIMULI_ROOT / RUST_DST_NAME
RUST_BG_THRESHOLD = 5  # pixels <= this on all channels are background (black)
CLIP_EMBEDS_PATH       = CACHE_DIR / "clip_embeds.pt"       # (300, 768)      CLIP pooled
T5_EMBEDS_PATH         = CACHE_DIR / "t5_embeds.pt"         # (300, 512, 4096) T5 sequence
SIGLIP_EMBEDDINGS_PATH = CACHE_DIR / "siglip_embeddings.pt"
SIGLIP_STRIPPED_PATH   = CACHE_DIR / "siglip_stripped.pt"  # (300, 1152) aperture-stripped
SIGLIP_PATCH14_PATH    = CACHE_DIR / "siglip_patch14.pt"   # (300, 196, 1152)
SIGLIP_PATCH8_PATH     = CACHE_DIR / "siglip_patch8.pt"    # (300,  64, 1152)
SIGLIP_PATCH_TOKENS    = {14: 196, 8: 64}
RUST_LOSS_MASK_PATH = RUST_STIM_DIR / "loss_mask.pt"
BLIP2_CAPTIONS_PATH = CACHE_DIR / "blip2_captions.json"
BLIP2_DETAILED_CAPTIONS_PATH = CACHE_DIR / "blip2_detailed_captions.json"

SEED = 42

N_STIMULI = 300
N_TRAIN = 200
N_VAL = 50
N_TEST = 50

RUST_TIME_WINDOW = (0, 250)

SIGLIP_DIM = 1152  # google/siglip-so400m-patch14-384
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
