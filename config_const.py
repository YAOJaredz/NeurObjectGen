"""Project-wide config constants: seeds, paths, dataset sizes."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / "cache"
STIMULI_ROOT = REPO_ROOT / "stimuli"
RUST_SRC_NAME = "20231025_Rust_NaturalImages300_300ms"
RUST_DST_NAME = "rust_cropped"
RUST_STIM_DIR = STIMULI_ROOT / RUST_DST_NAME
RUST_BG_THRESHOLD = 5  # pixels <= this on all channels are background (black)
CLIP_EMBEDS_PATH          = CACHE_DIR / "clip_embeds.pt"          # (300, 768)       CLIP pooled, short captions
T5_EMBEDS_PATH            = CACHE_DIR / "t5_embeds.pt"            # (300, 512, 4096)  T5 sequence, short captions
CLIP_DETAILED_EMBEDS_PATH = CACHE_DIR / "clip_detailed_embeds.pt" # (300, 768)       CLIP pooled, detailed captions
T5_DETAILED_EMBEDS_PATH   = CACHE_DIR / "t5_detailed_embeds.pt"   # (300, 512, 4096)  T5 sequence, detailed captions
T5_XXL_POOLED_PATH        = CACHE_DIR / "t5_xxl_pooled_detailed.pt"  # (300, 4096) mean-pooled, standalone google/t5-v1_1-xxl
SIGLIP_EMBEDDINGS_PATH = CACHE_DIR / "siglip_embeddings.pt"
SIGLIP_STRIPPED_PATH   = CACHE_DIR / "siglip_stripped.pt"  # (300, 1152) aperture-stripped
SIGLIP_PATCH14_PATH    = CACHE_DIR / "siglip_patch14.pt"   # (300, 196, 1152)
SIGLIP_PATCH8_PATH     = CACHE_DIR / "siglip_patch8.pt"    # (300,  64, 1152)
RUST_LOSS_MASK_PATH = RUST_STIM_DIR / "loss_mask.pt"
HVM_LOSS_MASK_PATH  = STIMULI_ROOT / "hvm_cropped" / "loss_mask.pt"
BLIP2_CAPTIONS_PATH = CACHE_DIR / "blip2_captions.json"
BLIP2_DETAILED_CAPTIONS_PATH = CACHE_DIR / "blip2_detailed_captions.json"

HVM_BLIP2_CAPTIONS_PATH          = CACHE_DIR / "hvm_blip2_captions.json"
HVM_BLIP2_DETAILED_CAPTIONS_PATH  = CACHE_DIR / "hvm_blip2_detailed_captions.json"
HVM_SIGLIP_EMBEDDINGS_PATH        = CACHE_DIR / "hvm_siglip_embeddings.pt"    # (450, 1152)
HVM_CLIP_EMBEDS_PATH              = CACHE_DIR / "hvm_clip_embeds.pt"           # (450, 768)
HVM_CLIP_DETAILED_EMBEDS_PATH     = CACHE_DIR / "hvm_clip_detailed_embeds.pt"  # (450, 768)
HVM_T5_EMBEDS_PATH                = CACHE_DIR / "hvm_t5_embeds.pt"             # (450, 512, 4096)
HVM_T5_DETAILED_EMBEDS_PATH       = CACHE_DIR / "hvm_t5_detailed_embeds.pt"    # (450, 512, 4096)

SEED = 42

N_STIMULI = 300
N_TRAIN = 200
N_VAL = 50
N_TEST = 50

RUST_TIME_WINDOW = (0, 250)
HVM_TIME_WINDOW  = (0, 250)

HVM_N_STIMULI  = 450
HVM_CATEGORIES = ('apple', 'bear', 'car', 'chair', 'dog', 'elephant', 'head', 'plane', 'table', 'turtle')
HVM_N_CAT      = 10
HVM_N_VAR      = 45   # variations per category
HVM_N_VAL      = 90   # 9 per category × 10
HVM_N_TEST     = 90   # 9 per category × 10

HVM_SRC_DIR        = Path('/mnt/smb/locker/issa-locker/Data/West/Saved_Images/E8')
HVM_RAW_DIR        = STIMULI_ROOT / 'hvm'
HVM_STIM_DIR       = STIMULI_ROOT / 'hvm_cropped'
HVM_NOFIXATION_DIR = STIMULI_ROOT / 'hvm_nofixation'

RUST_NOFIXATION_DIR = STIMULI_ROOT / 'rust_nofixation'

SIGLIP_DIM = 1152  # google/siglip-so400m-patch14-384
CLIP_DIM = 768     # FLUX CLIP pooled embedding dim
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

T5_PCA_K          = 128
T5_PCA_BASIS_PATH = CACHE_DIR / f"t5_pca_basis_k{T5_PCA_K}.pt"   # (K, 4096)
T5_PCA_MEAN_PATH  = CACHE_DIR / f"t5_pca_mean_k{T5_PCA_K}.pt"    # (4096,)

# --- Model identifiers ---
SIGLIP_MODEL_ID       = "google/siglip-so400m-patch14-384"
T5_XXL_MODEL_ID       = "google/t5-v1_1-xxl"
INSTRUCTBLIP_MODEL_ID = "Salesforce/instructblip-vicuna-7b"
INSTANTX_REPO         = "InstantX/FLUX.1-dev-IP-Adapter"
INSTANTX_WEIGHTS      = "ip-adapter.bin"

# --- FLUX / IP-Adapter architecture (fixed by InstantX checkpoint) ---
FLUX_JOINT_DIM  = 4096   # cross_attention_dim
FLUX_HIDDEN_DIM = 3072   # num_attention_heads * attention_head_dim
NUM_IP_TOKENS   = 128    # from image_proj shape: 524288 = 128 * 4096

# --- Aperture geometry ---
APERTURE_CENTER_FRAC = 0.06    # fixation square side as fraction of image size
HVM_RADIUS_FRAC      = 0.4909  # HVM circle radius / image size (measured from 276×276 px)
HVM_CENTER_FRAC      = 0.0362  # HVM fixation square / image size

# --- T5 token cache paths ---
T5_TOKENS_SHORT_PATH        = CACHE_DIR / "t5_xxl_tokens_short.pt"
T5_TOKENS_DETAILED_PATH     = CACHE_DIR / "t5_xxl_tokens_detailed.pt"
HVM_T5_TOKENS_SHORT_PATH    = CACHE_DIR / "hvm_t5_xxl_tokens_short.pt"
HVM_T5_TOKENS_DETAILED_PATH = CACHE_DIR / "hvm_t5_xxl_tokens_detailed.pt"
