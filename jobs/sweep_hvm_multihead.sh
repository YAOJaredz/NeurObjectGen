#!/bin/bash
# Sweep MultiHeadTransformer on HVM (SigLIP + CLIP-short), category-conditioned.
#
# Grid: nce_weight x d_model x n_layers x shared_dim = 3x2x2x2 = 24 runs
#   nce_weight: 0.1, 0.5, 0.8
#   d_model:    64, 128
#   n_layers:   2, 4
#   shared_dim: 256, 512
#
# Two heads: SigLIP (1152-d) + CLIP-short (768-d). Always category-conditioned
# (adds learned Embedding(10, shared_dim) at the shared latent).
# Checkpoint criterion: best val_mean_cos = (cos_siglip + cos_clip) / 2.
#
# Fixed: lr=3e-4, n_heads=4, weight_decay=0.01, dropout=0.1,
#        loss_weight_siglip=1.0, loss_weight_clip=1.0, uniformity_weight=0.1,
#        target_noise=0.02, input_noise=0.05, neuron_dropout=0.1,
#        nce_temperature=0.07, epochs=200, batch_size=128.

#SBATCH --job-name=hvm_multihead_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=0-04:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/hvm_multihead_sweep_%A_%a.log
#SBATCH --array=0-23%12        # 24 runs, max 12 concurrent

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
mkdir -p /home/yy3658/NeurObjectGen/jobs/logs

SIF=~/vscode.sif
PYTHON_BIN=~/miniconda3/envs/objGen/bin/python
PYTHON="apptainer exec -B /run:/run --mount type=bind,src=/mnt/smb/locker/issa-locker,dst=/mnt/smb/locker/issa-locker --mount type=bind,src=/share/issa,dst=/share/issa --nv $SIF $PYTHON_BIN"
echo "Using Apptainer: $SIF"

$PYTHON -V

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
NCE_WEIGHTS=(0.1 0.5 0.8)
D_MODELS=(64 128)
N_LAYERS=(2 4)
SHARED_DIMS=(256 512)
DROPOUT=0.1
NCE_TEMP=0.07

N_NW=${#NCE_WEIGHTS[@]}       # 3
N_DM=${#D_MODELS[@]}          # 2
N_NL=${#N_LAYERS[@]}          # 2
N_SD=${#SHARED_DIMS[@]}       # 2

# 3 x 2 x 2 x 2 = 24 total
# n_heads=4 divides both d_model=64 and d_model=128
N_HEADS=4

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
SD_IDX=$((idx % N_SD));  idx=$((idx / N_SD))
NL_IDX=$((idx % N_NL));  idx=$((idx / N_NL))
DM_IDX=$((idx % N_DM));  idx=$((idx / N_DM))
NW_IDX=$((idx % N_NW))

NCE_WEIGHT=${NCE_WEIGHTS[$NW_IDX]}
D_MODEL=${D_MODELS[$DM_IDX]}
N_LAYER=${N_LAYERS[$NL_IDX]}
SHARED_DIM=${SHARED_DIMS[$SD_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: nce_weight=${NCE_WEIGHT} d_model=${D_MODEL} n_layers=${N_LAYER} shared_dim=${SHARED_DIM}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON train/train_multihead.py \
    --dataset          hvm                   \
    --use-category                           \
    --d-model          "${D_MODEL}"          \
    --n-heads          "${N_HEADS}"          \
    --n-layers         "${N_LAYER}"          \
    --shared-dim       "${SHARED_DIM}"       \
    --dropout          "${DROPOUT}"          \
    --lr               3e-4                  \
    --weight-decay     1e-2                  \
    --loss-weight-siglip 1.0                 \
    --loss-weight-clip   1.0                 \
    --uniformity-weight  0.1                 \
    --target-noise     0.02                  \
    --nce-weight       "${NCE_WEIGHT}"       \
    --nce-temperature  "${NCE_TEMP}"         \
    --epochs           200                   \
    --batch-size       128
