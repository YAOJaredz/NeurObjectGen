#!/bin/bash
# Sweep MultiHeadTransformer hyperparameters via SLURM job arrays.
#
# Grid: loss_weight_t5 x d_model x n_layers x shared_dim x t5_pca_k = 3x2x2x2x2 = 48 runs
#   loss_weight_t5: 0.0 (2-head ablation), 0.1, 0.5
#   d_model:        64, 128
#   n_layers:       2, 4
#   shared_dim:     256, 512
#   t5_pca_k:       128, 256
#
# T5 head uses MSE loss. nce_weight fixed at 0.8 (InfoNCE primary for SigLIP/CLIP heads).
# Checkpoint criterion: (sig_2afc + clip_2afc + val_cos_t5) / 3
#
# Fixed: lr=3e-4, n_heads=4, weight_decay=0.01, dropout=0.1,
#        loss_weight_siglip=1.0, loss_weight_clip=1.0, uniformity_weight=0.1,
#        target_noise=0.02, nce_weight=0.8, nce_temperature=0.07, epochs=200, batch_size=64

#SBATCH --job-name=multihead_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=0-04:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/multihead_sweep_%A_%a.log
#SBATCH --array=0-47%16        # 48 runs, max 16 concurrent

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
LOSS_WEIGHT_T5S=(0.0 0.1 0.5)
D_MODELS=(64 128)
N_LAYERS=(2 4)
SHARED_DIMS=(256 512)
T5_PCA_KS=(128 256)
DROPOUT=0.1
NCE_WEIGHT=0.8   # fixed
NCE_TEMP=0.07

N_WT=${#LOSS_WEIGHT_T5S[@]}   # 3
N_DM=${#D_MODELS[@]}          # 2
N_NL=${#N_LAYERS[@]}          # 2
N_SD=${#SHARED_DIMS[@]}       # 2
N_PK=${#T5_PCA_KS[@]}         # 2

# 3 x 2 x 2 x 2 x 2 = 48 total
# n_heads=4 divides both d_model=64 and d_model=128
N_HEADS=4

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
PK_IDX=$((idx % N_PK));  idx=$((idx / N_PK))
SD_IDX=$((idx % N_SD));  idx=$((idx / N_SD))
NL_IDX=$((idx % N_NL));  idx=$((idx / N_NL))
DM_IDX=$((idx % N_DM));  idx=$((idx / N_DM))
WT_IDX=$((idx % N_WT))

LOSS_WEIGHT_T5=${LOSS_WEIGHT_T5S[$WT_IDX]}
D_MODEL=${D_MODELS[$DM_IDX]}
N_LAYER=${N_LAYERS[$NL_IDX]}
SHARED_DIM=${SHARED_DIMS[$SD_IDX]}
T5_PCA_K=${T5_PCA_KS[$PK_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: loss_weight_t5=${LOSS_WEIGHT_T5} d_model=${D_MODEL} n_layers=${N_LAYER} shared_dim=${SHARED_DIM} t5_pca_k=${T5_PCA_K} nce_weight=${NCE_WEIGHT}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON scripts/train_multihead.py \
    --d-model          "${D_MODEL}"          \
    --n-heads          "${N_HEADS}"          \
    --n-layers         "${N_LAYER}"          \
    --shared-dim       "${SHARED_DIM}"       \
    --dropout          "${DROPOUT}"           \
    --lr               3e-4                  \
    --weight-decay     1e-2                  \
    --t5-pca-k         "${T5_PCA_K}"          \
    --loss-weight-t5   "${LOSS_WEIGHT_T5}"   \
    --loss-weight-siglip 1.0                 \
    --loss-weight-clip   1.0                 \
    --uniformity-weight  0.1                 \
    --target-noise     0.02                  \
    --nce-weight       "${NCE_WEIGHT}"       \
    --nce-temperature  "${NCE_TEMP}"         \
    --epochs           200                   \
    --batch-size       64
