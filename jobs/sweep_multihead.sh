#!/bin/bash
# Sweep MultiHeadTransformer hyperparameters via SLURM job arrays.
#
# Grid: loss_weight_t5 x d_model x n_layers x shared_dim x dropout = 3x2x2x2x2 = 48 runs
#   loss_weight_t5: 0.0 (2-head ablation), 0.1, 0.5
#   d_model:        64, 128
#   n_layers:       1, 2
#   shared_dim:     256, 512
#   dropout:        0.1, 0.3
#
# Fixed: lr=3e-4, n_heads=4, weight_decay=0.01,
#        loss_weight_siglip=1.0, loss_weight_clip=1.0, uniformity_weight=0.1,
#        target_noise=0.02, epochs=200, batch_size=64

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
#SBATCH --array=0-71%6        # 72 runs, max 8 concurrent

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
N_LAYERS=(1 2 4)
SHARED_DIMS=(256 512)
DROPOUTS=(0.1 0.3)

N_WT=${#LOSS_WEIGHT_T5S[@]}   # 3
N_DM=${#D_MODELS[@]}          # 2
N_NL=${#N_LAYERS[@]}          # 3
N_SD=${#SHARED_DIMS[@]}       # 2
N_DO=${#DROPOUTS[@]}          # 2

# 3 x 2 x 3 x 2 x 2 = 72 total
# n_heads=4 divides both d_model=64 and d_model=128
N_HEADS=4

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
DO_IDX=$((idx % N_DO));  idx=$((idx / N_DO))
SD_IDX=$((idx % N_SD));  idx=$((idx / N_SD))
NL_IDX=$((idx % N_NL));  idx=$((idx / N_NL))
DM_IDX=$((idx % N_DM));  idx=$((idx / N_DM))
WT_IDX=$((idx % N_WT))

LOSS_WEIGHT_T5=${LOSS_WEIGHT_T5S[$WT_IDX]}
D_MODEL=${D_MODELS[$DM_IDX]}
N_LAYER=${N_LAYERS[$NL_IDX]}
SHARED_DIM=${SHARED_DIMS[$SD_IDX]}
DROPOUT=${DROPOUTS[$DO_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: loss_weight_t5=${LOSS_WEIGHT_T5} d_model=${D_MODEL} n_layers=${N_LAYER} shared_dim=${SHARED_DIM} dropout=${DROPOUT}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON scripts/train_multihead.py \
    --d-model          "${D_MODEL}"          \
    --n-heads          "${N_HEADS}"          \
    --n-layers         "${N_LAYER}"          \
    --shared-dim       "${SHARED_DIM}"       \
    --dropout          "${DROPOUT}"          \
    --lr               3e-4                  \
    --weight-decay     1e-2                  \
    --loss-weight-t5   "${LOSS_WEIGHT_T5}"   \
    --loss-weight-siglip 1.0                 \
    --loss-weight-clip   1.0                 \
    --uniformity-weight  0.1                 \
    --target-noise     0.02                  \
    --epochs           200                   \
    --batch-size       64
