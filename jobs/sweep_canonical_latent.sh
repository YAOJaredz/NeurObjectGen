#!/bin/bash
# Sweep TemporalTransformer on HVM — canonical object latent target (16×16×16 bbox crop).
#
# Grid: use_residual x d_model x n_layers x lr = 2x2x2x2 = 16 jobs
#   use_residual: 0 (full latent), 1 (predict residual from category mean)
#   d_model:      64, 128
#   n_layers:     1, 2
#   lr:           1e-3, 3e-4
#
# Fixed: canonical_lat=16, pca_dim=0, n_heads=4, dropout=0.1, weight_decay=1e-2,
#        batch_size=32, epochs=200, warmup_frac=0.1, use-category,
#        input_noise=0.05, neuron_dropout=0.1, nce_weight=0.1, nce_temperature=0.07.

#SBATCH --job-name=obj_latent_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/canonical_latent_sweep_%A_%a.log
#SBATCH --array=0-15%8   # 16 runs, max 8 concurrent

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
USE_RESIDUALS=(0 1)
D_MODELS=(64 128)
N_LAYERS=(1 2)
LRS=(1e-3 3e-4)
CANONICAL_LAT=16

N_UR=${#USE_RESIDUALS[@]}  # 2
N_DM=${#D_MODELS[@]}       # 2
N_NL=${#N_LAYERS[@]}       # 2
N_LR=${#LRS[@]}            # 2

N_HEADS=4

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
LR_IDX=$((idx % N_LR));  idx=$((idx / N_LR))
NL_IDX=$((idx % N_NL));  idx=$((idx / N_NL))
DM_IDX=$((idx % N_DM));  idx=$((idx / N_DM))
UR_IDX=$((idx % N_UR))

USE_RESIDUAL=${USE_RESIDUALS[$UR_IDX]}
D_MODEL=${D_MODELS[$DM_IDX]}
N_LAYER=${N_LAYERS[$NL_IDX]}
LR=${LRS[$LR_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: use_residual=${USE_RESIDUAL} d_model=${D_MODEL} n_layers=${N_LAYER} lr=${LR}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
RESIDUAL_FLAG=""
if [ "${USE_RESIDUAL}" = "1" ]; then
    RESIDUAL_FLAG="--use-residual"
fi

$PYTHON train/train_canonical_latent.py \
    --d-model          "${D_MODEL}"       \
    --n-heads          "${N_HEADS}"       \
    --n-layers         "${N_LAYER}"       \
    --canonical-lat    "${CANONICAL_LAT}" \
    --dropout          0.1                \
    --lr               "${LR}"            \
    --weight-decay     1e-2               \
    --epochs           200                \
    --batch-size       32                 \
    --warmup-frac      0.1                \
    --use-category                        \
    --input-noise      0.05               \
    --neuron-dropout   0.1                \
    --nce-weight       0.1                \
    --nce-temperature  0.07               \
    ${RESIDUAL_FLAG}
