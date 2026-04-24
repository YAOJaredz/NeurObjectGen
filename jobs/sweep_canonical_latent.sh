#!/bin/bash
# Sweep TemporalTransformer on HVM — canonical object latent target (16xSxS bbox crop).
#
# Grid: d_model x n_layers x lr = 3x3x2 = 18 jobs
#   d_model:  64, 128, 256
#   n_layers: 1, 2, 4
#   lr:       1e-3, 3e-4
#
# Fixed: canonical_lat=16, n_heads=4, dropout=0.1, weight_decay=1e-2,
#        batch_size=32, epochs=200, warmup_frac=0.1, use-category, image_size=512.

#SBATCH --job-name=canonical_latent_sweep
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
#SBATCH --array=0-17%6   # 18 runs, max 6 concurrent

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
D_MODELS=(64 128 256)
N_LAYERS=(1 2 4)
LRS=(1e-3 3e-4)
CANONICAL_LAT=16

N_DM=${#D_MODELS[@]}   # 3
N_NL=${#N_LAYERS[@]}   # 3
N_LR=${#LRS[@]}        # 2

# n_heads=4 divides 64, 128, and 256 evenly
N_HEADS=4

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
LR_IDX=$((idx % N_LR));  idx=$((idx / N_LR))
NL_IDX=$((idx % N_NL));  idx=$((idx / N_NL))
DM_IDX=$((idx % N_DM))

D_MODEL=${D_MODELS[$DM_IDX]}
N_LAYER=${N_LAYERS[$NL_IDX]}
LR=${LRS[$LR_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: d_model=${D_MODEL} n_layers=${N_LAYER} lr=${LR} canonical_lat=${CANONICAL_LAT}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON train/train_canonical_latent.py \
    --d-model       "${D_MODEL}"       \
    --n-heads       "${N_HEADS}"       \
    --n-layers      "${N_LAYER}"       \
    --canonical-lat "${CANONICAL_LAT}" \
    --dropout       0.1                \
    --lr            "${LR}"            \
    --weight-decay  1e-2               \
    --epochs        200                \
    --batch-size    32                 \
    --warmup-frac   0.1                \
    --use-category
