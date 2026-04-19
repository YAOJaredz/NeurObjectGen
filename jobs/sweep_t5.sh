#!/bin/bash
# Sweep standalone T5-PCA Transformer hyperparameters via SLURM job arrays.
#
# Grid: n_layers x d_model x nce_weight x uniformity_weight = 3x2x3x2 = 36 runs
#   n_layers:          1, 2, 3
#   d_model:           64, 128
#   nce_weight:        0.5, 0.8, 1.0
#   uniformity_weight: 0.0, 0.1
#
# Fixed: n_heads=4, t5_pca_k=128, dropout=0.1, nce_temperature=0.07,
#        target_noise=0.0, lr=3e-4, weight_decay=0.01, epochs=200, batch_size=64

#SBATCH --job-name=t5_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=0-04:00:00
#SBATCH --partition=issa
#SBATCH --nodelist=ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/t5_sweep_%A_%a.log
#SBATCH --array=0-35%6        # 36 runs, max 12 concurrent

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
N_LAYERS_VALS=(1 2 3)
D_MODELS=(128 256)
NCE_WEIGHTS=(0.5 0.8 1.0)
UNIF_WEIGHTS=(0.0 0.1)

N_NL=${#N_LAYERS_VALS[@]}   # 3
N_DM=${#D_MODELS[@]}        # 2
N_NW=${#NCE_WEIGHTS[@]}     # 3
N_UW=${#UNIF_WEIGHTS[@]}    # 2

# 3 x 2 x 3 x 2 = 36 total
# n_heads=4 divides both d_model=128 and d_model=256
N_HEADS=4

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
UW_IDX=$((idx % N_UW));  idx=$((idx / N_UW))
NW_IDX=$((idx % N_NW));  idx=$((idx / N_NW))
DM_IDX=$((idx % N_DM));  idx=$((idx / N_DM))
NL_IDX=$((idx % N_NL))

N_LAYER=${N_LAYERS_VALS[$NL_IDX]}
D_MODEL=${D_MODELS[$DM_IDX]}
NCE_WEIGHT=${NCE_WEIGHTS[$NW_IDX]}
UNIF_WEIGHT=${UNIF_WEIGHTS[$UW_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: n_layers=${N_LAYER} d_model=${D_MODEL} nce_weight=${NCE_WEIGHT} uniformity_weight=${UNIF_WEIGHT}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON scripts/train_t5.py \
    --d-model           "${D_MODEL}"    \
    --n-heads           "${N_HEADS}"   \
    --n-layers          "${N_LAYER}"   \
    --t5-pca-k          128            \
    --dropout           0.1            \
    --nce-weight        "${NCE_WEIGHT}" \
    --nce-temperature   0.07           \
    --uniformity-weight "${UNIF_WEIGHT}" \
    --target-noise      0.0            \
    --epochs            200            \
    --batch-size        2048             \
    --lr                3e-4           \
    --weight-decay      1e-2           \
    --per-token                        \
    --captions          both
