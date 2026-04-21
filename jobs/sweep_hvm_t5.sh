#!/bin/bash
# Sweep T5-PCA Transformer on HVM dataset with short captions.
#
# Grid: n_layers x d_model x nce_weight x t5_pca_k = 2x2x3x3 = 36 runs
#   n_layers:   2, 4
#   d_model:    128, 256
#   nce_weight: 0.1, 0.5, 0.8
#   t5_pca_k:   64, 128, 256
#
# Fixed: n_heads=4, uniformity_weight=0.1, dropout=0.1, nce_temperature=0.07,
#        target_noise=0.0, lr=3e-4, weight_decay=0.01,
#        epochs=200, batch_size=256, dataset=hvm, captions=short, use_category=true
#
# Prerequisite: run scripts/cache_hvm_t5_xxl_tokens.py --captions short

#SBATCH --job-name=hvm_t5_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=0-06:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/hvm_t5_sweep_%A_%a.log
#SBATCH --array=0-35%12       # 36 runs, max 12 concurrent

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
N_LAYERS_VALS=(2 4)
D_MODELS=(128 256)
NCE_WEIGHTS=(0.1 0.5 0.8)
T5_PCA_KS=(64 128 256)

N_NL=${#N_LAYERS_VALS[@]}   # 2
N_DM=${#D_MODELS[@]}        # 2
N_NW=${#NCE_WEIGHTS[@]}     # 3
N_PK=${#T5_PCA_KS[@]}       # 3

# 2 x 2 x 3 x 3 = 36 total
N_HEADS=4

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
PK_IDX=$((idx % N_PK));  idx=$((idx / N_PK))
NW_IDX=$((idx % N_NW));  idx=$((idx / N_NW))
DM_IDX=$((idx % N_DM));  idx=$((idx / N_DM))
NL_IDX=$((idx % N_NL))

N_LAYER=${N_LAYERS_VALS[$NL_IDX]}
D_MODEL=${D_MODELS[$DM_IDX]}
NCE_WEIGHT=${NCE_WEIGHTS[$NW_IDX]}
T5_PCA_K=${T5_PCA_KS[$PK_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: n_layers=${N_LAYER} d_model=${D_MODEL} nce_weight=${NCE_WEIGHT} t5_pca_k=${T5_PCA_K}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON train/train_t5.py \
    --dataset           hvm              \
    --use-category                       \
    --d-model           ${D_MODEL}       \
    --n-heads           ${N_HEADS}       \
    --n-layers          ${N_LAYER}       \
    --t5-pca-k          ${T5_PCA_K}      \
    --dropout           0.1              \
    --nce-weight        ${NCE_WEIGHT}    \
    --nce-temperature   0.07             \
    --uniformity-weight 0.1              \
    --target-noise      0.0              \
    --input-noise       0.05             \
    --neuron-dropout    0.1              \
    --epochs            200              \
    --batch-size        256              \
    --lr                3e-4             \
    --weight-decay      1e-2             \
    --captions          short
