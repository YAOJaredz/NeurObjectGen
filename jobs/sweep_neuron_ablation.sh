#!/bin/bash
# Neuron-subset ablation sweep (no category conditioning).
#
# Retrains both the global (train_multihead) and obj-crop (train_multihead_obj)
# models across a range of neuron subset sizes and random seeds to measure how
# SigLIP decoding performance degrades with fewer neurons.
#
# Grid:
#   sizes  : 100, 250, 500, 750, 1000, 1500, 2000, full (8 levels)
#   seeds  : 0, 1, 2, 3, 4 (5 seeds)
#   models : global, obj-crop (2 models)
#
# Total jobs: 8 * 5 * 2 = 80
# Task IDs  0-39 → global model (train_multihead --dataset hvm)
# Task IDs 40-79 → obj model    (train_multihead_obj)
#
# Usage:
#   sbatch jobs/sweep_neuron_ablation.sh

#SBATCH --job-name=neur_ablation
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=0-08:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --gres=gpu:1
#SBATCH --array=0-79%16
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/neur_ablation_%A_%a.log

mkdir -p /home/yy3658/NeurObjectGen/jobs/logs

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
SIF=~/vscode.sif
PYTHON_BIN=~/miniconda3/envs/objGen/bin/python
PYTHON="apptainer exec -B /run:/run \
    --mount type=bind,src=/mnt/smb/locker/issa-locker,dst=/mnt/smb/locker/issa-locker \
    --mount type=bind,src=/share/issa,dst=/share/issa \
    --nv $SIF $PYTHON_BIN"
echo "Using Apptainer: $SIF"
$PYTHON -V

# ---------------------------------------------------------------------------
# Decode task ID
# ---------------------------------------------------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Model: 0 = global, 1 = obj-crop
N_SIZES=8
N_SEEDS=5
N_PER_MODEL=$((N_SIZES * N_SEEDS))   # 40

MODEL_IDX=$((TASK_ID / N_PER_MODEL))  # 0 or 1
LOCAL_ID=$((TASK_ID % N_PER_MODEL))

SEED_IDX=$((LOCAL_ID % N_SEEDS))
SIZE_IDX=$((LOCAL_ID / N_SEEDS))

# Map SIZE_IDX to neuron count ("full" means no --n-neurons flag)
SIZES=(100 250 500 750 1000 1500 2000 "full")
SIZE="${SIZES[$SIZE_IDX]}"

NEURON_SEED=$SEED_IDX

echo "Task ${TASK_ID}: model_idx=${MODEL_IDX} size=${SIZE} seed=${NEURON_SEED}"

# ---------------------------------------------------------------------------
# Build optional --n-neurons flag
# ---------------------------------------------------------------------------
if [ "$SIZE" = "full" ]; then
    NEURON_ARG=""
else
    NEURON_ARG="--n-neurons ${SIZE} --neuron-seed ${NEURON_SEED}"
fi

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
if [ "$MODEL_IDX" -eq 0 ]; then
    echo "=== Global multihead (size=${SIZE}, seed=${NEURON_SEED}) ==="
    $PYTHON -m train.train_multihead \
        --dataset             hvm \
        --d-model             64 \
        --n-heads             4 \
        --n-layers            2 \
        --shared-dim          512 \
        --dropout             0.1 \
        --epochs              200 \
        --batch-size          128 \
        --lr                  3e-4 \
        --weight-decay        0.01 \
        --warmup-frac         0.1 \
        --loss-weight-siglip  1.0 \
        --loss-weight-clip    1.0 \
        --uniformity-weight   0.1 \
        --target-noise        0.02 \
        --input-noise         0.05 \
        --neuron-dropout      0.1 \
        --nce-weight          0.1 \
        --nce-temperature     0.07 \
        $NEURON_ARG
else
    echo "=== Obj-crop multihead (size=${SIZE}, seed=${NEURON_SEED}) ==="
    $PYTHON -m train.train_multihead_obj \
        --d-model             64 \
        --n-heads             4 \
        --n-layers            2 \
        --shared-dim          512 \
        --dropout             0.1 \
        --epochs              200 \
        --batch-size          64 \
        --lr                  3e-4 \
        --weight-decay        0.01 \
        --warmup-frac         0.1 \
        --loss-weight-siglip  5.0 \
        --loss-weight-clip    1.0 \
        --uniformity-weight   0.1 \
        --target-noise        0.02 \
        --input-noise         0.05 \
        --neuron-dropout      0.1 \
        --nce-weight          0.1 \
        --nce-temperature     0.07 \
        $NEURON_ARG
fi
