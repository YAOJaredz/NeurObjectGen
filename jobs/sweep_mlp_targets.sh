#!/bin/bash
# Sweep MLP encoder over both embedding targets (siglip, clip).
# Grid: 2 targets x 3 bottlenecks x 2 dropouts = 12 jobs.

#SBATCH --job-name=mlp_target_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=0-02:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/mlp_target_sweep_%A_%a.log
#SBATCH --array=0-11%6   # 2 targets x 3 bottlenecks x 2 dropouts = 12

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
TARGETS=(siglip clip)
BOTTLENECKS=(64 128 256)
DROPOUTS=(0.1 0.3)

# fixed
LR=3e-4
WD=1e-1

N_TG=${#TARGETS[@]}          # 2
N_BN=${#BOTTLENECKS[@]}      # 3
N_DO=${#DROPOUTS[@]}         # 2

# 2 x 3 x 2 = 12 total
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
DO_IDX=$((idx % N_DO));   idx=$((idx / N_DO))
BN_IDX=$((idx % N_BN));   idx=$((idx / N_BN))
TG_IDX=$((idx % N_TG))

TARGET=${TARGETS[$TG_IDX]}
BOTTLENECK=${BOTTLENECKS[$BN_IDX]}
DROPOUT=${DROPOUTS[$DO_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: target=${TARGET} bottleneck=${BOTTLENECK} dropout=${DROPOUT}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON scripts/train_encoder.py \
    --model mlp \
    --target       "${TARGET}" \
    --bottleneck   "${BOTTLENECK}" \
    --dropout      "${DROPOUT}" \
    --lr           "${LR}" \
    --weight-decay "${WD}" \
    --epochs       150 \
    --batch-size   64 \
    --temperature 0.07
