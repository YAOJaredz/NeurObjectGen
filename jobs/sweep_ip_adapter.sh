#!/bin/bash
# Sweep IP-Adapter hyperparameters via SLURM job arrays.
# Grid: 2 lrs x 3 n_tokens x 2 hiddens = 12 runs

#SBATCH --job-name=ip_adapter_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=0-06:00:00
#SBATCH --partition=issa
#SBATCH --nodelist=ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/ip_adapter_sweep_%A_%a.log
#SBATCH --array=0-11%5   # 12 total, 5 concurrent

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
LRS=(1e-4 3e-4)
N_TOKENS=(1 4 16)
HIDDENS=(512 1024)

N_LR=${#LRS[@]}   # 2
N_TOK=${#N_TOKENS[@]}  # 3
N_HID=${#HIDDENS[@]}   # 2

# 2 x 3 x 2 = 12 total
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
HID_IDX=$((idx % N_HID));  idx=$((idx / N_HID))
TOK_IDX=$((idx % N_TOK));  idx=$((idx / N_TOK))
LR_IDX=$((idx % N_LR))

LR=${LRS[$LR_IDX]}
N_TOK_VAL=${N_TOKENS[$TOK_IDX]}
HIDDEN=${HIDDENS[$HID_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: lr=${LR} n_tokens=${N_TOK_VAL} hidden=${HIDDEN}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON scripts/train_ip_adapter.py \
    --lr              "${LR}"       \
    --n-tokens        "${N_TOK_VAL}" \
    --hidden          "${HIDDEN}"   \
    --guidance-scale  3.5           \
    --epochs          100           \
    --batch-size      32            \
    --micro-batch     4             \
    --weight-decay    1e-4
