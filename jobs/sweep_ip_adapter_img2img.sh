#!/bin/bash
# Sweep img2img IP-Adapter hyperparameters via SLURM job arrays.
# Grid: 3 n_tokens x 3 hiddens = 9 runs
# n_tokens reduced to 4/8/16 to limit disruption to FLUX's T5 attention.
# Embedding source: siglip (switch to neural after initial validation)

#SBATCH --job-name=ip_adapter_img2img
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=48gb
#SBATCH --time=0-10:00:00
#SBATCH --partition=issa
#SBATCH --nodelist=ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/ip_adapter_img2img_%A_%a.log
#SBATCH --array=0-8%3   # 9 total, 4 concurrent

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
mkdir -p /home/yy3658/NeurObjectGen/jobs/logs

SIF=~/vscode.sif
PYTHON_BIN=~/miniconda3/envs/objGen/bin/python
PYTHON="apptainer exec -B /run:/run --mount type=bind,src=/mnt/smb/locker/issa-locker,dst=/mnt/smb/locker/issa-locker --mount type=bind,src=/share/issa,dst=/share/issa --nv $SIF $PYTHON_BIN"
echo "Using Apptainer: $SIF"

$PYTHON -V

cd /home/yy3658/NeurObjectGen
export PYTHONPATH="/home/yy3658/NeurObjectGen${PYTHONPATH:+:${PYTHONPATH}}"

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
N_TOKENS=(4 8 16)
HIDDENS=(128 512 1024)

N_TOK=${#N_TOKENS[@]}   # 3
N_HID=${#HIDDENS[@]}    # 3

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
HID_IDX=$((idx % N_HID));  idx=$((idx / N_HID))
TOK_IDX=$((idx % N_TOK))

N_TOK_VAL=${N_TOKENS[$TOK_IDX]}
HIDDEN=${HIDDENS[$HID_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: n_tokens=${N_TOK_VAL} hidden=${HIDDEN}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON scripts/train_ip_adapter_img2img.py \
    --n-tokens         "${N_TOK_VAL}"  \
    --hidden           "${HIDDEN}"     \
    --lr               1e-3            \
    --guidance-scale   3.5             \
    --epochs           50              \
    --batch-size       32              \
    --micro-batch      4               \
    --weight-decay     1e-4            \
    --embedding-source neural
