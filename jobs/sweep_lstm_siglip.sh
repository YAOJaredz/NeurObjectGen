#!/bin/bash
# Sweep LSTM encoder — siglip target.
# Grid: 3 hiddens x 2 n_layers x 2 lrs x 2 dropouts x 2 wds = 48 jobs.

#SBATCH --job-name=lstm_siglip_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=0-02:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/lstm_siglip_sweep_%A_%a.log
#SBATCH --array=0-47%8   # 3 hiddens x 2 n_layers x 2 lrs x 2 dropouts x 2 wds = 48

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
HIDDENS=(64 128 256)
N_LAYERS=(1 2)
LRS=(1e-3 3e-4)
DROPOUTS=(0.1 0.3)
WEIGHT_DECAYS=(1e-2 1e-1)

N_H=${#HIDDENS[@]}           # 3
N_NL=${#N_LAYERS[@]}         # 2
N_LR=${#LRS[@]}              # 2
N_DO=${#DROPOUTS[@]}         # 2
N_WD=${#WEIGHT_DECAYS[@]}    # 2

# 3 x 2 x 2 x 2 x 2 = 48 total
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
WD_IDX=$((idx % N_WD));   idx=$((idx / N_WD))
DO_IDX=$((idx % N_DO));   idx=$((idx / N_DO))
LR_IDX=$((idx % N_LR));   idx=$((idx / N_LR))
NL_IDX=$((idx % N_NL));   idx=$((idx / N_NL))
H_IDX=$((idx % N_H))

HIDDEN=${HIDDENS[$H_IDX]}
N_LAYER=${N_LAYERS[$NL_IDX]}
LR=${LRS[$LR_IDX]}
DROPOUT=${DROPOUTS[$DO_IDX]}
WD=${WEIGHT_DECAYS[$WD_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: hidden=${HIDDEN} n_layers=${N_LAYER} lr=${LR} dropout=${DROPOUT} wd=${WD}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON scripts/train_encoder.py \
    --model lstm \
    --target       siglip \
    --hidden       "${HIDDEN}" \
    --n-layers     "${N_LAYER}" \
    --dropout      "${DROPOUT}" \
    --lr           "${LR}" \
    --weight-decay "${WD}" \
    --epochs       150 \
    --batch-size   64 \
    --temperature  0.07
