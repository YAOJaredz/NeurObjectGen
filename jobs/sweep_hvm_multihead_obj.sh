#!/bin/bash
# Sweep MultiHeadTransformer on HVM with bbox-crop obj-SigLIP target.
# Grid: 3 d_models x 3 n_layers x 2 lrs = 18 jobs. weight_decay fixed at 1e-2.

#SBATCH --job-name=hvm_mh_obj
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/hvm_mh_obj_%A_%a.log
#SBATCH --array=0-17%8

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
mkdir -p /home/yy3658/NeurObjectGen/jobs/logs

SIF=~/vscode.sif
PYTHON_BIN=~/miniconda3/envs/objGen/bin/python
PYTHON="apptainer exec -B /run:/run --mount type=bind,src=/mnt/smb/locker/issa-locker,dst=/mnt/smb/locker/issa-locker --mount type=bind,src=/share/issa,dst=/share/issa --nv $SIF $PYTHON_BIN"

$PYTHON -V

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
D_MODELS=(64 128 256)
N_LAYERS=(1 2 4)
LRS=(1e-3 3e-4)

N_DM=${#D_MODELS[@]}
N_NL=${#N_LAYERS[@]}
N_LR=${#LRS[@]}

N_HEADS=4
WEIGHT_DECAY=1e-2

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
LR_IDX=$((idx % N_LR));    idx=$((idx / N_LR))
NL_IDX=$((idx % N_NL));    idx=$((idx / N_NL))
DM_IDX=$((idx % N_DM))

D_MODEL=${D_MODELS[$DM_IDX]}
N_LAYER=${N_LAYERS[$NL_IDX]}
LR=${LRS[$LR_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: d_model=${D_MODEL} n_layers=${N_LAYER} lr=${LR}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON train/train_multihead_obj.py \
    --use-category \
    --d-model       "${D_MODEL}" \
    --n-heads       "${N_HEADS}" \
    --n-layers      "${N_LAYER}" \
    --shared-dim    512 \
    --dropout       0.1 \
    --lr            "${LR}" \
    --weight-decay  "${WEIGHT_DECAY}" \
    --epochs        200 \
    --batch-size    64 \
    --warmup-frac   0.1 \
    --nce-weight    0.8 \
    --nce-temperature 0.07 \
    --target-noise  0.02 \
    --input-noise   0.05 \
    --neuron-dropout 0.1 \
    --uniformity-weight 0.1
