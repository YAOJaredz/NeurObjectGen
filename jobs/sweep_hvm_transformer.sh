#!/bin/bash
# Sweep Transformer encoder on HVM — siglip target, with and without category conditioning.
# Grid: 2 d_models x 2 n_layers x 1 dropout x 2 lrs x 2 wds x 2 cat = 32 jobs.

#SBATCH --job-name=hvm_transformer_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=0-02:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/hvm_transformer_sweep_%A_%a.log
#SBATCH --array=0-31%8   # 2 d_models x 2 n_layers x 1 dropout x 2 lrs x 2 wds x 2 cat = 32

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
D_MODELS=(64 128)
N_LAYERS=(1 2)
DROPOUTS=(0.1)
LRS=(1e-3 3e-4)
WEIGHT_DECAYS=(1e-2 1e-1)
USE_CATS=(0 1)   # 0 = unconditioned, 1 = category-conditioned

N_DM=${#D_MODELS[@]}         # 2
N_NL=${#N_LAYERS[@]}         # 2
N_DO=${#DROPOUTS[@]}         # 1
N_LR=${#LRS[@]}              # 2
N_WD=${#WEIGHT_DECAYS[@]}    # 2
N_CAT=${#USE_CATS[@]}        # 2

# n_heads=4 divides both 64 and 128 evenly
N_HEADS=4

# 2 x 2 x 1 x 2 x 2 x 2 = 32 total
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
CAT_IDX=$((idx % N_CAT));  idx=$((idx / N_CAT))
WD_IDX=$((idx % N_WD));    idx=$((idx / N_WD))
LR_IDX=$((idx % N_LR));    idx=$((idx / N_LR))
DO_IDX=$((idx % N_DO));    idx=$((idx / N_DO))
NL_IDX=$((idx % N_NL));    idx=$((idx / N_NL))
DM_IDX=$((idx % N_DM))

D_MODEL=${D_MODELS[$DM_IDX]}
N_LAYER=${N_LAYERS[$NL_IDX]}
DROPOUT=${DROPOUTS[$DO_IDX]}
LR=${LRS[$LR_IDX]}
WD=${WEIGHT_DECAYS[$WD_IDX]}
USE_CAT=${USE_CATS[$CAT_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: d_model=${D_MODEL} n_layers=${N_LAYER} dropout=${DROPOUT} lr=${LR} wd=${WD} use_cat=${USE_CAT}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
CAT_FLAG=""
[ "$USE_CAT" -eq 1 ] && CAT_FLAG="--use-category"

$PYTHON train/train_encoder.py \
    --dataset       hvm \
    --model         transformer \
    --target        siglip \
    --loss          cosine \
    --d-model       "${D_MODEL}" \
    --n-heads       "${N_HEADS}" \
    --n-layers      "${N_LAYER}" \
    --dropout       "${DROPOUT}" \
    --lr            "${LR}" \
    --weight-decay  "${WD}" \
    --epochs        200 \
    --batch-size    64 \
    --target-noise  0.02 \
    --uniformity-weight 0.1 \
    --warmup-frac   0.1 \
    $CAT_FLAG
