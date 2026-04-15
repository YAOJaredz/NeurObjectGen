#!/bin/bash
# Sweep Transformer encoder over both embedding targets (siglip, clip).
# Grid: 2 targets x 2 d_models x 3 n_layers x 2 lrs = 24 jobs.
# dropout=0.1 and weight_decay=1e-2 fixed (best from prior sweep).

#SBATCH --job-name=transformer_target_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=0-02:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/transformer_target_sweep_%A_%a.log
#SBATCH --array=0-23%8   # 2 targets x 2 d_models x 3 n_layers x 2 lrs = 24

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
D_MODELS=(64 128 512)
N_LAYERS=(1 2)
LRS=(1e-3 3e-4)

N_TG=${#TARGETS[@]}          # 2
N_DM=${#D_MODELS[@]}         # 2
N_NL=${#N_LAYERS[@]}         # 3
N_LR=${#LRS[@]}              # 2

# fixed
# n_heads=4 divides both 64 and 128 evenly
N_HEADS=4
DROPOUT=0.3
WD=1e-1

# 2 x 2 x 3 x 2 = 24 total
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
LR_IDX=$((idx % N_LR));   idx=$((idx / N_LR))
NL_IDX=$((idx % N_NL));   idx=$((idx / N_NL))
DM_IDX=$((idx % N_DM));   idx=$((idx / N_DM))
TG_IDX=$((idx % N_TG))

TARGET=${TARGETS[$TG_IDX]}
D_MODEL=${D_MODELS[$DM_IDX]}
N_LAYER=${N_LAYERS[$NL_IDX]}
LR=${LRS[$LR_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: target=${TARGET} d_model=${D_MODEL} n_layers=${N_LAYER} lr=${LR}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON scripts/train_encoder.py \
    --model transformer \
    --target       "${TARGET}" \
    --d-model      "${D_MODEL}" \
    --n-heads      "${N_HEADS}" \
    --n-layers     "${N_LAYER}" \
    --dropout      "${DROPOUT}" \
    --lr           "${LR}" \
    --weight-decay "${WD}" \
    --epochs       150 \
    --batch-size   64 \
    --temperature 0.07
