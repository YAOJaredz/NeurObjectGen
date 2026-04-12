#!/bin/bash
# Sweep Transformer encoder hyperparameters via SLURM job arrays.

#SBATCH --job-name=transformer_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=0-02:00:00
#SBATCH --partition=issa
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/transformer_sweep_%A_%a.log
#SBATCH --array=0-15%5        # 2 d_models x 2 n_layers x 2 dropouts x 2 lrs = 16

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
mkdir -p /home/yy3658/NeurObjectGen/jobs/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate objGen
echo "Activated conda environment: objGen"

python -V

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
D_MODELS=(64 128)
N_LAYERS=(1 2)
DROPOUTS=(0.1 0.3)
LRS=(1e-3 3e-4)

N_DM=${#D_MODELS[@]}         # 2
N_NL=${#N_LAYERS[@]}         # 2
N_DO=${#DROPOUTS[@]}         # 2
N_LR=${#LRS[@]}              # 2

# 2 x 2 x 2 x 2 = 16 total
# n_heads fixed at 4 (divides both 64 and 128 evenly)
N_HEADS=4
WD=1e-2

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
LR_IDX=$((idx % N_LR));   idx=$((idx / N_LR))
DO_IDX=$((idx % N_DO));   idx=$((idx / N_DO))
NL_IDX=$((idx % N_NL));   idx=$((idx / N_NL))
DM_IDX=$((idx % N_DM))

D_MODEL=${D_MODELS[$DM_IDX]}
N_LAYER=${N_LAYERS[$NL_IDX]}
DROPOUT=${DROPOUTS[$DO_IDX]}
LR=${LRS[$LR_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: d_model=${D_MODEL} n_layers=${N_LAYER} dropout=${DROPOUT} lr=${LR}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
python scripts/train_encoder.py \
    --model transformer \
    --d-model      "${D_MODEL}" \
    --n-heads      "${N_HEADS}" \
    --n-layers     "${N_LAYER}" \
    --dropout      "${DROPOUT}" \
    --lr           "${LR}" \
    --weight-decay "${WD}" \
    --epochs       150 \
    --batch-size   64 \
    --temperature  0.07
