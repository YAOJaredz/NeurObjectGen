#!/bin/bash
# Sweep MLP encoder hyperparameters via SLURM job arrays.

#SBATCH --job-name=mlp_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=0-02:00:00
#SBATCH --partition=issa
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/mlp_sweep_%A_%a.log
#SBATCH --array=0-23%5        # 3 bottlenecks x 2 dropouts x 2 lrs x 2 wds = 24

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
BOTTLENECKS=(64 128 256)
DROPOUTS=(0.1 0.3)
LRS=(1e-3 3e-4)
WEIGHT_DECAYS=(1e-2 1e-1)

N_BN=${#BOTTLENECKS[@]}      # 3
N_DO=${#DROPOUTS[@]}         # 2
N_LR=${#LRS[@]}              # 2
N_WD=${#WEIGHT_DECAYS[@]}    # 2

# 3 x 2 x 2 x 2 = 24 total
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
WD_IDX=$((idx % N_WD));   idx=$((idx / N_WD))
LR_IDX=$((idx % N_LR));   idx=$((idx / N_LR))
DO_IDX=$((idx % N_DO));   idx=$((idx / N_DO))
BN_IDX=$((idx % N_BN))

BOTTLENECK=${BOTTLENECKS[$BN_IDX]}
DROPOUT=${DROPOUTS[$DO_IDX]}
LR=${LRS[$LR_IDX]}
WD=${WEIGHT_DECAYS[$WD_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: bottleneck=${BOTTLENECK} dropout=${DROPOUT} lr=${LR} wd=${WD}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
python scripts/train_encoder.py \
    --model mlp \
    --bottleneck   "${BOTTLENECK}" \
    --dropout      "${DROPOUT}" \
    --lr           "${LR}" \
    --weight-decay "${WD}" \
    --epochs       150 \
    --batch-size   64 \
    --temperature  0.07
