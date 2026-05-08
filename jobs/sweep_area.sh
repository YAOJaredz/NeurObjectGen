#!/bin/bash
# Per-area encoder sweep.
#
# Trains the global MultiHeadTransformer on neurons from individual brain
# areas and compound regions:
#   Individual: TE0, TE2, TE3, PRH, ENT, PHC
#   Compound:   IT (= TE0+TE2+TE3), MT (= PHC+PRH)
#
# The area string is passed directly to get_hvm_responses(area=...).
#
# Checkpoints land at:
#   checkpoints/multihead/hvm_<AREA>/d64_nh4_nl2_sd512_.../best.pt
#
# Total jobs: 8
# Task IDs 0-7 map to AREAS array below.
#
# Usage:
#   sbatch jobs/sweep_area.sh

#SBATCH --job-name=area_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=0-08:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/area_sweep_%A_%a.log

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
# Decode task ID → brain area
# ---------------------------------------------------------------------------
AREAS=(TE0 TE2 TE3 PRH ENT PHC IT MT)
AREA="${AREAS[${SLURM_ARRAY_TASK_ID}]}"
echo "Task ${SLURM_ARRAY_TASK_ID}: area=${AREA}"

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
$PYTHON -m train.train_multihead \
    --dataset             hvm \
    --area                ${AREA} \
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
    --nce-temperature     0.07
