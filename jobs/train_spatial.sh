#!/bin/bash
# Train spatial ridge baselines via SLURM job array.
#
# Full grid sweep (24 jobs):
#   image_size  in {16, 32}
#   n_components in {32, 64}
#   n_neural_pcs in {50, 100}
#   alpha        in {1e4, 1e6, 1e8}
#
# Submit:           sbatch jobs/train_spatial.sh
# Single (debug):   SLURM_ARRAY_TASK_ID=0 bash jobs/train_spatial.sh

#SBATCH --job-name=spatial_ridge
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --time=0-02:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax11,ax10,ax09
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/spatial_ridge_%A_%a.log
#SBATCH --array=0-23

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
mkdir -p /home/yy3658/NeurObjectGen/jobs/logs

SIF=~/vscode.sif
PYTHON_BIN=~/miniconda3/envs/objGen/bin/python
PYTHON="apptainer exec -B /run:/run --mount type=bind,src=/mnt/smb/locker/issa-locker,dst=/mnt/smb/locker/issa-locker --mount type=bind,src=/share/issa,dst=/share/issa $SIF $PYTHON_BIN"
echo "Using Apptainer: $SIF"
$PYTHON -V

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# ---------------------------------------------------------------------------
# Grid: image_size x n_components x n_neural_pcs x alpha
# ---------------------------------------------------------------------------
IMAGE_SIZES=(16 32)
N_COMPONENTS=(32 64)
N_NEURAL_PCS=(50 100)
ALPHAS=(1e4 1e6 1e8)

# Unpack task ID into indices
alpha_idx=$((TASK_ID % 3))
pcs_idx=$(( (TASK_ID / 3) % 2 ))
comp_idx=$(( (TASK_ID / 6) % 2 ))
size_idx=$(( (TASK_ID / 12) % 2 ))

IMG=${IMAGE_SIZES[$size_idx]}
K=${N_COMPONENTS[$comp_idx]}
PCS=${N_NEURAL_PCS[$pcs_idx]}
ALPHA=${ALPHAS[$alpha_idx]}

echo "Task $TASK_ID: image_size=$IMG n_components=$K n_neural_pcs=$PCS alpha=$ALPHA"

$PYTHON scripts/train_spatial_ridge.py \
    --target pixel \
    --image-size  $IMG \
    --n-components $K \
    --n-neural-pcs $PCS \
    --alpha $ALPHA
