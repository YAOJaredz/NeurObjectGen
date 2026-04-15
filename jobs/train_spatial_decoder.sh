#!/bin/bash
# Train nonlinear spatial decoders (MLP + Transformer) via SLURM job array.
#
# Array layout (9 jobs):
#   0 — mlp,         K=64,  bottleneck=128
#   1 — mlp,         K=128, bottleneck=256
#   2 — mlp,         K=256, bottleneck=512
#   3 — mlp,         K=128, bottleneck=256, dropout=0.2
#   4 — mlp,         K=128, bottleneck=512
#   5 — transformer, K=64,  d_model=64,  n_heads=4, n_layers=1
#   6 — transformer, K=128, d_model=64,  n_heads=4, n_layers=2
#   7 — transformer, K=128, d_model=128, n_heads=4, n_layers=2
#   8 — transformer, K=256, d_model=128, n_heads=8, n_layers=2
#
# Submit:           sbatch jobs/train_spatial_decoder.sh
# Single (debug):   SLURM_ARRAY_TASK_ID=1 bash jobs/train_spatial_decoder.sh

#SBATCH --job-name=spatial_decoder
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=0-02:00:00
#SBATCH --partition=issa
#SBATCH --nodelist=ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/spatial_decoder_%A_%a.log
#SBATCH --array=0-8%3

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
mkdir -p /home/yy3658/NeurObjectGen/jobs/logs

SIF=~/vscode.sif
PYTHON_BIN=~/miniconda3/envs/objGen/bin/python
PYTHON="apptainer exec -B /run:/run --mount type=bind,src=/mnt/smb/locker/issa-locker,dst=/mnt/smb/locker/issa-locker --mount type=bind,src=/share/issa,dst=/share/issa --nv $SIF $PYTHON_BIN"
echo "Using Apptainer: $SIF"
$PYTHON -V

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case $TASK_ID in

  # --- MLP ---
  0)
    echo "Task 0: mlp K=64 bn=128"
    $PYTHON scripts/train_spatial_decoder.py \
        --model mlp --n-components 64 --bottleneck 128 \
        --dropout 0.1 --lr 1e-3 --alpha-l2 1e-3 --epochs 200 --batch-size 32
    ;;
  1)
    echo "Task 1: mlp K=128 bn=256"
    $PYTHON scripts/train_spatial_decoder.py \
        --model mlp --n-components 128 --bottleneck 256 \
        --dropout 0.1 --lr 1e-3 --alpha-l2 1e-3 --epochs 200 --batch-size 32
    ;;
  2)
    echo "Task 2: mlp K=256 bn=512"
    $PYTHON scripts/train_spatial_decoder.py \
        --model mlp --n-components 256 --bottleneck 512 \
        --dropout 0.1 --lr 1e-3 --alpha-l2 1e-3 --epochs 200 --batch-size 32
    ;;
  3)
    echo "Task 3: mlp K=128 bn=256 dropout=0.2"
    $PYTHON scripts/train_spatial_decoder.py \
        --model mlp --n-components 128 --bottleneck 256 \
        --dropout 0.2 --lr 1e-3 --alpha-l2 1e-3 --epochs 200 --batch-size 32
    ;;
  4)
    echo "Task 4: mlp K=128 bn=512"
    $PYTHON scripts/train_spatial_decoder.py \
        --model mlp --n-components 128 --bottleneck 512 \
        --dropout 0.1 --lr 1e-3 --alpha-l2 1e-3 --epochs 200 --batch-size 32
    ;;

  # --- Transformer ---
  5)
    echo "Task 5: transformer K=64 d=64 nh=4 nl=1"
    $PYTHON scripts/train_spatial_decoder.py \
        --model transformer --n-components 64 \
        --d-model 64 --n-heads 4 --n-layers 1 \
        --dropout 0.1 --lr 1e-3 --alpha-l2 1e-3 --epochs 200 --batch-size 32
    ;;
  6)
    echo "Task 6: transformer K=128 d=64 nh=4 nl=2"
    $PYTHON scripts/train_spatial_decoder.py \
        --model transformer --n-components 128 \
        --d-model 64 --n-heads 4 --n-layers 2 \
        --dropout 0.1 --lr 1e-3 --alpha-l2 1e-3 --epochs 200 --batch-size 32
    ;;
  7)
    echo "Task 7: transformer K=128 d=128 nh=4 nl=2"
    $PYTHON scripts/train_spatial_decoder.py \
        --model transformer --n-components 128 \
        --d-model 128 --n-heads 4 --n-layers 2 \
        --dropout 0.1 --lr 1e-3 --alpha-l2 1e-3 --epochs 200 --batch-size 32
    ;;
  8)
    echo "Task 8: transformer K=256 d=128 nh=8 nl=2"
    $PYTHON scripts/train_spatial_decoder.py \
        --model transformer --n-components 256 \
        --d-model 128 --n-heads 8 --n-layers 2 \
        --dropout 0.1 --lr 1e-3 --alpha-l2 1e-3 --epochs 200 --batch-size 32
    ;;

  *)
    echo "Unknown TASK_ID=$TASK_ID"; exit 1
    ;;
esac
