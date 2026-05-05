#!/bin/bash
# Train both multihead models (full-image global and obj-crop) without category
# conditioning, using the same hyperparameters as the best category-conditioned
# runs (best_hvm_multihead_config.json and best_hvm_multihead_obj_config.json).
#
# Runs the two training jobs sequentially on a single GPU.
#
# Usage:
#   sbatch jobs/train_multihead_obj_nocat.sh

#SBATCH --job-name=mh_nocat
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=0-08:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --gres=gpu:1
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/mh_nocat_%j.log

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
# Global (full-image SigLIP + CLIP) — best: d64 nh4 nl2 sd512 bs128 nw0.1
# Checkpoints -> checkpoints/multihead/hvm/
# ---------------------------------------------------------------------------
echo ""
echo "=== Training global multihead (no category) ==="
$PYTHON -m train.train_multihead \
    --dataset             hvm \
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

# ---------------------------------------------------------------------------
# Obj-crop (obj-SigLIP + CLIP) — best: d64 nh4 nl2 sd512 bs64 ws5.0 nw0.1
# Checkpoints -> checkpoints/multihead/hvm_obj/
# ---------------------------------------------------------------------------
echo ""
echo "=== Training obj multihead (no category) ==="
$PYTHON -m train.train_multihead_obj \
    --d-model             64 \
    --n-heads             4 \
    --n-layers            2 \
    --shared-dim          512 \
    --dropout             0.1 \
    --epochs              200 \
    --batch-size          64 \
    --lr                  3e-4 \
    --weight-decay        0.01 \
    --warmup-frac         0.1 \
    --loss-weight-siglip  5.0 \
    --loss-weight-clip    1.0 \
    --uniformity-weight   0.1 \
    --target-noise        0.02 \
    --input-noise         0.05 \
    --neuron-dropout      0.1 \
    --nce-weight          0.1 \
    --nce-temperature     0.07
