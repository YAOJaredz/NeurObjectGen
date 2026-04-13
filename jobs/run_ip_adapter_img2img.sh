#!/bin/bash
#SBATCH --job-name=ip_img2img
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --time=0-10:00:00
#SBATCH --partition=issa
#SBATCH --nodelist=ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/ip_img2img_%j.log

mkdir -p /home/yy3658/NeurObjectGen/jobs/logs

# Fail fast if GPU is not visible inside the container
nvidia-smi || { echo "ERROR: no GPU visible, aborting"; exit 1; }

SIF=~/vscode.sif
PYTHON_BIN=~/miniconda3/envs/objGen/bin/python
PYTHON="apptainer exec -B /run:/run --mount type=bind,src=/mnt/smb/locker/issa-locker,dst=/mnt/smb/locker/issa-locker --mount type=bind,src=/share/issa,dst=/share/issa --nv $SIF $PYTHON_BIN"

cd /home/yy3658/NeurObjectGen
export PYTHONPATH="/home/yy3658/NeurObjectGen${PYTHONPATH:+:${PYTHONPATH}}"

$PYTHON scripts/train_ip_adapter_img2img.py \
    --embedding-source siglip  \
    --patch-grid       14      \
    --lr               1e-3    \
    --epochs           50      \
    --batch-size       32      \
    --micro-batch      2       \
    --weight-decay     1e-4    \
    --guidance-scale   3.5
