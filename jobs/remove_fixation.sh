#!/bin/bash
# Remove fixation square from HVM or Rust cropped stimuli using FLUX img2img.
# Usage: sbatch jobs/remove_fixation.sh [hvm|rust]

#SBATCH --job-name=remove_fixation
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
#SBATCH --partition=issa
#SBATCH --nodelist=ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/remove_fixation_%j.log

mkdir -p /home/yy3658/NeurObjectGen/jobs/logs

DATASET=${1:-hvm}

SIF=~/vscode.sif
PYTHON_BIN=~/miniconda3/envs/objGen/bin/python
PYTHON="apptainer exec -B /run:/run --mount type=bind,src=/mnt/smb/locker/issa-locker,dst=/mnt/smb/locker/issa-locker --mount type=bind,src=/share/issa,dst=/share/issa --nv $SIF $PYTHON_BIN"
echo "Using Apptainer: $SIF"
echo "Dataset: $DATASET"

$PYTHON -V

$PYTHON -m data_utils.remove_fixation --dataset hvm
