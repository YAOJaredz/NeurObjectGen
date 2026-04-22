#!/bin/bash
# Generate HVM condition comparisons swept over img2img strength values.
#
# Runs all 90 test stimuli across strengths: 0.35, 0.45, 0.55, 0.65, 0.75
# Output: outputs/generate_hvm_strength/s{strength}/stim{idx:04d}.png

#SBATCH --job-name=hvm_gen_strength
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00:00
#SBATCH --partition=issa
#SBATCH --nodelist=ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/hvm_gen_strength_%j.log

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
# Generate
# ---------------------------------------------------------------------------
$PYTHON scripts/generate_hvm_sweep_strength.py
