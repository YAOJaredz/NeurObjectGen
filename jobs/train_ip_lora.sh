#!/bin/bash
# Fine-tune the InstantX IP-Adapter to the HVM stimulus distribution via LoRA
# (proposal §6.5). Trains rank-r LoRA adapters on to_k_ip / to_v_ip across all
# 57 FLUX attention blocks using the standard flow-matching denoising objective
# on the 450 HVM stimuli, with stochastic IP dropout for CFG-style training.
#
# Multi-GPU on a single node: training is wrapped with HF Accelerate (DDP).
# Each GPU loads its own frozen FLUX copy, but only the ~6.5M LoRA grads are
# all-reduced per step, so per-GPU memory stays roughly the same as single-GPU
# while the effective batch size scales linearly with NGPUS.
#
# Override hyperparameters from the sbatch CLI, e.g.:
#   NGPUS=2 sbatch jobs/train_ip_lora.sh
#   NGPUS=4 RANK=16 STEPS=4000 sbatch jobs/train_ip_lora.sh

#SBATCH --job-name=ip_lora
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=0-12:00:00
#SBATCH --partition=issa
#SBATCH --nodelist=ax11
#SBATCH --gres=gpu:2
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/ip_lora_%j.log

NGPUS=2

mkdir -p /home/yy3658/NeurObjectGen/jobs/logs
mkdir -p /home/yy3658/NeurObjectGen/checkpoints/ip_lora

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
SIF=~/vscode.sif
PYTHON_BIN=~/miniconda3/envs/objGen/bin/python
ACCELERATE_BIN=~/miniconda3/envs/objGen/bin/accelerate
PYTHON="apptainer exec -B /run:/run \
    --mount type=bind,src=/mnt/smb/locker/issa-locker,dst=/mnt/smb/locker/issa-locker \
    --mount type=bind,src=/share/issa,dst=/share/issa \
    --nv $SIF $PYTHON_BIN"
ACCELERATE="apptainer exec -B /run:/run \
    --mount type=bind,src=/mnt/smb/locker/issa-locker,dst=/mnt/smb/locker/issa-locker \
    --mount type=bind,src=/share/issa,dst=/share/issa \
    --nv $SIF $ACCELERATE_BIN"
echo "Using Apptainer: $SIF"
$PYTHON -V

# Avoid CUDA fragmentation OOMs at the boundary of the bf16 forward pass.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Hyperparameters (override via env vars on the sbatch CLI)
# ---------------------------------------------------------------------------
STEPS=${STEPS:-2000}
RANK=${RANK:-8}
ALPHA=${ALPHA:-16}
LR=${LR:-1e-4}
P_DROP=${P_DROP:-0.5}
BATCH_SIZE=${BATCH_SIZE:-1}     # per-GPU micro-batch
GRAD_ACCUM=${GRAD_ACCUM:-4}
WARMUP=${WARMUP:-100}
SAVE_EVERY=${SAVE_EVERY:-500}
SEED=${SEED:-42}
SAVE_PATH=${SAVE_PATH:-checkpoints/ip_lora/hvm_r${RANK}.safetensors}

EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM * NGPUS))
echo "[ip_lora] ngpus=${NGPUS} per-gpu-batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} effective=${EFFECTIVE_BATCH}"
echo "[ip_lora] steps=${STEPS} rank=${RANK} alpha=${ALPHA} lr=${LR} p_drop=${P_DROP}"
echo "[ip_lora] save_path=${SAVE_PATH}"

# ---------------------------------------------------------------------------
# Train (single GPU → plain python; multi-GPU → accelerate launch)
# ---------------------------------------------------------------------------
TRAIN_ARGS=(
    --steps                 "${STEPS}"
    --rank                  "${RANK}"
    --alpha                 "${ALPHA}"
    --lr                    "${LR}"
    --p-drop                "${P_DROP}"
    --batch-size            "${BATCH_SIZE}"
    --grad-accum            "${GRAD_ACCUM}"
    --warmup-steps          "${WARMUP}"
    --save-every            "${SAVE_EVERY}"
    --seed                  "${SEED}"
    --save-path             "${SAVE_PATH}"
    --gradient-checkpointing
)

if [[ "${NGPUS}" -gt 1 ]]; then
    $ACCELERATE launch \
        --num_processes "${NGPUS}" \
        --num_machines 1 \
        --mixed_precision no \
        --dynamo_backend no \
        -m train.train_ip_lora "${TRAIN_ARGS[@]}"
else
    $PYTHON -m train.train_ip_lora "${TRAIN_ARGS[@]}"
fi
