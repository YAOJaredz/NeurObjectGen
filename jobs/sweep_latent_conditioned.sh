#!/bin/bash
# Sweep ConditionedLatentModel on HVM.
# Neural encoder → embed, LatentRefiner([embed; cat_mean_proj]) → canonical latent.
#
# Grid: embed_dim x n_layers x refiner_hidden x lr = 2x2x2x2 = 16 jobs
#   embed_dim:      64, 128
#   n_layers:       1, 2
#   refiner_hidden: 256, 512
#   lr:             1e-3, 3e-4
#
# Fixed: canonical_lat=16, n_heads=4, dropout=0.1, weight_decay=1e-2,
#        batch_size=32, epochs=200, warmup_frac=0.1,
#        input_noise=0.05, neuron_dropout=0.1, nce_weight=0.1, nce_temperature=0.07.

#SBATCH --job-name=latent_cond_sweep
#SBATCH --account=yy3658
#SBATCH --chdir=/home/yy3658/NeurObjectGen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --partition=issa
#SBATCH --exclude=ax09,ax10,ax11
#SBATCH --output=/home/yy3658/NeurObjectGen/jobs/logs/latent_cond_sweep_%A_%a.log
#SBATCH --array=0-15%8   # 16 runs, max 8 concurrent

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
# Hyperparameter grid
# ---------------------------------------------------------------------------
EMBED_DIMS=(64 128)
N_LAYERS=(1 2)
REFINER_HIDDENS=(256 512)
LRS=(1e-3 3e-4)

N_ED=${#EMBED_DIMS[@]}       # 2
N_NL=${#N_LAYERS[@]}         # 2
N_RH=${#REFINER_HIDDENS[@]}  # 2
N_LR=${#LRS[@]}              # 2

N_HEADS=4

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

idx=$TASK_ID
LR_IDX=$((idx % N_LR));  idx=$((idx / N_LR))
RH_IDX=$((idx % N_RH));  idx=$((idx / N_RH))
NL_IDX=$((idx % N_NL));  idx=$((idx / N_NL))
ED_IDX=$((idx % N_ED))

EMBED_DIM=${EMBED_DIMS[$ED_IDX]}
N_LAYER=${N_LAYERS[$NL_IDX]}
REFINER_HIDDEN=${REFINER_HIDDENS[$RH_IDX]}
LR=${LRS[$LR_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: embed_dim=${EMBED_DIM} n_layers=${N_LAYER} refiner_hidden=${REFINER_HIDDEN} lr=${LR}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
$PYTHON train/train_latent_conditioned.py \
    --embed-dim        "${EMBED_DIM}"      \
    --n-heads          "${N_HEADS}"        \
    --n-layers         "${N_LAYER}"        \
    --refiner-hidden   "${REFINER_HIDDEN}" \
    --canonical-lat    16                  \
    --dropout          0.1                 \
    --lr               "${LR}"             \
    --weight-decay     1e-2                \
    --epochs           200                 \
    --batch-size       32                  \
    --warmup-frac      0.1                 \
    --input-noise      0.05                \
    --neuron-dropout   0.1                 \
    --nce-weight       0.1                 \
    --nce-temperature  0.07
