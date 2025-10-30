#!/bin/bash
#SBATCH --job-name=hnet-1stage-XL
#SBATCH --partition=gpu_h100_il
#SBATCH --mem=510000mb
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/training_1stage_XL_%j.out
#SBATCH --error=logs/training_1stage_XL_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load devel/cuda/12.8

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating it with uv sync..."
    uv sync --extra dev --extra datasets
fi

# Activate uv virtual environment
source .venv/bin/activate

# Load hyperparameters
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPERPARAMS="$SCRIPT_DIR/hyperparams.json"

# Source the hyperparameter loading function
source "$(dirname "$SCRIPT_DIR")/load_hyperparams.sh"

# Load hyperparameters
load_hyperparams "$HYPERPARAMS"

# Set environment variables for distributed training
MASTER_ADDR=$(hostname)
export MASTER_ADDR
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "Number of tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Time limit: $SLURM_TIME_LIMIT"

# Print distributed training environment variables
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Print CUDA information
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run distributed training with diagnostic fixes
echo "Starting distributed training with fixed hyperparameters..."
srun python scripts/train.py \
    --distributed \
    --config-path "$MODEL_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --learning-rate "$LEARNING_RATE" \
    --lr-multipliers "$LR_MULTIPLIERS" \
    --warmup-steps "$WARMUP_STEPS" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --num-training-steps "$NUM_TRAINING_STEPS" \
    --save-interval "$SAVE_INTERVAL" \
    --eval-interval "$EVAL_INTERVAL" \
    --log-interval "$LOG_INTERVAL" \
    --dtype "$DTYPE" \
    --backend "$BACKEND"

echo "Training completed!"
