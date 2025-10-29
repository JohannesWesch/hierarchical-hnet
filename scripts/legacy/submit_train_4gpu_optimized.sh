#!/bin/bash
#SBATCH --job-name=hnet-distributed-training-optimized
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=760000mb
#SBATCH --time=12:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

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

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
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

# Run distributed training with optimized settings
echo "Starting distributed training with optimized settings..."
srun python scripts/train.py \
    --distributed \
    --config-path configs/hnet_2stage_XL.json \
    --batch-size 48 \
    --gradient-accumulation-steps 1 \
    --learning-rate 3e-4 \
    --max-seq-length 2048 \
    --num-training-steps 100000 \
    --save-interval 1000 \
    --eval-interval 500 \
    --log-interval 10 \
    --output-dir ./outputs/hnet_2stage_XL_distributed_optimized \
    --backend nccl

echo "Training completed!"
