#!/bin/bash
#SBATCH --job-name=hnet_fixed_training
#SBATCH --output=outputs/hnet_2stage_XL_fixed/train_%j.log
#SBATCH --error=outputs/hnet_2stage_XL_fixed/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --partition=gpu

# =============================================================================
# CUSTOMIZE THESE SETTINGS FOR YOUR CLUSTER
# =============================================================================

# Load required modules (uncomment and modify as needed)
# module load python/3.9
# module load cuda/11.8
# module load gcc/9.3.0
# module load openmpi/4.1.0

# Activate your conda environment
source ~/.bashrc
conda activate hnet

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=4
export RANK=0

# Training parameters (you can modify these)
LEARNING_RATE=2e-4
LR_MULTIPLIERS="2.0,1.5,1.0"
WARMUP_STEPS=5000
MAX_GRAD_NORM=5.0
BATCH_SIZE=8
GRADIENT_ACCUMULATION=1
NUM_TRAINING_STEPS=100000
SAVE_INTERVAL=1000
EVAL_INTERVAL=1000
DTYPE=bfloat16

# Resume from checkpoint (set to "none" to start from scratch)
RESUME_FROM="outputs/hnet_2stage_XL_distributed/checkpoint_9000.pt"

# =============================================================================
# JOB EXECUTION
# =============================================================================

# Create output directory
mkdir -p outputs/hnet_2stage_XL_fixed

# Print job information
echo "=========================================="
echo "H-Net Fixed Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="
echo "Training Configuration:"
echo "  Learning Rate: $LEARNING_RATE"
echo "  LR Multipliers: $LR_MULTIPLIERS"
echo "  Warmup Steps: $WARMUP_STEPS"
echo "  Max Grad Norm: $MAX_GRAD_NORM"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "  Training Steps: $NUM_TRAINING_STEPS"
echo "  Resume From: $RESUME_FROM"
echo "=========================================="

# Check if resume checkpoint exists
if [ "$RESUME_FROM" != "none" ] && [ ! -f "$RESUME_FROM" ]; then
    echo "WARNING: Resume checkpoint not found: $RESUME_FROM"
    echo "Starting training from scratch..."
    RESUME_FROM=""
fi

# Build the training command
TRAIN_CMD="python scripts/train_fixed.py \
    --config-path configs/hnet_2stage_XL.json \
    --output-dir outputs/hnet_2stage_XL_fixed \
    --learning-rate $LEARNING_RATE \
    --lr-multipliers $LR_MULTIPLIERS \
    --warmup-steps $WARMUP_STEPS \
    --max-grad-norm $MAX_GRAD_NORM \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
    --num-training-steps $NUM_TRAINING_STEPS \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --dtype $DTYPE \
    --distributed"

# Add resume flag if checkpoint exists
if [ -n "$RESUME_FROM" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume-from $RESUME_FROM"
fi

# Run the training
echo "Starting training..."
echo "Command: $TRAIN_CMD"
echo "=========================================="

eval $TRAIN_CMD

# Check training exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully at: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training failed at: $(date)"
    echo "Check the error logs for details"
    echo "=========================================="
    exit 1
fi
