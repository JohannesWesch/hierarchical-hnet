#!/bin/bash
# Distributed training script for H-Net using FineWeb-Edu dataset with automatic resume support

# Load required modules
module load devel/cuda/12.8

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating it with uv sync..."
    uv sync --extra dev --extra datasets
fi

# Activate uv virtual environment
source .venv/bin/activate

# Training configuration
CONFIG_PATH="configs/hnet_2stage_XL.json"
OUTPUT_DIR="./outputs/hnet_2stage_XL_fineweb_distributed"

# Hyperparameters (optimized for distributed training)
BATCH_SIZE=16  # Increased for distributed training
GRADIENT_ACCUM=2  # Reduced since we have more GPUs
LEARNING_RATE=3e-4
LR_MULTIPLIERS="3.0,1.7,0.9"
WEIGHT_DECAY=0.1
MAX_GRAD_NORM=1.0
LB_WEIGHT=0.01

# Training schedule
NUM_STEPS=100000
WARMUP_STEPS=2000

# System settings
DTYPE="bfloat16"
SEED=42

# Find latest checkpoint
LATEST_CHECKPOINT=""
if [ -d "$OUTPUT_DIR" ]; then
    # Find the checkpoint with the highest step number
    LATEST_CHECKPOINT=$(ls -1 $OUTPUT_DIR/checkpoint_*.pt 2>/dev/null | sort -V | tail -n 1)
fi

# Build distributed training command
CMD="srun python scripts/train.py \
    --distributed \
    --config-path $CONFIG_PATH \
    --output-dir $OUTPUT_DIR \
    --use-hf-dataset \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUM \
    --learning-rate $LEARNING_RATE \
    --lr-multipliers $LR_MULTIPLIERS \
    --weight-decay $WEIGHT_DECAY \
    --max-grad-norm $MAX_GRAD_NORM \
    --load-balancing-weight $LB_WEIGHT \
    --num-training-steps $NUM_STEPS \
    --warmup-steps $WARMUP_STEPS \
    --dtype $DTYPE \
    --seed $SEED \
    --save-interval 1000 \
    --eval-interval 500 \
    --log-interval 10 \
    --backend nccl"

# Add resume flag if checkpoint exists
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "Found checkpoint: $LATEST_CHECKPOINT"
    echo "Resuming distributed training from checkpoint..."
    CMD="$CMD --resume-from $LATEST_CHECKPOINT"
else
    echo "No checkpoint found. Starting distributed training from scratch..."
fi

# Run distributed training
eval $CMD
