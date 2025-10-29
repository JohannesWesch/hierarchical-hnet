#!/bin/bash

# Load hyperparameters
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPERPARAMS="$SCRIPT_DIR/hyperparams.json"

# Source the hyperparameter loading function
source "$(dirname "$SCRIPT_DIR")/load_hyperparams.sh"

# Load hyperparameters
load_hyperparams "$HYPERPARAMS"

# Activate environment
source .venv/bin/activate

# Print training information
echo "Starting H-Net 1-stage L training on dev GPU..."
echo "Model config: $MODEL_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "LR multipliers: $LR_MULTIPLIERS"
echo "Warmup steps: $WARMUP_STEPS"
echo "Max grad norm: $MAX_GRAD_NORM"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo "Training steps: $NUM_TRAINING_STEPS"
echo "Data type: $DTYPE"

# Run training on dev GPU
python scripts/train_fixed.py \
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
    --dtype "$DTYPE"

echo "Training completed!"
