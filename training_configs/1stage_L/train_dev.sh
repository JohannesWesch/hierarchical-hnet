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
echo "Starting H-Net 1-stage L training on dev GPU with WSD scheduler..."
echo "Model config: $MODEL_CONFIG"
echo "Output directory: $OUTPUT_DIR"
echo "Learning rate: $LEARNING_RATE"
echo "Scale LR by world size: $SCALE_LR_BY_WORLD_SIZE"
echo "LR multipliers: $LR_MULTIPLIERS"
echo "LR scheduler: $LR_SCHEDULER"
if [ "$WARMUP_RATIO" != "None" ]; then
    echo "Warmup ratio: $WARMUP_RATIO"
    echo "Stable ratio: $STABLE_RATIO"
    echo "Decay ratio: $DECAY_RATIO"
    echo "Decay type: $DECAY_TYPE"
fi
echo "Min LR: $MIN_LR"
echo "Max grad norm: $MAX_GRAD_NORM"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Max seq length: $MAX_SEQ_LENGTH"
echo "Training steps: $NUM_TRAINING_STEPS"
echo "Data type: $DTYPE"

# Run training on dev GPU with WSD scheduler
echo "Starting training with WSD scheduler..."

# Build command with required args
CMD="python scripts/train.py \
    --config-path $MODEL_CONFIG \
    --output-dir $OUTPUT_DIR \
    --learning-rate $LEARNING_RATE \
    --lr-multipliers $LR_MULTIPLIERS \
    --lr-scheduler $LR_SCHEDULER \
    --min-lr $MIN_LR \
    --max-grad-norm $MAX_GRAD_NORM \
    --weight-decay $WEIGHT_DECAY \
    --load-balancing-weight $LOAD_BALANCING_WEIGHT \
    --adam-beta1 $ADAM_BETA1 \
    --adam-beta2 $ADAM_BETA2 \
    --adam-eps $ADAM_EPS \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
    --max-seq-length $MAX_SEQ_LENGTH \
    --num-training-steps $NUM_TRAINING_STEPS \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --log-interval $LOG_INTERVAL \
    --dtype $DTYPE \
    --backend $BACKEND \
    --label-smoothing $LABEL_SMOOTHING"

# Add WSD-specific parameters if set
if [ "$WARMUP_RATIO" != "None" ]; then
    CMD="$CMD --warmup-ratio $WARMUP_RATIO"
fi
if [ "$STABLE_RATIO" != "None" ]; then
    CMD="$CMD --stable-ratio $STABLE_RATIO"
fi
if [ "$DECAY_RATIO" != "None" ]; then
    CMD="$CMD --decay-ratio $DECAY_RATIO"
fi
if [ "$DECAY_TYPE" != "None" ]; then
    CMD="$CMD --decay-type $DECAY_TYPE"
fi

# Add optional parameters if set
if [ "$EMA_DECAY" != "None" ]; then
    CMD="$CMD --ema-decay $EMA_DECAY"
fi
if [ "$DROPOUT" != "None" ]; then
    CMD="$CMD --dropout $DROPOUT"
fi
if [ "$SCALE_LR_BY_WORLD_SIZE" = "true" ]; then
    CMD="$CMD --scale-lr-by-world-size"
fi

# Execute training
eval "$CMD"

echo "Training completed!"
