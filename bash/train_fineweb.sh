#!/bin/bash
# Training script for H-Net using FineWeb-Edu dataset from HuggingFace

# Training configuration
CONFIG_PATH="configs/hnet_2stage_L.json"
OUTPUT_DIR="./outputs/hnet_2stage_L_fineweb"

# Hyperparameters
BATCH_SIZE=8
GRADIENT_ACCUM=4
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

# Run training with FineWeb-Edu dataset
python ../scripts/train.py \
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
    --log-interval 10

