#!/bin/bash
# Example distributed training script for H-Net using torchrun

# Training configuration
CONFIG_PATH="configs/hnet_2stage_L.json"
TRAIN_DATA="/path/to/train/data"
VAL_DATA="/path/to/val/data"
OUTPUT_DIR="./outputs/hnet_2stage_L_distributed"

# Distributed settings
NUM_NODES=1
NUM_GPUS=8
MASTER_ADDR="localhost"
MASTER_PORT=29500

# Hyperparameters
BATCH_SIZE=4  # Per GPU
GRADIENT_ACCUM=8
LEARNING_RATE=3e-4
LR_MULTIPLIERS="3.0,1.7,0.9"

# Run distributed training
torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    --config-path $CONFIG_PATH \
    --train-data-path $TRAIN_DATA \
    --val-data-path $VAL_DATA \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUM \
    --learning-rate $LEARNING_RATE \
    --lr-multipliers $LR_MULTIPLIERS \
    --num-training-steps 100000 \
    --warmup-steps 2000 \
    --dtype bfloat16 \
    --seed 42

