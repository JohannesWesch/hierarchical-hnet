#!/bin/bash
# Local testing script for distributed training without SLURM
# This script uses torchrun to launch distributed training locally

# Configuration
NUM_GPUS=4  # Change this to match your available GPUs
CONFIG_PATH="configs/hnet_2stage_XL.json"
OUTPUT_DIR="./outputs/hnet_2stage_XL_local_test"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting local distributed training with $NUM_GPUS GPUs"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"

# Launch distributed training using torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    scripts/train.py \
    --distributed \
    --config-path $CONFIG_PATH \
    --batch-size 8 \
    --gradient-accumulation-steps 1 \
    --learning-rate 3e-4 \
    --max-seq-length 2048 \
    --num-training-steps 1000 \
    --save-interval 100 \
    --eval-interval 50 \
    --log-interval 5 \
    --output-dir $OUTPUT_DIR \
    --backend nccl

echo "Local distributed training completed!"
