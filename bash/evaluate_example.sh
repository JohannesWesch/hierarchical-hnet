#!/bin/bash
# Example evaluation script for H-Net

# Model configuration
MODEL_PATH="./outputs/hnet_2stage_L/checkpoint_100000.pt"
CONFIG_PATH="configs/hnet_2stage_L.json"

# Evaluation data
DATA_PATH="/path/to/test/data"

# Settings
BATCH_SIZE=16
MAX_SEQ_LENGTH=2048
OUTPUT_PATH="./outputs/hnet_2stage_L/evaluation_results.json"

# Run evaluation
python ../scripts/evaluate.py \
    --model-path $MODEL_PATH \
    --config-path $CONFIG_PATH \
    --data-path $DATA_PATH \
    --batch-size $BATCH_SIZE \
    --max-seq-length $MAX_SEQ_LENGTH \
    --output-path $OUTPUT_PATH \
    --device cuda

