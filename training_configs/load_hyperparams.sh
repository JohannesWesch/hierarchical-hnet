#!/bin/bash
# Helper script to load hyperparameters from JSON file

load_hyperparams() {
    local config_file=$1

    if [ ! -f "$config_file" ]; then
        echo "Error: Hyperparameter config file not found: $config_file"
        exit 1
    fi

    # Load all hyperparameters from JSON file
    MODEL_CONFIG=$(python -c "import json; print(json.load(open('$config_file'))['model_config'])")
    OUTPUT_DIR=$(python -c "import json; print(json.load(open('$config_file'))['output_dir'])")
    LEARNING_RATE=$(python -c "import json; print(json.load(open('$config_file'))['learning_rate'])")
    LR_MULTIPLIERS=$(python -c "import json; print(json.load(open('$config_file'))['lr_multipliers'])")
    WARMUP_STEPS=$(python -c "import json; print(json.load(open('$config_file'))['warmup_steps'])")
    MAX_GRAD_NORM=$(python -c "import json; print(json.load(open('$config_file'))['max_grad_norm'])")
    BATCH_SIZE=$(python -c "import json; print(json.load(open('$config_file'))['batch_size'])")
    GRADIENT_ACCUMULATION_STEPS=$(python -c "import json; print(json.load(open('$config_file'))['gradient_accumulation_steps'])")
    MAX_SEQ_LENGTH=$(python -c "import json; print(json.load(open('$config_file'))['max_seq_length'])")
    NUM_TRAINING_STEPS=$(python -c "import json; print(json.load(open('$config_file'))['num_training_steps'])")
    SAVE_INTERVAL=$(python -c "import json; print(json.load(open('$config_file'))['save_interval'])")
    EVAL_INTERVAL=$(python -c "import json; print(json.load(open('$config_file'))['eval_interval'])")
    LOG_INTERVAL=$(python -c "import json; print(json.load(open('$config_file'))['log_interval'])")
    DTYPE=$(python -c "import json; print(json.load(open('$config_file'))['dtype'])")
    BACKEND=$(python -c "import json; print(json.load(open('$config_file'))['backend'])")

    echo "Loaded hyperparameters from: $config_file"
    echo "Model config: $MODEL_CONFIG"
    echo "Output dir: $OUTPUT_DIR"
    echo "Learning rate: $LEARNING_RATE"
    echo "LR multipliers: $LR_MULTIPLIERS"
    echo "Warmup steps: $WARMUP_STEPS"
    echo "Max grad norm: $MAX_GRAD_NORM"
    echo "Batch size: $BATCH_SIZE"
    echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
    echo "Max seq length: $MAX_SEQ_LENGTH"
    echo "Training steps: $NUM_TRAINING_STEPS"
    echo "Save interval: $SAVE_INTERVAL"
    echo "Eval interval: $EVAL_INTERVAL"
    echo "Log interval: $LOG_INTERVAL"
    echo "Data type: $DTYPE"
    echo "Backend: $BACKEND"
}
