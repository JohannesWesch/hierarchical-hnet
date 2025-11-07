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
    # Handle lr_multipliers as either JSON array or string
    LR_MULTIPLIERS=$(python -c "import json; m=json.load(open('$config_file'))['lr_multipliers']; print(','.join(map(str, m)) if isinstance(m, list) else m)")
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

    # Load optimizer hyperparameters (with defaults for backward compatibility)
    WEIGHT_DECAY=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('weight_decay', 0.1))")
    LOAD_BALANCING_WEIGHT=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('load_balancing_weight', 0.01))")
    # Handle downsampling_factors as either JSON array or string
    DOWNSAMPLING_FACTORS=$(python -c "import json; c=json.load(open('$config_file')); d=c.get('downsampling_factors', None); print(','.join(map(str, d)) if d is not None and isinstance(d, list) else (d if d is not None else 'None'))")
    ADAM_BETA1=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('adam_beta1', 0.9))")
    ADAM_BETA2=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('adam_beta2', 0.95))")
    ADAM_EPS=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('adam_eps', 1e-08))")
    MIN_LR=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('min_lr', 1e-05))")

    # Load LR scheduler parameters
    LR_SCHEDULER=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('lr_scheduler', 'cosine'))")
    WARMUP_RATIO=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('warmup_ratio', 'None'))")
    STABLE_RATIO=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('stable_ratio', 'None'))")
    DECAY_RATIO=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('decay_ratio', 'None'))")
    DECAY_TYPE=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('decay_type', 'inverse_sqrt'))")

    # Load regularization parameters
    LABEL_SMOOTHING=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('label_smoothing', 0.0))")
    EMA_DECAY=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('ema_decay', 'None'))")
    DROPOUT=$(python -c "import json; c=json.load(open('$config_file')); print(c.get('dropout', 'None'))")
    SCALE_LR_BY_WORLD_SIZE=$(python -c "import json; c=json.load(open('$config_file')); print(str(c.get('scale_lr_by_world_size', False)).lower())")

    echo "Loaded hyperparameters from: $config_file"
    echo "Model config: $MODEL_CONFIG"
    echo "Output dir: $OUTPUT_DIR"
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
    echo "Save interval: $SAVE_INTERVAL"
    echo "Eval interval: $EVAL_INTERVAL"
    echo "Log interval: $LOG_INTERVAL"
    echo "Data type: $DTYPE"
    echo "Backend: $BACKEND"
    echo "Weight decay: $WEIGHT_DECAY"
    echo "Load balancing weight: $LOAD_BALANCING_WEIGHT"
    if [ "$DOWNSAMPLING_FACTORS" != "None" ]; then
        echo "Downsampling factors: $DOWNSAMPLING_FACTORS"
    fi
    echo "Adam beta1: $ADAM_BETA1"
    echo "Adam beta2: $ADAM_BETA2"
    echo "Adam eps: $ADAM_EPS"
    echo "Label smoothing: $LABEL_SMOOTHING"
    if [ "$EMA_DECAY" != "None" ]; then
        echo "EMA decay: $EMA_DECAY"
    fi
    if [ "$DROPOUT" != "None" ]; then
        echo "Dropout: $DROPOUT"
    fi
}
