#!/bin/bash
#SBATCH --job-name=hnet-1stage-L
#SBATCH --partition=gpu_h100_il
#SBATCH --mem=510000mb
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/training_1stage_L_%j.out
#SBATCH --error=logs/training_1stage_L_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules
module load devel/cuda/12.8

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating it with uv sync..."
    uv sync --extra dev --extra datasets
fi

# Activate uv virtual environment
source .venv/bin/activate

# Ensure we run from the submission directory so relative paths resolve
cd "$SLURM_SUBMIT_DIR" || exit

# Source the hyperparameter loading function
source training_configs/load_hyperparams.sh

# Load hyperparameters
HYPERPARAMS="training_configs/1stage_L/hyperparams.json"
load_hyperparams "$HYPERPARAMS"

# Set environment variables for distributed training
MASTER_ADDR=$(hostname)
export MASTER_ADDR
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODEID"
echo "Number of tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Time limit: $SLURM_TIME_LIMIT"

# Print distributed training environment variables
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "LOCAL_RANK: $LOCAL_RANK"

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Print CUDA information
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run distributed training with WSD scheduler
echo "Starting distributed training with WSD scheduler..."

# Build command with required args
CMD="srun python scripts/train.py \
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
