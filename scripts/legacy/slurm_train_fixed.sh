#!/bin/bash
#SBATCH --job-name=hnet_fixed_training
#SBATCH --output=outputs/hnet_2stage_XL_fixed/train_%j.log
#SBATCH --error=outputs/hnet_2stage_XL_fixed/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --partition=gpu

# Load any required modules (adjust based on your cluster)
# module load python/3.9
# module load cuda/11.8
# module load gcc/9.3.0

# Activate your conda environment
source ~/.bashrc
conda activate hnet

# Set environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=4
export RANK=0

# Create output directory if it doesn't exist
mkdir -p outputs/hnet_2stage_XL_fixed

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Run the fixed training script
python scripts/train_fixed.py \
    --config-path configs/hnet_2stage_XL.json \
    --output-dir outputs/hnet_2stage_XL_fixed \
    --learning-rate 2e-4 \
    --lr-multipliers 2.0,1.5,1.0 \
    --warmup-steps 5000 \
    --max-grad-norm 5.0 \
    --batch-size 8 \
    --gradient-accumulation-steps 1 \
    --num-training-steps 100000 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --dtype bfloat16 \
    --distributed \
    --resume-from outputs/hnet_2stage_XL_distributed/checkpoint_9000.pt

echo "Training completed at: $(date)"
