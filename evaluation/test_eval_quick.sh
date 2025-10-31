#!/bin/bash
# Quick test script for evaluation suite
# Run this on a GPU node to verify the evaluation pipeline works
#
# Usage:
#   sbatch evaluation/test_eval_quick.sh
#   OR
#   srun --partition=gpu --gres=gpu:1 evaluation/test_eval_quick.sh

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=test_eval
#SBATCH --output=logs/test_eval_%j.out
#SBATCH --error=logs/test_eval_%j.err

set -e

echo "Testing H-Net Evaluation Suite"
echo "=============================="
echo ""

# Set paths
MODEL_PATH="outputs/hnet_1stage_L/checkpoint_10000.pt"
CONFIG_PATH="configs/hnet_1stage_L.json"
OUTPUT_DIR="test_evaluation_results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Test 1: BPB evaluation with small sample
echo "[Test 1/2] Testing BPB evaluation (5 samples)..."
uv run python -m evaluation.evaluate_bpb \
    --model-path "$MODEL_PATH" \
    --config-path "$CONFIG_PATH" \
    --max-samples 5 \
    --max-seq-length 2048 \
    --output "$OUTPUT_DIR/test_bpb.json"

echo "✓ BPB test passed!"
echo ""

# Test 2: Downstream evaluation with single task and limit
echo "[Test 2/2] Testing downstream evaluation (lambada_openai, 10 examples)..."
uv run python -m evaluation.evaluate_downstream \
    --model-path "$MODEL_PATH" \
    --config-path "$CONFIG_PATH" \
    --tasks "lambada_openai" \
    --batch-size 1 \
    --limit 10 \
    --output "$OUTPUT_DIR/test_downstream.json"

echo "✓ Downstream test passed!"
echo ""

echo "=============================="
echo "All tests passed successfully!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=============================="
