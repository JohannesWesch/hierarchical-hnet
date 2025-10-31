#!/bin/bash
# Run full evaluation suite for H-Net models
#
# Usage:
#   ./evaluation/run_all.sh MODEL_PATH CONFIG_PATH [OUTPUT_DIR] [LIMIT]
#
# Examples:
#   # Full evaluation
#   ./evaluation/run_all.sh outputs/hnet_1stage_L/checkpoint_10000.pt configs/hnet_1stage_L.json
#
#   # Quick test with 100 samples per task
#   ./evaluation/run_all.sh outputs/hnet_1stage_L/checkpoint_10000.pt configs/hnet_1stage_L.json results_test 100

set -e  # Exit on error

# Parse arguments
MODEL_PATH=$1
CONFIG_PATH=$2
OUTPUT_DIR=${3:-"evaluation_results"}
LIMIT=${4:-""}

# Validate arguments
if [ -z "$MODEL_PATH" ] || [ -z "$CONFIG_PATH" ]; then
    echo "Usage: $0 MODEL_PATH CONFIG_PATH [OUTPUT_DIR] [LIMIT]"
    echo ""
    echo "Arguments:"
    echo "  MODEL_PATH    Path to model checkpoint (.pt file)"
    echo "  CONFIG_PATH   Path to model configuration (.json file)"
    echo "  OUTPUT_DIR    Directory to save results (default: evaluation_results)"
    echo "  LIMIT         Max samples per task (default: none = full evaluation)"
    echo ""
    echo "Examples:"
    echo "  # Full evaluation"
    echo "  $0 outputs/hnet_1stage_L/checkpoint_10000.pt configs/hnet_1stage_L.json"
    echo ""
    echo "  # Quick test with 100 samples per task"
    echo "  $0 outputs/hnet_1stage_L/checkpoint_10000.pt configs/hnet_1stage_L.json results_test 100"
    exit 1
fi

# Check if files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo "H-Net Evaluation Suite"
echo "================================================================"
echo "Model:      $MODEL_PATH"
echo "Config:     $CONFIG_PATH"
echo "Output dir: $OUTPUT_DIR"
if [ -n "$LIMIT" ]; then
    echo "Limit:      $LIMIT samples per task"
else
    echo "Limit:      Full evaluation (no limit)"
fi
echo "================================================================"
echo ""

# Run BPB evaluation
echo "[1/2] Running bits-per-byte evaluation..."
echo "----------------------------------------------------------------"
uv run python -m evaluation.evaluate_bpb \
    --model-path "$MODEL_PATH" \
    --config-path "$CONFIG_PATH" \
    --max-samples 100 \
    --max-seq-length 8192 \
    --output "$OUTPUT_DIR/bpb_results.json"

echo ""
echo "✓ BPB evaluation complete!"
echo ""

# Run downstream evaluation
echo "[2/2] Running downstream zero-shot tasks..."
echo "----------------------------------------------------------------"
DOWNSTREAM_CMD="uv run python -m evaluation.evaluate_downstream \
    --model-path \"$MODEL_PATH\" \
    --config-path \"$CONFIG_PATH\" \
    --tasks \"lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa\" \
    --batch-size 1 \
    --output \"$OUTPUT_DIR/downstream_results.json\""

# Add limit if specified
if [ -n "$LIMIT" ]; then
    DOWNSTREAM_CMD="$DOWNSTREAM_CMD --limit $LIMIT"
fi

eval "$DOWNSTREAM_CMD"

echo ""
echo "✓ Downstream evaluation complete!"
echo ""

# Summary
echo "================================================================"
echo "Evaluation Complete!"
echo "================================================================"
echo "Results saved to: $OUTPUT_DIR/"
echo "  - BPB results:        $OUTPUT_DIR/bpb_results.json"
echo "  - Downstream results: $OUTPUT_DIR/downstream_results.json"
echo "================================================================"
