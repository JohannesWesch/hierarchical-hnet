#!/bin/bash
# Run evaluation tasks in parallel across multiple GPUs
#
# Usage: sbatch evaluation/run_parallel_tasks.sh MODEL_PATH CONFIG_PATH OUTPUT_DIR
# sbatch evaluation/run_parallel_tasks.sh outputs/hnet_1stage_L_v1/checkpoint_200000.pt configs/hnet_1stage_L.json evaluation_results/hnet_1stage_L_v1
# sbatch evaluation/run_parallel_tasks.sh pretrained_models/hnet_1stage_L.pt configs/hnet_1stage_L.json evaluation_results/hnet_1stage_L_pretrained

#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --job-name=eval_parallel
#SBATCH --output=logs/eval_parallel_%j.out
#SBATCH --error=logs/eval_parallel_%j.err

MODEL_PATH=${1:-"outputs/hnet_1stage_L/checkpoint_10000.pt"}
CONFIG_PATH=${2:-"configs/hnet_1stage_L.json"}
OUTPUT_DIR=${3:-"evaluation_results"}

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "================================================================"
echo "Parallel H-Net Evaluation"
echo "================================================================"
echo "Model:      $MODEL_PATH"
echo "Config:     $CONFIG_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "GPUs:       4"
echo "================================================================"

# Define task groups (roughly balanced by size)
# Group 1: hellaswag (10042 questions) - largest, gets its own GPU
# Group 2: lambada_openai (5153)
# Group 3: arc_easy (2376) + piqa (1838)
# Group 4: winogrande (1267) + arc_challenge (1172) + openbookqa (500)

# GPU 0: hellaswag (largest task)
CUDA_VISIBLE_DEVICES=0 uv run python -m evaluation.evaluate_downstream \
    --model-path "$MODEL_PATH" \
    --config-path "$CONFIG_PATH" \
    --tasks "hellaswag" \
    --batch-size 1 \
    --output "$OUTPUT_DIR/hellaswag_results.json" \
    > logs/eval_hellaswag.log 2>&1 &
PID1=$!

# GPU 1: lambada_openai
CUDA_VISIBLE_DEVICES=1 uv run python -m evaluation.evaluate_downstream \
    --model-path "$MODEL_PATH" \
    --config-path "$CONFIG_PATH" \
    --tasks "lambada_openai" \
    --batch-size 1 \
    --output "$OUTPUT_DIR/lambada_results.json" \
    > logs/eval_lambada.log 2>&1 &
PID2=$!

# GPU 2: arc_easy + piqa
CUDA_VISIBLE_DEVICES=2 uv run python -m evaluation.evaluate_downstream \
    --model-path "$MODEL_PATH" \
    --config-path "$CONFIG_PATH" \
    --tasks "arc_easy,piqa" \
    --batch-size 1 \
    --output "$OUTPUT_DIR/arc_easy_piqa_results.json" \
    > logs/eval_arc_easy_piqa.log 2>&1 &
PID3=$!

# GPU 3: winogrande + arc_challenge + openbookqa
CUDA_VISIBLE_DEVICES=3 uv run python -m evaluation.evaluate_downstream \
    --model-path "$MODEL_PATH" \
    --config-path "$CONFIG_PATH" \
    --tasks "winogrande,arc_challenge,openbookqa" \
    --batch-size 1 \
    --output "$OUTPUT_DIR/winogrande_arc_openbookqa_results.json" \
    > logs/eval_winogrande_arc_openbookqa.log 2>&1 &
PID4=$!

echo "Started parallel evaluation on 4 GPUs"
echo "  GPU 0 (PID $PID1): hellaswag"
echo "  GPU 1 (PID $PID2): lambada_openai"
echo "  GPU 2 (PID $PID3): arc_easy, piqa"
echo "  GPU 3 (PID $PID4): winogrande, arc_challenge, openbookqa"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/eval_*.log"
echo ""

# Wait for all background jobs to complete
wait $PID1
echo "✓ GPU 0 complete (hellaswag)"

wait $PID2
echo "✓ GPU 1 complete (lambada_openai)"

wait $PID3
echo "✓ GPU 2 complete (arc_easy, piqa)"

wait $PID4
echo "✓ GPU 3 complete (winogrande, arc_challenge, openbookqa)"

echo ""
echo "================================================================"
echo "All evaluations complete!"
echo "================================================================"
echo "Results saved to: $OUTPUT_DIR/"
echo "  - $OUTPUT_DIR/hellaswag_results.json"
echo "  - $OUTPUT_DIR/lambada_results.json"
echo "  - $OUTPUT_DIR/arc_easy_piqa_results.json"
echo "  - $OUTPUT_DIR/winogrande_arc_openbookqa_results.json"
echo "================================================================"

# Optionally merge results
echo ""
echo "To merge all results:"
echo "  python -c \"import json; import glob; merged = {}; [merged.update(json.load(open(f))) for f in glob.glob('$OUTPUT_DIR/*_results.json')]; json.dump(merged, open('$OUTPUT_DIR/all_results.json', 'w'), indent=2)\""
