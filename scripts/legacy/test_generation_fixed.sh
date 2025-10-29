# Test generation with fixed parameters
python scripts/generate_fixed.py \
    --model-path outputs/hnet_2stage_XL_distributed/checkpoint_9000.pt \
    --config-path configs/hnet_2stage_XL.json \
    --temperature 0.8 \
    --top-p 0.9 \
    --max-tokens 2048