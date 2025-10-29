#!/bin/bash
# Resume training with fixed hyperparameters (non-distributed version)

python scripts/train_fixed.py \
    --config-path configs/hnet_2stage_XL.json \
    --output-dir outputs/hnet_2stage_XL_fixed \
    --learning-rate 2e-4 \
    --lr-multipliers 2.0,1.5,1.0 \
    --warmup-steps 5000 \
    --max-grad-norm 5.0 \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --num-training-steps 100000 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --dtype bfloat16 \
    --resume-from outputs/hnet_2stage_XL_distributed/checkpoint_9000.pt
