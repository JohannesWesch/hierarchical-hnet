# H-Net Evaluation Suite

This directory contains tools for evaluating H-Net language models on standard benchmarks, including bits-per-byte (BPB) and downstream zero-shot tasks.

## Overview

The evaluation suite provides:
- **Bits-per-byte (BPB) evaluation** on FineWeb-Edu dataset
- **Downstream zero-shot tasks** via lm-evaluation-harness
- **Unified evaluation pipeline** for reproducing paper results

## Quick Start

### 1. Run Full Evaluation Suite

```bash
./evaluation/run_all.sh \
    outputs/hnet_1stage_L/checkpoint_10000.pt \
    configs/hnet_1stage_L.json \
    evaluation_results
```

This runs both BPB and downstream evaluations and saves results to `evaluation_results/`.

### 2. Test Evaluation Pipeline

Quick test on GPU node:
```bash
sbatch evaluation/test_eval_quick.sh
```

Or interactively:
```bash
srun --partition=gpu --gres=gpu:1 evaluation/test_eval_quick.sh
```

## Individual Evaluation Scripts

### Bits-per-Byte Evaluation

Evaluate model on FineWeb-Edu validation set:

```bash
python -m evaluation.evaluate_bpb \
    --model-path outputs/hnet_1stage_L/checkpoint_10000.pt \
    --config-path configs/hnet_1stage_L.json \
    --max-samples 100 \
    --max-seq-length 8192 \
    --output bpb_results.json
```

**Options:**
- `--model-path`: Path to model checkpoint (.pt file)
- `--config-path`: Path to model config (.json file)
- `--dataset`: HuggingFace dataset name (default: `HuggingFaceFW/fineweb-edu`)
- `--split`: Dataset split (default: `train`)
- `--max-samples`: Number of samples to evaluate (default: 100)
- `--max-seq-length`: Max sequence length (default: 8192)
- `--device`: Device to use (default: auto-detect)
- `--output`: Path to save JSON results

**Output:**
```json
{
  "bits_per_byte": 1.234,
  "perplexity": 3.456,
  "avg_loss_per_byte": 0.789,
  "total_bytes": 1000000,
  "num_samples": 100
}
```

### Downstream Zero-Shot Tasks

Evaluate on standard NLP benchmarks:

```bash
python -m evaluation.evaluate_downstream \
    --model-path outputs/hnet_1stage_L/checkpoint_10000.pt \
    --config-path configs/hnet_1stage_L.json \
    --tasks "lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa" \
    --batch-size 1 \
    --output downstream_results.json
```

**Options:**
- `--model-path`: Path to model checkpoint
- `--config-path`: Path to model config
- `--tasks`: Comma-separated list of tasks
- `--batch-size`: Batch size (default: 1)
- `--device`: Device to use (default: cuda)
- `--limit`: Limit examples per task (for testing)
- `--output`: Path to save JSON results

**Available Tasks:**
- `lambada_openai`: Language modeling task
- `hellaswag`: Commonsense reasoning
- `piqa`: Physical commonsense
- `arc_easy`, `arc_challenge`: Science Q&A
- `winogrande`: Coreference resolution
- `openbookqa`: Open-book Q&A

## Architecture

### Files

```
evaluation/
├── __init__.py                 # Package initialization
├── utils_eval.py              # Model loading utilities
├── evaluate_bpb.py            # BPB evaluation script
├── evaluate_downstream.py     # Downstream task script
├── hnet_lm_wrapper.py         # lm-eval integration
├── run_all.sh                 # Convenience runner
├── test_eval_quick.sh         # Quick test script
└── README.md                  # This file
```

### Key Components

**`utils_eval.py`**: Shared utilities for loading models
```python
from evaluation.utils_eval import load_model_for_eval

model = load_model_for_eval(
    model_path="checkpoint.pt",
    config_path="config.json",
    device="cuda"
)
```

**`hnet_lm_wrapper.py`**: Adapter for lm-eval harness
- Converts between byte-level H-Net and token-based lm-eval API
- Handles text encoding/decoding transparently
- Implements `loglikelihood` and `loglikelihood_rolling` methods

## Usage Examples

### Example 1: Quick Test

Test on small sample before full evaluation:

```bash
python -m evaluation.evaluate_bpb \
    --model-path checkpoint.pt \
    --config-path config.json \
    --max-samples 10 \
    --output quick_test.json
```

### Example 2: Single Task Evaluation

Evaluate on just one downstream task:

```bash
python -m evaluation.evaluate_downstream \
    --model-path checkpoint.pt \
    --config-path config.json \
    --tasks "lambada_openai" \
    --limit 100 \
    --output lambada_results.json
```

### Example 3: Cluster Batch Job

Create SLURM script:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_hnet

conda activate hnet

./evaluation/run_all.sh \
    outputs/hnet_2stage_XL/checkpoint_100000.pt \
    configs/hnet_2stage_XL.json \
    results_2stage_XL
```

Submit: `sbatch eval_job.sh`

## Reproducing Paper Results

To reproduce the H-Net paper benchmarks:

1. **Train or download checkpoints** for each model size (L, XL)
2. **Run full evaluation suite**:
   ```bash
   # Stage 1 Large
   ./evaluation/run_all.sh \
       outputs/hnet_1stage_L/checkpoint_100000.pt \
       configs/hnet_1stage_L.json \
       results_1stage_L

   # Stage 2 XL
   ./evaluation/run_all.sh \
       outputs/hnet_2stage_XL/checkpoint_100000.pt \
       configs/hnet_2stage_XL.json \
       results_2stage_XL
   ```
3. **Compare results** with paper's Table 1

## Notes

### Byte-Level vs Token-Level

H-Net operates on bytes (vocab size 256), not tokens. The evaluation suite handles this transparently:
- **BPB evaluation**: Native byte-level metric
- **Downstream tasks**: Wrapper converts text ↔ bytes automatically

### Memory Considerations

- Default max sequence length: 8192 bytes
- Adjust `--max-seq-length` based on available GPU memory
- Use `--batch-size 1` for large models

### Performance

- BPB evaluation: ~5-10 minutes for 100 samples
- Downstream tasks: ~1-2 hours for all 7 tasks
- Total time: ~2-3 hours per checkpoint

## Troubleshooting

**Import errors**: Run scripts as modules with `python -m evaluation.evaluate_bpb`

**CUDA errors**: Ensure you're on a GPU node (`srun --gres=gpu:1`)

**Out of memory**: Reduce `--max-seq-length` or `--max-samples`

**lm-eval not found**: Install with `conda run -n hnet pip install lm-eval`

## Citation

If you use this evaluation suite, please cite the H-Net paper:

```bibtex
@article{hwang2024dynamic,
  title={Dynamic Chunking for End-to-End Hierarchical Sequence Modeling},
  author={Hwang, Sukjun and Wang, Brandon and Gu, Albert},
  journal={arXiv preprint arXiv:2507.07955},
  year={2024}
}
```
