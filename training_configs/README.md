# H-Net Training Configuration

This directory contains organized training configurations for different H-Net model variants.

## 📁 Directory Structure

```
training_configs/
├── 1stage_L/           # 1-stage Large model
│   ├── hyperparams.json
│   ├── train_cluster.sh    # SLURM batch script
│   └── train_dev.sh        # Direct GPU script
├── 1stage_XL/          # 1-stage Extra Large model
│   ├── hyperparams.json
│   ├── train_cluster.sh
│   └── train_dev.sh
├── 2stage_L/           # 2-stage Large model
│   ├── hyperparams.json
│   ├── train_cluster.sh
│   └── train_dev.sh
├── 2stage_XL/          # 2-stage Extra Large model
│   ├── hyperparams.json
│   ├── train_cluster.sh
│   └── train_dev.sh
├── load_hyperparams.sh # Helper script for loading configs
└── README.md           # This file
```

## 🚀 Quick Start

### Cluster Training (SLURM)

Submit a training job to the cluster:

```bash
# 2-stage XL model (most common)
sbatch training_configs/2stage_XL/train_cluster.sh

# 2-stage L model
sbatch training_configs/2stage_L/train_cluster.sh

# 1-stage XL model
sbatch training_configs/1stage_XL/train_cluster.sh

# 1-stage L model
sbatch training_configs/1stage_L/train_cluster.sh
```

### Select different GPUS
# gpu_h100
```bash
#SBATCH --partition=gpu_h100
#SBATCH --mem=760000mb
#SBATCH --time=72:00:00
```

# gpu_h100_il
```bash
#SBATCH --partition=gpu_h100_il
#SBATCH --mem=510000mb
#SBATCH --time=48:00:00
```

### Dev GPU Training

Run training directly on your development GPU:

```bash
# 2-stage XL model
./training_configs/2stage_XL/train_dev.sh

# 2-stage L model
./training_configs/2stage_L/train_dev.sh

# 1-stage XL model
./training_configs/1stage_XL/train_dev.sh

# 1-stage L model
./training_configs/1stage_L/train_dev.sh
```

## ⚙️ Configuration

### Hyperparameter Files

Each model type has a `hyperparams.json` file containing:

```json
{
  "model_config": "configs/hnet_2stage_XL.json",
  "output_dir": "outputs/hnet_2stage_XL",
  "learning_rate": 2e-4,
  "lr_multipliers": "2.0,1.5,1.0",
  "warmup_steps": 5000,
  "max_grad_norm": 5.0,
  "batch_size": 8,
  "gradient_accumulation_steps": 4,
  "max_seq_length": 2048,
  "num_training_steps": 100000,
  "save_interval": 1000,
  "eval_interval": 500,
  "log_interval": 10,
  "dtype": "bfloat16",
  "backend": "nccl"
}
```

### Model-Specific Settings

- **1-stage models**: Use 2 LR multipliers (e.g., "2.0,1.0")
- **2-stage models**: Use 3 LR multipliers (e.g., "2.0,1.5,1.0")
- **L vs XL**: Different batch sizes based on memory requirements
- **All models**: Include diagnostic fixes (stable training)

## 🔧 Customization

### Modifying Hyperparameters

Edit the `hyperparams.json` file in the desired model directory:

```bash
# Edit 2-stage XL hyperparameters
nano training_configs/2stage_XL/hyperparams.json
```

### Adding New Model Types

1. Create new directory: `training_configs/new_model/`
2. Copy `hyperparams.json` from similar model
3. Update model config path and output directory
4. Copy and modify `train_cluster.sh` and `train_dev.sh`
5. Update job names and log file names

## 📊 Monitoring Training

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# View job details
scontrol show job <JOB_ID>
```

### Monitor Logs

```bash
# Watch training progress
tail -f logs/training_2stage_XL_<JOB_ID>.out

# Check for errors
tail -f logs/training_2stage_XL_<JOB_ID>.err
```

### Cancel Jobs

```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER
```

## 🎯 Model Selection Guide

### 2-stage XL (Recommended)
- **Use case**: Best performance, most complex
- **Resources**: 4x H100 GPUs, 760GB RAM
- **Training time**: ~12 hours for 100k steps
- **Script**: `training_configs/2stage_XL/train_cluster.sh`

### 2-stage L
- **Use case**: Good performance, moderate complexity
- **Resources**: 4x H100 GPUs, 760GB RAM
- **Training time**: ~10 hours for 100k steps
- **Script**: `training_configs/2stage_L/train_cluster.sh`

### 1-stage XL
- **Use case**: Simpler architecture, still large
- **Resources**: 4x H100 GPUs, 760GB RAM
- **Training time**: ~8 hours for 100k steps
- **Script**: `training_configs/1stage_XL/train_cluster.sh`

### 1-stage L
- **Use case**: Fastest training, good for testing
- **Resources**: 4x H100 GPUs, 760GB RAM
- **Training time**: ~6 hours for 100k steps
- **Script**: `training_configs/1stage_L/train_cluster.sh`

## 🔍 Diagnostic Fixes Applied

All training configurations include the following stability improvements:

- **Learning Rate**: 2e-4 (reduced from 3e-4)
- **LR Multipliers**: Reduced ratios (2.0,1.5,1.0 vs 3.0,1.7,0.9)
- **Warmup Steps**: 5000 (increased from 2000)
- **Gradient Clipping**: 5.0 (increased from 1.0)
- **Stable Training**: Prevents gradient explosion

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `batch_size` in hyperparams.json
   - Increase `gradient_accumulation_steps`

2. **Module Not Found**:
   - Ensure virtual environment is activated
   - Check if `.venv` directory exists

3. **Permission Denied**:
   - Make scripts executable: `chmod +x *.sh`

4. **Checkpoint Not Found**:
   - Verify checkpoint path exists
   - Remove `--resume-from` to start fresh

### Getting Help

- Check logs in `logs/` directory
- Review hyperparameter settings in `hyperparams.json`
- Ensure model config files exist in `configs/`
- Verify SLURM partition and resource availability

## 📝 Notes

- All scripts use `train.py` with diagnostic improvements
- Chinese and Code model configs remain in `configs/` folder
- Main training script stays in `scripts/train.py`
- Hyperparameters are easily adjustable per model type
- Both cluster and dev scripts share the same configuration
