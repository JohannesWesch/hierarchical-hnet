# Scripts Directory

This directory contains all executable scripts organized by category.

## 📁 Directory Structure

```
scripts/
├── analysis/           # Analysis and diagnostic scripts
│   ├── analyze_architecture.py
│   ├── analyze_data_pipeline.py
│   ├── analyze_generation.py
│   ├── analyze_training_config.py
│   └── analyze_training.py
├── utilities/          # Utility and monitoring scripts
│   ├── monitor_training.py
│   ├── test_fixes.py
│   └── test_fixes_simple.py
├── legacy/             # Old and deprecated scripts
│   ├── submit_train_4gpu.sh
│   ├── submit_train_4gpu_optimized.sh
│   ├── submit_train_multi_node.sh
│   ├── train_distributed.sh
│   ├── train_example.sh
│   ├── train_fineweb_distributed.sh
│   ├── train_fineweb_resume.sh
│   ├── train_fineweb.sh
│   ├── slurm_train_custom.sh
│   ├── slurm_train_fixed.sh
│   ├── fix_training_config.py
│   ├── resume_training_local.sh
│   └── test_generation_fixed.sh
├── train_fixed.py      # Main training script (with diagnostic fixes)
├── train.py            # Original training script
├── generate_fixed.py   # Generation script (with fixes)
├── generate.py         # Original generation script
├── evaluate.py         # Model evaluation script
├── download_checkpoint.py
├── run_distributed_local.sh
├── evaluate_example.sh
└── README.md           # This file
```

## 🎯 **Main Scripts (Keep in Root)**

### **Training Scripts**
- `train_fixed.py` - Main training script with diagnostic fixes
- `train.py` - Original training script

### **Generation Scripts**
- `generate_fixed.py` - Generation with improved parameters
- `generate.py` - Original generation script

### **Evaluation Scripts**
- `evaluate.py` - Model evaluation and testing
- `evaluate_example.sh` - Example evaluation script

### **Utility Scripts**
- `download_checkpoint.py` - Download model checkpoints
- `run_distributed_local.sh` - Local distributed training

## 📊 **Analysis Scripts** (`scripts/analysis/`)

Diagnostic and analysis tools:

- `analyze_training.py` - Training metrics analysis
- `analyze_architecture.py` - Model architecture validation
- `analyze_training_config.py` - Hyperparameter analysis
- `analyze_data_pipeline.py` - Data pipeline validation
- `analyze_generation.py` - Generation pipeline analysis

**Usage:**
```bash
python scripts/analysis/analyze_training.py
```

## 🔧 **Utility Scripts** (`scripts/utilities/`)

Monitoring and testing tools:

- `monitor_training.py` - Real-time training monitoring
- `test_fixes.py` - Comprehensive fix testing
- `test_fixes_simple.py` - Simple fix verification

**Usage:**
```bash
python scripts/utilities/monitor_training.py
python scripts/utilities/test_fixes_simple.py
```

## 🗂️ **Legacy Scripts** (`scripts/legacy/`)

Old and deprecated scripts kept for reference:

- Old SLURM batch scripts
- Deprecated training scripts
- Temporary fix scripts
- Old generation scripts

**Note:** These are kept for reference but should not be used for new training.

## 🚀 **Recommended Usage**

### **For Training:**
Use the organized training configs in `training_configs/`:
```bash
# Cluster training
sbatch training_configs/2stage_XL/train_cluster.sh

# Dev GPU training
./training_configs/2stage_XL/train_dev.sh
```

### **For Analysis:**
```bash
# Run comprehensive analysis
python scripts/analysis/analyze_training.py

# Monitor training progress
python scripts/utilities/monitor_training.py
```

### **For Generation:**
```bash
# Use fixed generation script
python scripts/generate_fixed.py --model-path <checkpoint> --prompt "Hello world"
```

## 📝 **Notes**

- All scripts are organized by function and purpose
- Main training should use `training_configs/` directory
- Analysis scripts help diagnose training issues
- Legacy scripts are kept for reference only
- All scripts maintain their original functionality
