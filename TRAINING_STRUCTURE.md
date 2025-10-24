# H-Net Training Infrastructure Structure

This document outlines the complete folder structure and components of the H-Net training infrastructure.

## Directory Structure

```
hierarchical-hnet/
├── configs/
│   ├── hnet_1stage_L.json              # Existing model configs
│   ├── hnet_2stage_L.json
│   ├── hnet_2stage_XL.json
│   └── training_config_example.json     # NEW: Example training config
│
├── scripts/
│   ├── download_checkpoint.py           # Existing
│   ├── train_example.sh                 # NEW: Single-GPU training example
│   ├── train_distributed.sh             # NEW: Multi-GPU training example
│   └── evaluate_example.sh              # NEW: Evaluation example
│
├── hnet/                                 # Existing model implementation
│   ├── models/
│   │   ├── hnet.py                      # Core H-Net architecture
│   │   ├── mixer_seq.py                 # HNetForCausalLM wrapper
│   │   └── config_hnet.py               # Model configuration
│   ├── modules/
│   │   ├── dc.py                        # Dynamic chunking/routing
│   │   ├── isotropic.py                 # Non-hierarchical components
│   │   └── ...
│   └── utils/
│       ├── tokenizers.py                # ByteTokenizer
│       └── train.py                     # Basic training utilities (load_balancing_loss, group_params)
│
├── training/                             # NEW: Training infrastructure
│   ├── __init__.py                      # Package initialization
│   ├── data.py                          # Data loading and collation
│   ├── losses.py                        # Loss functions
│   ├── metrics.py                       # Evaluation metrics
│   ├── utils.py                         # Training utilities
│   ├── config.py                        # Configuration classes
│   └── distributed.py                   # Distributed training support
│
├── train.py                              # NEW: Main training script
├── evaluate.py                           # NEW: Evaluation script
├── generate.py                           # Existing generation script
├── README.md                             # Existing README
└── README_TRAINING.md                    # NEW: Training guide
```

## Component Overview

### Core Training Scripts

#### `train.py`
Main training script with:
- Argument parsing for all hyperparameters
- Model initialization with weight initialization
- Learning rate multiplier application
- Data loading setup
- Optimizer and scheduler configuration
- Main training loop
- Checkpointing and evaluation

**Key Functions:**
- `parse_args()`: Parse command-line arguments
- `initialize_model()`: Create and initialize H-Net model
- `setup_optimizer()`: Create optimizer with grouped parameters
- `compute_loss()`: Calculate total loss (CE + load balancing)
- `train_step()`: Single training iteration
- `evaluate()`: Validation evaluation
- `train()`: Main training loop
- `main()`: Entry point

#### `evaluate.py`
Evaluation script with:
- Model loading from checkpoint
- Batch evaluation on test data
- Comprehensive metrics computation
- Results saving to JSON

**Key Functions:**
- `load_model()`: Load model from checkpoint
- `evaluate_model()`: Run evaluation loop
- `compute_detailed_metrics()`: Calculate all metrics
- `save_results()`: Save results to file

### Training Package (`training/`)

#### `data.py` - Data Loading
Classes for efficient data loading:

**PackedDataset**
- Loads text data
- Tokenizes with ByteTokenizer
- Creates packed sequences

**PackedDataCollator**
- Collates batches with cu_seqlens
- Handles variable-length sequences
- Supports packed format for flash-attention

**TextDataset**
- Simple line-by-line text loading
- Document-level processing

**StreamingDataset**
- Memory-efficient streaming
- For large-scale datasets

#### `losses.py` - Loss Functions
Loss computation classes:

**LanguageModelingLoss**
- Cross-entropy loss for next-token prediction
- Ignores padding tokens

**LoadBalancingLoss**
- Balances routing decisions
- Encourages even chunk distribution

**HierarchicalLoadBalancingLoss**
- Multi-stage load balancing
- Per-stage weighting

**CombinedLoss**
- Total loss = CE + λ × LB
- Returns detailed loss breakdown

**Utility Functions:**
- `compute_perplexity()`: Calculate perplexity from loss
- `compute_token_accuracy()`: Token-level accuracy
- `get_routing_statistics()`: Routing behavior analysis

#### `metrics.py` - Evaluation Metrics
Metrics computation classes:

**MetricsComputer**
- Perplexity
- Token accuracy
- Top-k accuracy

**RoutingMetrics**
- Boundary prediction rates
- Chunk size statistics
- Load balance metrics

**ThroughputMetrics**
- Tokens per second
- Samples per second
- FLOPS utilization

**MetricsTracker**
- Tracks metrics over time
- Computes averages and statistics

#### `utils.py` - Training Utilities
Helper functions for training:

**Logging:**
- `setup_logging()`: Configure logging
- `log_training_stats()`: Log training metrics

**Checkpointing:**
- `save_checkpoint()`: Save model + optimizer state
- `load_checkpoint()`: Load checkpoint for resuming

**Scheduling:**
- `get_lr_scheduler()`: Create LR scheduler
- `get_cosine_schedule_with_warmup()`: Cosine decay
- `get_linear_schedule_with_warmup()`: Linear decay

**Monitoring:**
- `AverageMeter`: Track running averages
- `ProgressTracker`: Estimate ETA
- `compute_num_params()`: Count parameters
- `get_grad_norm()`: Compute gradient norm

**Metrics:**
- `compute_metrics()`: Calculate evaluation metrics
- `format_time()`: Format time strings

#### `config.py` - Configuration Management
Configuration dataclasses:

**DataConfig**
- Data paths and loading settings

**OptimizationConfig**
- Learning rates, schedulers, loss weights

**ModelConfig**
- Model initialization settings

**CheckpointConfig**
- Saving and evaluation intervals

**SystemConfig**
- Device, distributed settings

**TrainingConfig**
- Complete training configuration
- JSON serialization support

#### `distributed.py` - Distributed Training
Distributed training utilities:

**Setup:**
- `setup_distributed()`: Initialize distributed environment
- `cleanup_distributed()`: Cleanup
- `setup_ddp_model()`: Wrap model with DDP

**Communication:**
- `all_reduce()`: Reduce tensors across processes
- `all_gather()`: Gather tensors
- `broadcast()`: Broadcast from source
- `barrier()`: Synchronization barrier

**Utilities:**
- `get_rank()`, `get_world_size()`: Process info
- `is_main_process()`: Check if rank 0
- `reduce_dict()`: Reduce metric dictionaries
- `DistributedSampler`: Data sampling for distributed

### Example Scripts (`scripts/`)

#### `train_example.sh`
Single-GPU training example with common hyperparameters

#### `train_distributed.sh`
Multi-GPU training with `torchrun`

#### `evaluate_example.sh`
Model evaluation example

## Usage Patterns

### 1. Basic Training
```bash
python train.py \
    --config-path configs/hnet_2stage_L.json \
    --train-data-path /data/train \
    --output-dir ./outputs/my_model
```

### 2. Custom Hyperparameters
```bash
python train.py \
    --config-path configs/hnet_2stage_L.json \
    --train-data-path /data/train \
    --learning-rate 5e-4 \
    --lr-multipliers "4.0,2.0,1.0" \
    --load-balancing-weight 0.02
```

### 3. Resume Training
```bash
python train.py \
    --resume-from ./outputs/my_model/checkpoint_10000.pt \
    --train-data-path /data/train
```

### 4. Distributed Training
```bash
torchrun --nproc_per_node=8 train.py \
    --config-path configs/hnet_2stage_L.json \
    --train-data-path /data/train
```

### 5. Evaluation
```bash
python evaluate.py \
    --model-path ./outputs/my_model/checkpoint_final.pt \
    --config-path configs/hnet_2stage_L.json \
    --data-path /data/test
```

## Key Design Decisions

### 1. Learning Rate Multipliers
- Hierarchical models need per-stage learning rates
- Applied via parameter groups in optimizer
- Outer stages (high-level) get higher LR
- Inner stages (low-level) get lower LR

### 2. Load Balancing Loss
- Critical for dynamic chunking performance
- Encourages balanced routing decisions
- Weighted sum across hierarchy stages
- Typical weight: 0.01-0.1

### 3. Packed Sequences
- Efficient GPU utilization
- Uses cu_seqlens for variable lengths
- Compatible with flash-attention
- No padding waste

### 4. Checkpointing
- Saves model, optimizer, scheduler
- Enables seamless resuming
- Keeps last N checkpoints
- Includes configuration and metrics

### 5. Modular Design
- Separate concerns (data, losses, metrics)
- Easy to extend and customize
- Reusable components
- Clean interfaces

## Next Steps for Implementation

The current implementation provides the complete structure with all methods defined but using `pass` statements. To make it functional:

1. **Implement data loading** (`training/data.py`)
   - Complete `PackedDataset.__getitem__()`
   - Implement `PackedDataCollator.__call__()`
   - Add data loading logic

2. **Implement loss functions** (`training/losses.py`)
   - Complete `LanguageModelingLoss.__call__()`
   - Implement `LoadBalancingLoss.__call__()`
   - Use `hnet.utils.train.load_balancing_loss()`

3. **Implement training utilities** (`training/utils.py`)
   - Complete checkpoint save/load
   - Implement LR schedulers
   - Add logging functions

4. **Implement training loop** (`train.py`)
   - Complete `train_step()`
   - Implement `train()` main loop
   - Add evaluation logic

5. **Add configuration** (`training/config.py`)
   - Implement config serialization
   - Add validation logic

6. **Test and debug**
   - Unit tests for components
   - Integration testing
   - Small-scale training run

## Documentation

See `README_TRAINING.md` for:
- Quick start guide
- Detailed hyperparameter explanations
- Troubleshooting tips
- Best practices

