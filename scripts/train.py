"""
Training script for H-Net language models.

This script provides the main training loop for H-Net models, including:
- Model initialization and configuration
- Data loading with packed sequences
- Loss computation (cross-entropy + load balancing)
- Optimizer configuration with per-stage learning rate multipliers
- Checkpointing and evaluation
"""

import argparse
import json
import os
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.utils.train import group_params
from training.data import HuggingFaceStreamingDataset, PackedDataCollator
from training.utils import (
    get_lr_scheduler,
    log_training_stats,
    save_checkpoint,
    setup_logging,
)


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train H-Net language model")

    # Model configuration
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to model configuration JSON file",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Data configuration
    parser.add_argument(
        "--train-data-path",
        type=str,
        default=None,
        help="Path to training data (optional, uses FineWeb-Edu by default)",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        default=None,
        help="Path to validation data (optional)",
    )
    parser.add_argument(
        "--use-hf-dataset",
        action="store_true",
        default=True,
        help="Use HuggingFace FineWeb-Edu dataset (default: True)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Base learning rate",
    )
    parser.add_argument(
        "--lr-multipliers",
        type=str,
        default="3.0,1.7,0.9",
        help="Comma-separated learning rate multipliers per stage",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="Weight decay coefficient",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--load-balancing-weight",
        type=float,
        default=0.01,
        help="Weight for load balancing loss",
    )
    parser.add_argument(
        "--num-training-steps",
        type=int,
        default=100000,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Number of warmup steps",
    )

    # Optimization settings
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--initializer-range",
        type=float,
        default=0.02,
        help="Standard deviation for weight initialization",
    )

    # Checkpointing and logging
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory for saving checkpoints and logs",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log training stats every N steps",
    )

    # System settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_config(config_path: str) -> HNetConfig:
    """Load model configuration from JSON file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Extract sub-configs
    attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))

    # Create main config
    hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    return hnet_cfg


def initialize_model(
    config: HNetConfig,
    device: torch.device,
    dtype: torch.dtype,
    initializer_range: float,
    lr_multipliers: list[float],
) -> HNetForCausalLM:
    """Initialize H-Net model with proper weight initialization."""
    # Create model
    model = HNetForCausalLM(config, device=device, dtype=dtype)

    # Initialize weights
    model.init_weights(initializer_range)

    # Apply learning rate multipliers
    model.apply_lr_multiplier(lr_multipliers)

    return model


def setup_optimizer(
    model: HNetForCausalLM,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Setup optimizer with grouped parameters."""
    # Group parameters with lr_multipliers (already applied via apply_lr_multiplier)
    param_groups = group_params(model)

    # The param_groups already have lr_multiplier and weight_decay set
    # We need to multiply the base learning_rate by the lr_multiplier for each group
    for group in param_groups:
        # Get the lr_multiplier (if it exists)
        lr_mult = group.get("lr_multiplier", 1.0)
        # Set the actual learning rate
        group["lr"] = learning_rate * lr_mult

    # Create optimizer (don't pass lr here since we set it per group)
    optimizer = AdamW(param_groups, betas=(0.9, 0.95))

    return optimizer


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bpred_outputs: list,
    load_balancing_weight: float,
) -> Dict[str, torch.Tensor]:
    """Compute total loss including cross-entropy and load balancing losses."""
    from training.losses import (
        CombinedLoss,
        HierarchicalLoadBalancingLoss,
        LanguageModelingLoss,
    )

    # Create loss functions
    lm_loss = LanguageModelingLoss(ignore_index=-100)
    lb_loss = HierarchicalLoadBalancingLoss()
    combined_loss = CombinedLoss(lm_loss, lb_loss, lb_weight=load_balancing_weight)

    # Compute losses
    loss_dict = combined_loss(logits, targets, bpred_outputs)

    return loss_dict


def train_step(
    model: HNetForCausalLM,
    batch: Dict[str, torch.Tensor],
    optimizer: AdamW,
    scheduler: LambdaLR,
    max_grad_norm: float,
    load_balancing_weight: float,
    gradient_accumulation_steps: int,
    step: int,
) -> Dict[str, float]:
    """Perform a single training step."""
    model.train()

    # Get batch data
    input_ids = batch["input_ids"]
    targets = batch["targets"]
    batch.get("cu_seqlens", None)
    batch.get("max_seqlen", None)

    # For packed mode, reshape to (1, total_tokens) and use mask=None
    # The model will use cu_seqlens and max_seqlen from mixer_kwargs
    input_ids_2d = input_ids.unsqueeze(0)  # (total_tokens,) -> (1, total_tokens)
    targets_2d = targets.unsqueeze(0)  # (total_tokens,) -> (1, total_tokens)

    # Forward pass
    output = model(
        input_ids_2d,
        mask=None,  # Packed mode
    )

    logits = output.logits
    bpred_outputs = output.bpred_output

    # Compute loss
    loss_dict = compute_loss(logits, targets_2d, bpred_outputs, load_balancing_weight)
    loss = loss_dict["loss"]

    # Backward pass (with gradient accumulation)
    (loss / gradient_accumulation_steps).backward()

    # Return metrics
    metrics = {
        "loss": loss.item(),
        "ce_loss": loss_dict["ce_loss"].item(),
        "lb_loss": loss_dict["lb_loss"].item(),
    }

    # Add per-stage LB losses
    for stage_name, stage_loss in loss_dict["lb_losses_per_stage"].items():
        metrics[f"lb_{stage_name}"] = stage_loss.item()

    # Optimizer step every gradient_accumulation_steps
    if (step + 1) % gradient_accumulation_steps == 0:
        # Compute gradient norm BEFORE clipping
        from training.utils import get_grad_norm

        metrics["grad_norm"] = get_grad_norm(model)

        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return metrics


def evaluate(
    model: HNetForCausalLM,
    val_dataloader: DataLoader,
    device: torch.device,
    load_balancing_weight: float,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()

    from training.utils import AverageMeter

    loss_meter = AverageMeter("loss")
    ce_loss_meter = AverageMeter("ce_loss")
    lb_loss_meter = AverageMeter("lb_loss")

    with torch.no_grad():
        for batch in val_dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            # Reshape for packed mode
            input_ids_2d = input_ids.unsqueeze(0)  # (total_tokens,) -> (1, total_tokens)
            targets_2d = targets.unsqueeze(0)

            # Forward pass
            output = model(
                input_ids_2d,
                mask=None,  # Packed mode
            )

            logits = output.logits
            bpred_outputs = output.bpred_output

            # Compute loss
            loss_dict = compute_loss(logits, targets_2d, bpred_outputs, load_balancing_weight)

            # Update meters
            loss_meter.update(loss_dict["loss"].item())
            ce_loss_meter.update(loss_dict["ce_loss"].item())
            lb_loss_meter.update(loss_dict["lb_loss"].item())

    model.train()

    return {
        "val_loss": loss_meter.avg,
        "val_ce_loss": ce_loss_meter.avg,
        "val_lb_loss": lb_loss_meter.avg,
        "val_perplexity": torch.exp(torch.tensor(ce_loss_meter.avg)).item(),
    }


def train(
    model: HNetForCausalLM,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    optimizer: AdamW,
    scheduler: LambdaLR,
    args: argparse.Namespace,
    device: torch.device,
    start_step: int = 0,
):
    """Main training loop."""
    from training.utils import (
        AverageMeter,
        ProgressTracker,
    )

    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting training for {args.num_training_steps} steps")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training meters
    loss_meter = AverageMeter("loss", ":.4f")
    progress_tracker = ProgressTracker(args.num_training_steps)

    # Training loop
    step = start_step
    epoch = 0
    model.train()

    try:
        while step < args.num_training_steps:
            epoch += 1

            for batch in train_dataloader:
                if step >= args.num_training_steps:
                    break

                # Move batch to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

                # Training step
                import time

                step_start = time.time()

                metrics = train_step(
                    model,
                    batch,
                    optimizer,
                    scheduler,
                    args.max_grad_norm,
                    args.load_balancing_weight,
                    args.gradient_accumulation_steps,
                    step,
                )

                step_time = time.time() - step_start

                # Update meters
                loss_meter.update(metrics["loss"])
                progress_tracker.update(step)

                # Logging
                if (step + 1) % args.log_interval == 0:
                    # Get learning rates
                    lrs = {f"group_{i}": pg["lr"] for i, pg in enumerate(optimizer.param_groups)}

                    log_training_stats(step + 1, metrics["loss"], metrics, lrs, step_time, logger)

                # Checkpointing
                if (step + 1) % args.save_interval == 0:
                    logger.info(f"Saving checkpoint at step {step + 1}")
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        step + 1,
                        args.output_dir,
                        metrics=metrics,
                    )

                # Evaluation
                if val_dataloader is not None and (step + 1) % args.eval_interval == 0:
                    logger.info(f"Evaluating at step {step + 1}")
                    val_metrics = evaluate(
                        model, val_dataloader, device, args.load_balancing_weight
                    )
                    logger.info(f"Validation metrics: {val_metrics}")

                step += 1

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # Save final checkpoint
    logger.info("Saving final checkpoint")
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        step,
        args.output_dir,
        metrics={"final": True},
    )

    logger.info(f"Training complete! Total steps: {step}")


def main():
    """Main entry point for training script."""
    # Parse arguments
    args = parse_args()

    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"Using device: {device}, dtype: {dtype}")

    # Load configuration
    print(f"Loading model config from {args.config_path}")
    config = load_model_config(args.config_path)

    # Parse LR multipliers
    lr_multipliers = [float(x) for x in args.lr_multipliers.split(",")]
    print(f"LR multipliers: {lr_multipliers}")

    # Initialize model
    print("Initializing model...")
    model = initialize_model(
        config,
        device,
        dtype,
        args.initializer_range,
        lr_multipliers,
    )
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Setup data
    print("Setting up data loaders...")
    from training.data import TextDataset

    # Load FineWeb-Edu dataset from HuggingFace
    print("Loading FineWeb-Edu dataset from HuggingFace...")
    from datasets import load_dataset

    # Load dataset with proper train/val split (no data leakage)
    print("Loading HuggingFace dataset with train/val split...")
    print("  - Validation: first 1000 examples")
    print("  - Training: remaining examples (skipping first 1000)")

    base_hf_dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True
    )

    # Training data: skip first 1000 examples (reserved for validation)
    train_hf_dataset = base_hf_dataset.skip(1000)

    # Wrap in our dataset class
    train_dataset = HuggingFaceStreamingDataset(
        train_hf_dataset,
        text_column="text",
        add_bos=True,
        max_length=args.max_seq_length,
    )

    train_collator = PackedDataCollator(max_seq_length=args.max_seq_length)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_collator,
        num_workers=0,  # HF streaming datasets don't work well with multiple workers
    )

    # Validation data (use a small sample from the training set or separate validation set)
    val_dataloader = None
    if args.val_data_path:
        # If a specific validation path is provided, use it
        val_dataset = TextDataset(
            args.val_data_path,
            max_length=args.max_seq_length,
            add_bos=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=train_collator,
            num_workers=0,
        )
    else:
        # Use first 1000 examples for validation (training skips these)
        print("Creating validation set from first 1000 examples...")
        val_base_dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True
        )
        val_hf_dataset = val_base_dataset.take(1000)

        val_dataset = HuggingFaceStreamingDataset(
            val_hf_dataset,
            text_column="text",
            add_bos=True,
            max_length=args.max_seq_length,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=train_collator,
            num_workers=0,
        )

    # Setup optimizer and scheduler
    print("Setting up optimizer and scheduler...")
    optimizer = setup_optimizer(model, args.learning_rate, args.weight_decay)

    scheduler = get_lr_scheduler(
        optimizer,
        args.warmup_steps,
        args.num_training_steps,
        scheduler_type="cosine",
    )

    # Resume from checkpoint if provided
    start_step = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        from training.utils import load_checkpoint

        metadata = load_checkpoint(args.resume_from, model, optimizer, scheduler, device)
        start_step = metadata["step"]
        print(f"Resumed from step {start_step}")

    # Train
    print("Starting training...")
    train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        args,
        device,
        start_step,
    )


if __name__ == "__main__":
    main()
