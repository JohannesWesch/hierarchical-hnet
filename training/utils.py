"""
Training utilities for H-Net.

This module provides utilities for:
- Logging and monitoring
- Checkpointing
- Metrics computation
- Learning rate scheduling
"""

import glob
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from hnet.models.mixer_seq import HNetForCausalLM


def setup_logging(output_dir: str, rank: int = 0) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        output_dir: Directory for log files
        rank: Process rank (for distributed training)

    Returns:
        Configured logger instance
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger("hnet_training")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (only for rank 0)
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = os.path.join(output_dir, "train.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def cleanup_old_checkpoints(
    output_dir: str, keep_last_n: int = 3, logger: Optional[logging.Logger] = None
):
    """
    Clean up old checkpoints, keeping only the most recent ones.

    Args:
        output_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep (default: 3)
        logger: Logger instance for reporting cleanup actions
    """
    # Only cleanup on main process to avoid race conditions
    try:
        from training.distributed import is_main_process

        if not is_main_process():
            return
    except ImportError:
        pass  # Not in distributed mode

    # Find all checkpoint files
    checkpoint_pattern = os.path.join(output_dir, "checkpoint_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)

    if len(checkpoint_files) <= keep_last_n:
        return  # Nothing to clean up

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    # Delete old checkpoints
    files_to_delete = checkpoint_files[keep_last_n:]
    deleted_count = 0
    freed_space = 0

    for checkpoint_path in files_to_delete:
        try:
            file_size = os.path.getsize(checkpoint_path)
            os.remove(checkpoint_path)
            deleted_count += 1
            freed_space += file_size

            if logger:
                logger.info(f"Deleted old checkpoint: {os.path.basename(checkpoint_path)}")
        except OSError as e:
            if logger:
                logger.warning(f"Failed to delete {checkpoint_path}: {e}")

    if deleted_count > 0 and logger:
        freed_gb = freed_space / (1024**3)
        logger.info(
            f"Checkpoint cleanup: deleted {deleted_count} old checkpoint(s), "
            f"freed {freed_gb:.2f} GB"
        )


def save_checkpoint(
    model: HNetForCausalLM,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    step: int,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    keep_last_n: int = 3,
    logger: Optional[logging.Logger] = None,
    max_retries: int = 3,
):
    """
    Save training checkpoint and clean up old checkpoints with retry logic.

    Args:
        model: H-Net model (can be wrapped with DDP)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        step: Current training step
        output_dir: Directory to save checkpoint
        config: Model configuration
        metrics: Training metrics to save
        keep_last_n: Number of recent checkpoints to keep (default: 3)
        logger: Logger instance for reporting actions
        max_retries: Maximum number of retry attempts (default: 3)
    """
    # Only save on main process to avoid conflicts
    try:
        from training.distributed import is_main_process

        if not is_main_process():
            return
    except ImportError:
        pass  # Not in distributed mode

    os.makedirs(output_dir, exist_ok=True)

    # Handle DDP-wrapped models
    model_state_dict = model.state_dict()
    if hasattr(model, "module"):
        # Model is wrapped with DDP, get the unwrapped state dict
        model_state_dict = model.module.state_dict()

    # Create checkpoint dictionary
    checkpoint = {
        "step": step,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if config is not None:
        checkpoint["config"] = config

    # Save checkpoint with retry logic
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{step}.pt")

    for attempt in range(max_retries):
        try:
            # Use atomic write: save to temp file first, then rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=output_dir, prefix=f"checkpoint_{step}_", suffix=".pt.tmp"
            )
            os.close(temp_fd)  # Close the file descriptor, we'll use the path

            # Save to temporary file
            torch.save(checkpoint, temp_path)

            # Verify the file was written correctly by checking size
            if os.path.getsize(temp_path) == 0:
                raise RuntimeError("Checkpoint file is empty after write")

            # Atomic rename (overwrites if checkpoint already exists)
            os.replace(temp_path, checkpoint_path)

            if logger:
                checkpoint_size_gb = os.path.getsize(checkpoint_path) / (1024**3)
                logger.info(
                    f"Saved checkpoint to {checkpoint_path} " f"({checkpoint_size_gb:.2f} GB)"
                )

            # Success! Break out of retry loop
            break

        except (OSError, RuntimeError) as e:
            # Clean up temp file if it exists
            if "temp_path" in locals() and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                if logger:
                    logger.warning(
                        f"Checkpoint save attempt {attempt + 1}/{max_retries} failed: {e}"
                    )
                    logger.warning(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # Final attempt failed
                if logger:
                    logger.error(f"Failed to save checkpoint after {max_retries} attempts: {e}")
                    logger.error("Training will continue, but this checkpoint was not saved!")
                # Don't crash the training - just continue without this checkpoint
                return

    # Save config separately for easy access
    if config is not None:
        config_path = os.path.join(output_dir, "config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        except OSError as e:
            if logger:
                logger.warning(f"Failed to save config.json: {e}")

    # Clean up old checkpoints
    cleanup_old_checkpoints(output_dir, keep_last_n, logger)


def load_checkpoint(
    checkpoint_path: str,
    model: HNetForCausalLM,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: H-Net model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to

    Returns:
        Dictionary with checkpoint metadata (step, metrics, etc.)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Return metadata
    metadata = {
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
    }

    return metadata


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        logits: Model logits (B, L, vocab_size)
        targets: Target token ids (B, L)
        loss: Cross-entropy loss

    Returns:
        Dictionary of metrics (perplexity, accuracy, etc.)
    """
    with torch.no_grad():
        # Perplexity
        perplexity = torch.exp(loss).item()

        # Token accuracy
        predictions = torch.argmax(logits, dim=-1)
        mask = targets != -100
        correct = (predictions == targets) & mask
        accuracy = (correct.sum().float() / mask.sum().float()).item()

        return {
            "perplexity": perplexity,
            "accuracy": accuracy,
        }


def get_lr_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_type: str = "cosine",
) -> LambdaLR:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        scheduler_type: Type of scheduler ('cosine', 'linear', 'constant')

    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif scheduler_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif scheduler_type == "constant":
        # Constant LR after warmup
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create cosine learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles
        last_epoch: Last epoch for resuming

    Returns:
        Cosine scheduler with warmup
    """
    import math

    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create linear learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        last_epoch: Last epoch for resuming

    Returns:
        Linear scheduler with warmup
    """

    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Linear decay phase
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def log_training_stats(
    step: int,
    loss: float,
    metrics: Dict[str, float],
    learning_rates: Dict[str, float],
    step_time: float,
    logger: Optional[logging.Logger] = None,
):
    """
    Log training statistics.

    Args:
        step: Current training step
        loss: Training loss
        metrics: Dictionary of metrics
        learning_rates: Learning rates for each parameter group
        step_time: Time taken for the step
        logger: Logger instance
    """
    # Format learning rates
    lr_str = ", ".join([f"{k}: {v:.2e}" for k, v in learning_rates.items()])

    # Format metrics
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])

    # Calculate tokens per second if available
    tokens_per_sec = metrics.get("tokens_per_sec", 0)

    # Create log message
    log_msg = (
        f"Step {step} | "
        f"Loss: {loss:.4f} | "
        f"{metrics_str} | "
        f"LR: {lr_str} | "
        f"Time: {step_time:.3f}s"
    )

    if tokens_per_sec > 0:
        log_msg += f" | Tokens/s: {tokens_per_sec:.0f}"

    if logger is not None:
        logger.info(log_msg)
    else:
        print(log_msg)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics over multiple steps.
    """

    def __init__(self, name: str, fmt: str = ":f"):
        """
        Initialize average meter.

        Args:
            name: Name of the metric
            fmt: Format string for displaying values
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics with new value.

        Args:
            val: New value
            n: Weight/count for the value
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """String representation of current statistics."""
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressTracker:
    """
    Tracks training progress and estimates remaining time.
    """

    def __init__(self, total_steps: int):
        """
        Initialize progress tracker.

        Args:
            total_steps: Total number of training steps
        """
        self.total_steps = total_steps
        self.start_time = time.time()
        self.current_step = 0

    def update(self, step: int):
        """
        Update progress.

        Args:
            step: Current step number
        """
        self.current_step = step

    def get_eta(self) -> str:
        """
        Get estimated time remaining.

        Returns:
            Formatted ETA string
        """
        if self.current_step == 0:
            return "N/A"

        elapsed = time.time() - self.start_time
        steps_per_sec = self.current_step / elapsed
        remaining_steps = self.total_steps - self.current_step
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

        return format_time(eta_seconds)

    def get_progress_str(self) -> str:
        """
        Get progress string.

        Returns:
            Formatted progress string
        """
        percent = 100.0 * self.current_step / self.total_steps
        return f"{self.current_step}/{self.total_steps} ({percent:.1f}%) | ETA: {self.get_eta()}"


def compute_num_params(model: nn.Module) -> Dict[str, int]:
    """
    Compute number of parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with total, trainable, and non-trainable params
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {mins}m {secs}s"


def get_grad_norm(model: nn.Module) -> float:
    """
    Compute gradient norm across all parameters.

    Args:
        model: PyTorch model

    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm
