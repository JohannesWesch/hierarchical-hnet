"""
Training configuration management.

This module provides configuration classes and utilities for managing
training hyperparameters and settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Configuration for data loading."""

    train_data_path: str = ""
    val_data_path: Optional[str] = None
    max_seq_length: int = 2048
    batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "train_data_path": self.train_data_path,
            "val_data_path": self.val_data_path,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": self.pin_memory,
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""

    learning_rate: float = 3e-4
    lr_multipliers: List[float] = field(default_factory=lambda: [3.0, 1.7, 0.9])
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Scheduler settings
    num_training_steps: int = 100000
    num_warmup_steps: int = 2000
    scheduler_type: str = "cosine"  # cosine, linear, constant

    # Loss weights
    load_balancing_weight: float = 0.01

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "lr_multipliers": self.lr_multipliers,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_training_steps": self.num_training_steps,
            "num_warmup_steps": self.num_warmup_steps,
            "scheduler_type": self.scheduler_type,
            "load_balancing_weight": self.load_balancing_weight,
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class ModelConfig:
    """Configuration for model initialization."""

    config_path: str = ""
    dtype: str = "bfloat16"  # float32, float16, bfloat16
    initializer_range: float = 0.02
    tie_embeddings: bool = False

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "config_path": self.config_path,
            "dtype": self.dtype,
            "initializer_range": self.initializer_range,
            "tie_embeddings": self.tie_embeddings,
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""

    output_dir: str = "./outputs"
    save_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 10
    keep_last_n_checkpoints: int = 3
    save_optimizer_state: bool = True

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "output_dir": self.output_dir,
            "save_interval": self.save_interval,
            "eval_interval": self.eval_interval,
            "log_interval": self.log_interval,
            "keep_last_n_checkpoints": self.keep_last_n_checkpoints,
            "save_optimizer_state": self.save_optimizer_state,
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class SystemConfig:
    """Configuration for system settings."""

    seed: int = 42
    device: str = "cuda"
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "seed": self.seed,
            "device": self.device,
            "distributed": self.distributed,
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "data": self.data.to_dict(),
            "optimization": self.optimization.to_dict(),
            "model": self.model.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "system": self.system.to_dict(),
        }

    def to_json(self, path: str):
        """Save configuration to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        return cls(
            data=DataConfig.from_dict(config_dict.get("data", {})),
            optimization=OptimizationConfig.from_dict(config_dict.get("optimization", {})),
            model=ModelConfig.from_dict(config_dict.get("model", {})),
            checkpoint=CheckpointConfig.from_dict(config_dict.get("checkpoint", {})),
            system=SystemConfig.from_dict(config_dict.get("system", {})),
        )

    @classmethod
    def from_json(cls, path: str):
        """Load configuration from JSON file."""
        import json

        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def validate(self):
        """Validate configuration values."""
        # Basic validation - can be extended
        assert self.data.batch_size > 0, "Batch size must be positive"
        assert self.optimization.learning_rate > 0, "Learning rate must be positive"
        assert self.optimization.num_training_steps > 0, "Number of training steps must be positive"
