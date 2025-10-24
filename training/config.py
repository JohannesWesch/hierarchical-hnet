"""
Training configuration management.

This module provides configuration classes and utilities for managing
training hyperparameters and settings.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json


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
        pass
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        pass


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
        pass
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        pass


@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    
    config_path: str = ""
    dtype: str = "bfloat16"  # float32, float16, bfloat16
    initializer_range: float = 0.02
    tie_embeddings: bool = False
    
    def to_dict(self):
        """Convert to dictionary."""
        pass
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        pass


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
        pass
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        pass


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
        pass
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        pass


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
        pass
    
    def to_json(self, path: str):
        """Save configuration to JSON file."""
        pass
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create from dictionary."""
        pass
    
    @classmethod
    def from_json(cls, path: str):
        """Load configuration from JSON file."""
        pass
    
    def validate(self):
        """Validate configuration values."""
        pass

