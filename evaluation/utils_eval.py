"""
Shared utilities for H-Net model evaluation.

This module provides common functions for loading models and handling checkpoints.
"""

import json
from typing import Optional

import torch
from omegaconf import ListConfig

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.models.mixer_seq import HNetForCausalLM


def load_model_for_eval(
    model_path: str,
    config_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> HNetForCausalLM:
    """
    Load H-Net model from checkpoint for evaluation.

    Args:
        model_path: Path to the model checkpoint (.pt file)
        config_path: Path to the model configuration (.json file)
        device: Device to load model on (default: cuda)
        dtype: Model dtype (default: bfloat16)

    Returns:
        Loaded HNetForCausalLM model in eval mode
    """
    # Load configuration
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Create config objects
    attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    # Create model
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=dtype)
    model.eval()

    # Load checkpoint with proper handling for different PyTorch versions
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 6):
        with torch.serialization.safe_globals([ListConfig]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    else:
        checkpoint = torch.load(model_path, map_location=device)

    # Handle both training checkpoints and standalone model weights
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Training checkpoint format
        state_dict = checkpoint["model_state_dict"]
        step = checkpoint.get("step", "unknown")
        print(f"Loaded training checkpoint from step {step}")
    else:
        # Standalone model weights
        state_dict = checkpoint
        print("Loaded standalone model weights")

    model.load_state_dict(state_dict)

    return model


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Get torch device, defaulting to CUDA if available.

    Args:
        device_str: Device string (cuda/cpu) or None for auto-detection

    Returns:
        torch.device object
    """
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_str)

    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    return device
