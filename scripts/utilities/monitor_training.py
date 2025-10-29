#!/usr/bin/env python3
"""
Training monitoring script for H-Net.

This script monitors training progress and alerts on issues.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt


def monitor_training_log(log_path="outputs/hnet_2stage_XL_fixed/train.log"):
    """Monitor training log for issues."""

    print("Monitoring training log...")
    print(f"Log file: {log_path}")

    if not Path(log_path).exists():
        print(f"Log file not found: {log_path}")
        return

    # Patterns to monitor
    patterns = {
        "gradient_explosion": r"grad_norm: ([\d.]+)",
        "loss_spike": r"Loss: ([\d.]+)",
        "learning_rate": r"LR: group_0: ([\d.e-]+)",
    }

    # Monitor for issues
    with open(log_path, "r") as f:
        for line in f:
            # Check for gradient explosion
            if "grad_norm:" in line:
                match = re.search(patterns["gradient_explosion"], line)
                if match:
                    grad_norm = float(match.group(1))
                    if grad_norm > 10.0:
                        print(f"⚠️  WARNING: High gradient norm detected: {grad_norm}")
                    elif grad_norm > 5.0:
                        print(f"⚠️  CAUTION: Elevated gradient norm: {grad_norm}")

    print("Monitoring complete")


def plot_training_curves(log_path="outputs/hnet_2stage_XL_fixed/train.log"):
    """Plot training curves."""

    print("Plotting training curves...")

    if not Path(log_path).exists():
        print(f"Log file not found: {log_path}")
        return

    # Parse log file
    steps = []
    losses = []
    grad_norms = []

    with open(log_path, "r") as f:
        for line in f:
            if "Step" in line and "Loss:" in line:
                # Extract step, loss, and grad_norm
                step_match = re.search(r"Step (\d+)", line)
                loss_match = re.search(r"Loss: ([\d.]+)", line)
                grad_match = re.search(r"grad_norm: ([\d.]+)", line)

                if step_match and loss_match and grad_match:
                    steps.append(int(step_match.group(1)))
                    losses.append(float(loss_match.group(1)))
                    grad_norms.append(float(grad_match.group(1)))

    if not steps:
        print("No training data found in log")
        return

    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Loss curve
    ax1.plot(steps, losses, "b-", alpha=0.7)
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    # Gradient norm curve
    ax2.plot(steps, grad_norms, "r-", alpha=0.7)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title("Gradient Norm")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_monitor.png", dpi=300, bbox_inches="tight")
    print("✓ Training curves saved to training_monitor.png")


if __name__ == "__main__":
    monitor_training_log()
    plot_training_curves()
