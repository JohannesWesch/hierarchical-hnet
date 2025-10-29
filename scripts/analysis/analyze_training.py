#!/usr/bin/env python3
"""
Training metrics analysis script for H-Net diagnostic review.

This script parses the training log and analyzes:
1. Loss curves (total, CE, load balancing)
2. Learning rate schedules
3. Gradient norms
4. Perplexity trends
5. Load balancing per stage
"""

import argparse
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_training_log(log_path: str) -> Dict[str, List[float]]:
    """Parse training log and extract metrics."""

    # Pattern to match training step lines
    step_pattern = r"Step (\d+) \| Loss: ([\d.]+) \| loss: ([\d.]+), ce_loss: ([\d.]+), lb_loss: ([\d.]+), lb_stage_0: ([\d.]+), lb_stage_1: ([\d.]+), grad_norm: ([\d.]+) \| LR: group_0: ([\d.e-]+), group_1: ([\d.e-]+), group_2: ([\d.e-]+), group_3: ([\d.e-]+), group_4: ([\d.e-]+)"

    metrics = {
        "steps": [],
        "total_loss": [],
        "ce_loss": [],
        "lb_loss": [],
        "lb_stage_0": [],
        "lb_stage_1": [],
        "grad_norm": [],
        "lr_group_0": [],
        "lr_group_1": [],
        "lr_group_2": [],
        "lr_group_3": [],
        "lr_group_4": [],
        "perplexity": [],
    }

    with open(log_path, "r") as f:
        for line in f:
            match = re.search(step_pattern, line)
            if match:
                step = int(match.group(1))
                total_loss = float(match.group(2))
                ce_loss = float(match.group(4))
                lb_loss = float(match.group(5))
                lb_stage_0 = float(match.group(6))
                lb_stage_1 = float(match.group(7))
                grad_norm = float(match.group(8))
                lr_group_0 = float(match.group(9))
                lr_group_1 = float(match.group(10))
                lr_group_2 = float(match.group(11))
                lr_group_3 = float(match.group(12))
                lr_group_4 = float(match.group(13))

                metrics["steps"].append(step)
                metrics["total_loss"].append(total_loss)
                metrics["ce_loss"].append(ce_loss)
                metrics["lb_loss"].append(lb_loss)
                metrics["lb_stage_0"].append(lb_stage_0)
                metrics["lb_stage_1"].append(lb_stage_1)
                metrics["grad_norm"].append(grad_norm)
                metrics["lr_group_0"].append(lr_group_0)
                metrics["lr_group_1"].append(lr_group_1)
                metrics["lr_group_2"].append(lr_group_2)
                metrics["lr_group_3"].append(lr_group_3)
                metrics["lr_group_4"].append(lr_group_4)
                metrics["perplexity"].append(np.exp(ce_loss))

    return metrics


def plot_loss_curves(metrics: Dict[str, List[float]], save_path: str = None):
    """Plot loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Total loss
    axes[0, 0].plot(metrics["steps"], metrics["total_loss"], "b-", label="Total Loss", alpha=0.7)
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss Over Time")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # CE Loss and Perplexity
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()

    line1 = ax1.plot(metrics["steps"], metrics["ce_loss"], "g-", label="CE Loss", alpha=0.7)
    line2 = ax2.plot(metrics["steps"], metrics["perplexity"], "r-", label="Perplexity", alpha=0.7)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("CE Loss", color="g")
    ax2.set_ylabel("Perplexity", color="r")
    ax1.set_title("Cross-Entropy Loss and Perplexity")
    ax1.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    # Load Balancing Loss
    axes[1, 0].plot(metrics["steps"], metrics["lb_loss"], "m-", label="Total LB Loss", alpha=0.7)
    axes[1, 0].plot(metrics["steps"], metrics["lb_stage_0"], "c-", label="Stage 0 LB", alpha=0.7)
    axes[1, 0].plot(metrics["steps"], metrics["lb_stage_1"], "y-", label="Stage 1 LB", alpha=0.7)
    axes[1, 0].axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Target (1.0)")
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel("Load Balancing Loss")
    axes[1, 0].set_title("Load Balancing Loss by Stage")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Gradient Norm
    axes[1, 1].plot(metrics["steps"], metrics["grad_norm"], "orange", alpha=0.7)
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 1].set_ylabel("Gradient Norm")
    axes[1, 1].set_title("Gradient Norm Over Time")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss curves saved to {save_path}")
    else:
        plt.show()


def plot_learning_rates(metrics: Dict[str, List[float]], save_path: str = None):
    """Plot learning rate schedules."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(metrics["steps"], metrics["lr_group_0"], "b-", label="Group 0 (Stage 0)", alpha=0.7)
    ax.plot(metrics["steps"], metrics["lr_group_1"], "g-", label="Group 1 (Stage 0)", alpha=0.7)
    ax.plot(metrics["steps"], metrics["lr_group_2"], "r-", label="Group 2 (Stage 1)", alpha=0.7)
    ax.plot(metrics["steps"], metrics["lr_group_3"], "c-", label="Group 3 (Stage 1)", alpha=0.7)
    ax.plot(metrics["steps"], metrics["lr_group_4"], "m-", label="Group 4 (Stage 2)", alpha=0.7)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule by Parameter Group")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Learning rate plot saved to {save_path}")
    else:
        plt.show()


def analyze_metrics(metrics: Dict[str, List[float]]):
    """Analyze and print key metrics."""
    print("=== TRAINING METRICS ANALYSIS ===\n")

    # Basic stats
    print(f"Total training steps: {len(metrics['steps'])}")
    print(f"Final step: {metrics['steps'][-1]}")
    print(f"Training duration: {metrics['steps'][-1] - metrics['steps'][0]} steps\n")

    # Loss analysis
    print("=== LOSS ANALYSIS ===")
    print(f"Initial total loss: {metrics['total_loss'][0]:.4f}")
    print(f"Final total loss: {metrics['total_loss'][-1]:.4f}")
    print(f"Loss reduction: {metrics['total_loss'][0] - metrics['total_loss'][-1]:.4f}")
    print(
        f"Loss reduction %: {((metrics['total_loss'][0] - metrics['total_loss'][-1]) / metrics['total_loss'][0] * 100):.2f}%"
    )

    print(f"\nInitial CE loss: {metrics['ce_loss'][0]:.4f}")
    print(f"Final CE loss: {metrics['ce_loss'][-1]:.4f}")
    print(f"Initial perplexity: {metrics['perplexity'][0]:.2f}")
    print(f"Final perplexity: {metrics['perplexity'][-1]:.2f}")

    # Load balancing analysis
    print("\n=== LOAD BALANCING ANALYSIS ===")
    print(f"Final total LB loss: {metrics['lb_loss'][-1]:.4f}")
    print(f"Final stage 0 LB loss: {metrics['lb_stage_0'][-1]:.4f}")
    print(f"Final stage 1 LB loss: {metrics['lb_stage_1'][-1]:.4f}")
    print("Target LB loss per stage: 1.0")
    print(f"Stage 0 deviation: {abs(metrics['lb_stage_0'][-1] - 1.0):.4f}")
    print(f"Stage 1 deviation: {abs(metrics['lb_stage_1'][-1] - 1.0):.4f}")

    # Gradient analysis
    print("\n=== GRADIENT ANALYSIS ===")
    print(f"Initial gradient norm: {metrics['grad_norm'][0]:.4f}")
    print(f"Final gradient norm: {metrics['grad_norm'][-1]:.4f}")
    print(f"Max gradient norm: {max(metrics['grad_norm']):.4f}")
    print(f"Min gradient norm: {min(metrics['grad_norm']):.4f}")
    print(f"Avg gradient norm: {np.mean(metrics['grad_norm']):.4f}")

    # Learning rate analysis
    print("\n=== LEARNING RATE ANALYSIS ===")
    print(f"Final LR group 0: {metrics['lr_group_0'][-1]:.2e}")
    print(f"Final LR group 1: {metrics['lr_group_1'][-1]:.2e}")
    print(f"Final LR group 2: {metrics['lr_group_2'][-1]:.2e}")
    print(f"Final LR group 3: {metrics['lr_group_3'][-1]:.2e}")
    print(f"Final LR group 4: {metrics['lr_group_4'][-1]:.2e}")

    # Check for issues
    print("\n=== POTENTIAL ISSUES ===")

    # Check if LB loss is too high
    if metrics["lb_stage_0"][-1] > 1.1 or metrics["lb_stage_1"][-1] > 1.1:
        print("⚠️  WARNING: Load balancing loss is too high (>1.1)")
        print("   This indicates poor hierarchical routing decisions")

    # Check if perplexity is reasonable
    if metrics["perplexity"][-1] > 10:
        print("⚠️  WARNING: Final perplexity is very high (>10)")
        print("   This suggests the model hasn't learned well")

    # Check for gradient explosion
    if max(metrics["grad_norm"]) > 10:
        print("⚠️  WARNING: Gradient norms are very high (>10)")
        print("   This suggests potential gradient explosion")

    # Check for loss plateau
    recent_losses = (
        metrics["total_loss"][-100:] if len(metrics["total_loss"]) > 100 else metrics["total_loss"]
    )
    loss_std = np.std(recent_losses)
    if loss_std < 0.01:
        print("⚠️  WARNING: Loss appears to have plateaued")
        print("   Very low variance in recent losses")

    # Check learning rate schedule
    lr_ratio = metrics["lr_group_0"][-1] / metrics["lr_group_4"][-1]
    if lr_ratio > 5:
        print("⚠️  WARNING: Very high learning rate ratio between stages")
        print(f"   Stage 0 LR is {lr_ratio:.1f}x higher than Stage 2")


def main():
    parser = argparse.ArgumentParser(description="Analyze H-Net training metrics")
    parser.add_argument(
        "--log-path",
        type=str,
        default="/home/ka/ka_stud/ka_upygb/repos/hierarchical-hnet/outputs/hnet_2stage_XL_distributed/train.log",
        help="Path to training log file",
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save plots to files instead of displaying"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis_output",
        help="Directory to save analysis outputs",
    )

    args = parser.parse_args()

    print("Parsing training log...")
    metrics = parse_training_log(args.log_path)

    if not metrics["steps"]:
        print("No training steps found in log file!")
        return

    print(f"Found {len(metrics['steps'])} training steps")

    # Create output directory if saving plots
    if args.save_plots:
        import os

        os.makedirs(args.output_dir, exist_ok=True)

    # Analyze metrics
    analyze_metrics(metrics)

    # Create plots
    print("\nGenerating plots...")
    loss_plot_path = f"{args.output_dir}/loss_curves.png" if args.save_plots else None
    lr_plot_path = f"{args.output_dir}/learning_rates.png" if args.save_plots else None

    plot_loss_curves(metrics, loss_plot_path)
    plot_learning_rates(metrics, lr_plot_path)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
