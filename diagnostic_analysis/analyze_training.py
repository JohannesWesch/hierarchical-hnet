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

    # Pattern for 1-stage models with 2 LR groups
    step_pattern_1stage_2lr = (
        r"Step (\d+) \| Loss: ([\d.]+) \| loss: ([\d.]+), ce_loss: ([\d.]+), "
        r"lb_loss: ([\d.]+), lb_stage_0: ([\d.]+)(?:, [^|]+?)* \| "
        r"LR: group_0: ([\d.e-]+), group_1: ([\d.e-]+)"
    )

    # Pattern for 1-stage models with 3 LR groups
    step_pattern_1stage_3lr = (
        r"Step (\d+) \| Loss: ([\d.]+) \| loss: ([\d.]+), ce_loss: ([\d.]+), "
        r"lb_loss: ([\d.]+), lb_stage_0: ([\d.]+)(?:, [^|]+?)* \| "
        r"LR: group_0: ([\d.e-]+), group_1: ([\d.e-]+), group_2: ([\d.e-]+)"
    )

    # Pattern for 2-stage models (5 LR groups, 2 stages)
    step_pattern_2stage = (
        r"Step (\d+) \| Loss: ([\d.]+) \| loss: ([\d.]+), ce_loss: ([\d.]+), "
        r"lb_loss: ([\d.]+), lb_stage_0: ([\d.]+), lb_stage_1: ([\d.]+)"
        r"(?:, [^|]+?)* \| "
        r"LR: group_0: ([\d.e-]+), group_1: ([\d.e-]+), group_2: ([\d.e-]+), "
        r"group_3: ([\d.e-]+), group_4: ([\d.e-]+)"
    )

    # Pattern to match validation lines
    val_pattern = r"Validation metrics: \{'val_loss': ([\d.]+), 'val_ce_loss': ([\d.]+), 'val_lb_loss': ([\d.]+), 'val_perplexity': ([\d.]+)\}"

    # Pattern to extract step number before validation (from "Evaluating at step X")
    eval_step_pattern = r"Evaluating at step (\d+)"

    metrics = {
        "steps": [],
        "total_loss": [],
        "ce_loss": [],
        "lb_loss": [],
        "lb_stage_0": [],
        "lb_stage_1": [],
        "grad_norm": [],
        "grad_norm_steps": [],  # Track which steps have grad_norm
        "lr_group_0": [],
        "lr_group_1": [],
        "lr_group_2": [],
        "lr_group_3": [],
        "lr_group_4": [],
        "perplexity": [],
        "val_steps": [],
        "val_loss": [],
        "val_ce_loss": [],
        "val_lb_loss": [],
        "val_perplexity": [],
        "stage_0_F": [],
        "stage_0_G": [],
        "stage_0_actual_compression": [],
        "stage_0_target_compression": [],
        "stage_0_compression_error": [],
        "stage_1_F": [],
        "stage_1_G": [],
        "stage_1_actual_compression": [],
        "stage_1_target_compression": [],
        "stage_1_compression_error": [],
    }

    current_eval_step = None
    is_two_stage = None  # Auto-detect model type

    with open(log_path, "r") as f:
        for line in f:
            # Check for evaluation step marker
            eval_match = re.search(eval_step_pattern, line)
            if eval_match:
                current_eval_step = int(eval_match.group(1))

            # Try 2-stage pattern first
            match_2stage = re.search(step_pattern_2stage, line)
            if match_2stage:
                if is_two_stage is None:
                    is_two_stage = True
                    print("Detected 2-stage model format")

                step = int(match_2stage.group(1))
                total_loss = float(match_2stage.group(2))
                ce_loss = float(match_2stage.group(4))
                lb_loss = float(match_2stage.group(5))
                lb_stage_0 = float(match_2stage.group(6))
                lb_stage_1 = float(match_2stage.group(7))
                grad_match = re.search(r"grad_norm: ([\d.]+)", line)
                grad_norm = float(grad_match.group(1)) if grad_match else None
                lr_group_0 = float(match_2stage.group(8))
                lr_group_1 = float(match_2stage.group(9))
                lr_group_2 = float(match_2stage.group(10))
                lr_group_3 = float(match_2stage.group(11))
                lr_group_4 = float(match_2stage.group(12))

                metrics["steps"].append(step)
                metrics["total_loss"].append(total_loss)
                metrics["ce_loss"].append(ce_loss)
                metrics["lb_loss"].append(lb_loss)
                metrics["lb_stage_0"].append(lb_stage_0)
                metrics["lb_stage_1"].append(lb_stage_1)
                if grad_norm is not None:
                    metrics["grad_norm"].append(grad_norm)
                    metrics["grad_norm_steps"].append(step)
                metrics["lr_group_0"].append(lr_group_0)
                metrics["lr_group_1"].append(lr_group_1)
                metrics["lr_group_2"].append(lr_group_2)
                metrics["lr_group_3"].append(lr_group_3)
                metrics["lr_group_4"].append(lr_group_4)
                metrics["perplexity"].append(np.exp(ce_loss))
                _extract_compression_stats(
                    line,
                    metrics,
                    stage_indices=[0, 1],
                )
                continue

            # Try 1-stage pattern with 3 LR groups
            match_1stage_3lr = re.search(step_pattern_1stage_3lr, line)
            if match_1stage_3lr:
                if is_two_stage is None:
                    is_two_stage = False
                    print("Detected 1-stage model format (3 LR groups)")

                step = int(match_1stage_3lr.group(1))
                total_loss = float(match_1stage_3lr.group(2))
                ce_loss = float(match_1stage_3lr.group(4))
                lb_loss = float(match_1stage_3lr.group(5))
                lb_stage_0 = float(match_1stage_3lr.group(6))
                grad_match = re.search(r"grad_norm: ([\d.]+)", line)
                grad_norm = float(grad_match.group(1)) if grad_match else None
                lr_group_0 = float(match_1stage_3lr.group(7))
                lr_group_1 = float(match_1stage_3lr.group(8))
                lr_group_2 = float(match_1stage_3lr.group(9))

                metrics["steps"].append(step)
                metrics["total_loss"].append(total_loss)
                metrics["ce_loss"].append(ce_loss)
                metrics["lb_loss"].append(lb_loss)
                metrics["lb_stage_0"].append(lb_stage_0)
                if grad_norm is not None:
                    metrics["grad_norm"].append(grad_norm)
                    metrics["grad_norm_steps"].append(step)
                metrics["lr_group_0"].append(lr_group_0)
                metrics["lr_group_1"].append(lr_group_1)
                metrics["lr_group_2"].append(lr_group_2)
                metrics["perplexity"].append(np.exp(ce_loss))
                _extract_compression_stats(
                    line,
                    metrics,
                    stage_indices=[0],
                )
                continue

            # Try 1-stage pattern with 2 LR groups
            match_1stage_2lr = re.search(step_pattern_1stage_2lr, line)
            if match_1stage_2lr:
                if is_two_stage is None:
                    is_two_stage = False
                    print("Detected 1-stage model format (2 LR groups)")

                step = int(match_1stage_2lr.group(1))
                total_loss = float(match_1stage_2lr.group(2))
                ce_loss = float(match_1stage_2lr.group(4))
                lb_loss = float(match_1stage_2lr.group(5))
                lb_stage_0 = float(match_1stage_2lr.group(6))
                grad_match = re.search(r"grad_norm: ([\d.]+)", line)
                grad_norm = float(grad_match.group(1)) if grad_match else None
                lr_group_0 = float(match_1stage_2lr.group(7))
                lr_group_1 = float(match_1stage_2lr.group(8))

                metrics["steps"].append(step)
                metrics["total_loss"].append(total_loss)
                metrics["ce_loss"].append(ce_loss)
                metrics["lb_loss"].append(lb_loss)
                metrics["lb_stage_0"].append(lb_stage_0)
                if grad_norm is not None:
                    metrics["grad_norm"].append(grad_norm)
                    metrics["grad_norm_steps"].append(step)
                metrics["lr_group_0"].append(lr_group_0)
                metrics["lr_group_1"].append(lr_group_1)
                # For 2 LR groups, group 2 doesn't exist
                metrics["perplexity"].append(np.exp(ce_loss))
                _extract_compression_stats(
                    line,
                    metrics,
                    stage_indices=[0],
                )
                continue

            # Check for validation metrics
            val_match = re.search(val_pattern, line)
            if val_match and current_eval_step is not None:
                val_loss = float(val_match.group(1))
                val_ce_loss = float(val_match.group(2))
                val_lb_loss = float(val_match.group(3))
                val_perplexity = float(val_match.group(4))

                metrics["val_steps"].append(current_eval_step)
                metrics["val_loss"].append(val_loss)
                metrics["val_ce_loss"].append(val_ce_loss)
                metrics["val_lb_loss"].append(val_lb_loss)
                metrics["val_perplexity"].append(val_perplexity)

                current_eval_step = None  # Reset after capturing

    return metrics


def _extract_compression_stats(
    line: str,
    metrics: Dict[str, List[float]],
    stage_indices: List[int],
) -> None:
    """Extract compression statistics per stage if present."""
    for stage_idx in stage_indices:
        prefix = f"stage_{stage_idx}_"
        f_match = re.search(rf"{prefix}F_actual_selection: ([\d.]+)", line)
        g_match = re.search(rf"{prefix}G_avg_prob: ([\d.]+)", line)
        actual_match = re.search(rf"{prefix}actual_compression: ([\d.]+)", line)
        target_match = re.search(rf"{prefix}target_compression: ([\d.]+)", line)
        error_match = re.search(rf"{prefix}compression_error: ([\-\d.]+)", line)

        metrics[f"{prefix}F"].append(float(f_match.group(1)) if f_match else np.nan)
        metrics[f"{prefix}G"].append(float(g_match.group(1)) if g_match else np.nan)
        metrics[f"{prefix}actual_compression"].append(
            float(actual_match.group(1)) if actual_match else np.nan
        )
        metrics[f"{prefix}target_compression"].append(
            float(target_match.group(1)) if target_match else np.nan
        )
        metrics[f"{prefix}compression_error"].append(
            float(error_match.group(1)) if error_match else np.nan
        )


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

    # CE Loss and Perplexity (with validation)
    ax1 = axes[0, 1]
    ax2 = ax1.twinx()

    line1 = ax1.plot(metrics["steps"], metrics["ce_loss"], "g-", label="Train CE Loss", alpha=0.7)
    line2 = ax2.plot(
        metrics["steps"], metrics["perplexity"], "r-", label="Train Perplexity", alpha=0.7
    )

    # Add validation metrics if available
    if metrics["val_steps"]:
        line3 = ax1.plot(
            metrics["val_steps"],
            metrics["val_ce_loss"],
            "g--",
            marker="o",
            label="Val CE Loss",
            alpha=0.8,
            markersize=6,
        )
        line4 = ax2.plot(
            metrics["val_steps"],
            metrics["val_perplexity"],
            "r--",
            marker="s",
            label="Val Perplexity",
            alpha=0.8,
            markersize=6,
        )
        lines = line1 + line2 + line3 + line4
    else:
        lines = line1 + line2

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("CE Loss", color="g")
    ax2.set_ylabel("Perplexity", color="r")
    ax1.set_title("Cross-Entropy Loss and Perplexity")
    ax1.grid(True, alpha=0.3)

    # Combine legends
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    # Load Balancing Loss
    axes[1, 0].plot(metrics["steps"], metrics["lb_loss"], "m-", label="Total LB Loss", alpha=0.7)
    axes[1, 0].plot(metrics["steps"], metrics["lb_stage_0"], "c-", label="Stage 0 LB", alpha=0.7)
    # Only plot stage 1 if it exists (2-stage models)
    if metrics["lb_stage_1"]:
        axes[1, 0].plot(
            metrics["steps"], metrics["lb_stage_1"], "y-", label="Stage 1 LB", alpha=0.7
        )
    axes[1, 0].axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Target (1.0)")
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel("Load Balancing Loss")
    axes[1, 0].set_title("Load Balancing Loss by Stage")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Gradient Norm
    if metrics["grad_norm"] and metrics["grad_norm_steps"]:
        axes[1, 1].plot(metrics["grad_norm_steps"], metrics["grad_norm"], "orange", alpha=0.7)
        axes[1, 1].set_xlabel("Training Step")
        axes[1, 1].set_ylabel("Gradient Norm")
        axes[1, 1].set_title("Gradient Norm Over Time")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale("log")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No gradient norm data",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_xlabel("Training Step")
        axes[1, 1].set_ylabel("Gradient Norm")
        axes[1, 1].set_title("Gradient Norm Over Time (No Data)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss curves saved to {save_path}")
    else:
        plt.show()


def plot_learning_rates(metrics: Dict[str, List[float]], save_path: str = None):
    """Plot learning rate schedules."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Always plot the first 2 groups (exist in all models)
    ax.plot(metrics["steps"], metrics["lr_group_0"], "b-", label="Group 0", alpha=0.7)
    ax.plot(metrics["steps"], metrics["lr_group_1"], "g-", label="Group 1", alpha=0.7)

    # Plot group 2 if it exists (3+ LR group models)
    if metrics["lr_group_2"]:
        ax.plot(metrics["steps"], metrics["lr_group_2"], "r-", label="Group 2", alpha=0.7)

    # Only plot groups 3 and 4 if they exist (2-stage models)
    if metrics["lr_group_3"]:
        ax.plot(metrics["steps"], metrics["lr_group_3"], "c-", label="Group 3", alpha=0.7)
    if metrics["lr_group_4"]:
        ax.plot(metrics["steps"], metrics["lr_group_4"], "m-", label="Group 4", alpha=0.7)

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


def _has_valid(values: List[float]) -> bool:
    arr = np.array(values, dtype=float)
    return arr.size > 0 and not np.all(np.isnan(arr))


def plot_compression_stats(metrics: Dict[str, List[float]], save_path: str = None):
    """Plot routing compression statistics per stage."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    stage0_actual = np.array(metrics["stage_0_actual_compression"])
    stage0_target = np.array(metrics["stage_0_target_compression"])
    stage1_actual = np.array(metrics["stage_1_actual_compression"])
    stage1_target = np.array(metrics["stage_1_target_compression"])
    steps = np.array(metrics["steps"])

    if stage0_actual.size and not np.all(np.isnan(stage0_actual)):
        valid = ~np.isnan(stage0_actual)
        ax.plot(
            steps[valid],
            stage0_actual[valid],
            label="Stage 0 Actual Compression",
            color="tab:blue",
        )
        valid_target = ~np.isnan(stage0_target)
        ax.plot(
            steps[valid_target],
            stage0_target[valid_target],
            label="Stage 0 Target Compression",
            color="tab:blue",
            linestyle="--",
        )

    if stage1_actual.size and not np.all(np.isnan(stage1_actual)):
        valid = ~np.isnan(stage1_actual)
        ax.plot(
            steps[valid],
            stage1_actual[valid],
            label="Stage 1 Actual Compression",
            color="tab:orange",
        )
        valid_target = ~np.isnan(stage1_target)
        ax.plot(
            steps[valid_target],
            stage1_target[valid_target],
            label="Stage 1 Target Compression",
            color="tab:orange",
            linestyle="--",
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Compression Ratio")
    ax.set_title("Compression Targets vs Actual")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Compression plot saved to {save_path}")
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

    # Validation metrics if available
    if metrics["val_steps"]:
        print("\n=== VALIDATION ANALYSIS ===")
        print(f"Number of validation runs: {len(metrics['val_steps'])}")
        print(f"Initial validation loss: {metrics['val_loss'][0]:.4f}")
        print(f"Final validation loss: {metrics['val_loss'][-1]:.4f}")
        print(f"Initial validation perplexity: {metrics['val_perplexity'][0]:.2f}")
        print(f"Final validation perplexity: {metrics['val_perplexity'][-1]:.2f}")

        # Compare train vs val
        final_train_loss = metrics["ce_loss"][-1]
        final_val_loss = metrics["val_ce_loss"][-1]
        gap = final_val_loss - final_train_loss
        print(f"\nTrain-Val gap (CE loss): {gap:.4f}")
        if gap > 0.5:
            print("  ⚠️  Large gap suggests overfitting")
        elif gap < 0:
            print("  ✅ Val loss better than train (good generalization)")
        else:
            print("  ✅ Small gap (good generalization)")

    # Load balancing analysis
    print("\n=== LOAD BALANCING ANALYSIS ===")
    print(f"Final total LB loss: {metrics['lb_loss'][-1]:.4f}")
    print(f"Final stage 0 LB loss: {metrics['lb_stage_0'][-1]:.4f}")
    if metrics["lb_stage_1"]:
        print(f"Final stage 1 LB loss: {metrics['lb_stage_1'][-1]:.4f}")
    print("Target LB loss per stage: 1.0")
    print(f"Stage 0 deviation: {abs(metrics['lb_stage_0'][-1] - 1.0):.4f}")
    if metrics["lb_stage_1"]:
        print(f"Stage 1 deviation: {abs(metrics['lb_stage_1'][-1] - 1.0):.4f}")

    # Compression analysis
    if _has_valid(metrics["stage_0_actual_compression"]):
        print("\n=== COMPRESSION ANALYSIS ===")
        stage0_actual_arr = np.array(metrics["stage_0_actual_compression"], dtype=float)
        stage0_target_arr = np.array(metrics["stage_0_target_compression"], dtype=float)
        stage0_actual = stage0_actual_arr[~np.isnan(stage0_actual_arr)][-1]
        stage0_target = stage0_target_arr[~np.isnan(stage0_target_arr)][-1]
        print(
            f"Stage 0 compression: actual={stage0_actual:.2f}, "
            f"target={stage0_target:.2f}, error={stage0_actual - stage0_target:.2f}"
        )
        if _has_valid(metrics["stage_1_actual_compression"]):
            stage1_actual_arr = np.array(metrics["stage_1_actual_compression"], dtype=float)
            stage1_target_arr = np.array(metrics["stage_1_target_compression"], dtype=float)
            stage1_actual = stage1_actual_arr[~np.isnan(stage1_actual_arr)][-1]
            stage1_target = stage1_target_arr[~np.isnan(stage1_target_arr)][-1]
            print(
                f"Stage 1 compression: actual={stage1_actual:.2f}, "
                f"target={stage1_target:.2f}, error={stage1_actual - stage1_target:.2f}"
            )

    # Gradient analysis
    print("\n=== GRADIENT ANALYSIS ===")
    if metrics["grad_norm"]:
        print(f"Initial gradient norm: {metrics['grad_norm'][0]:.4f}")
        print(f"Final gradient norm: {metrics['grad_norm'][-1]:.4f}")
        print(f"Max gradient norm: {max(metrics['grad_norm']):.4f}")
        print(f"Min gradient norm: {min(metrics['grad_norm']):.4f}")
        print(f"Avg gradient norm: {np.mean(metrics['grad_norm']):.4f}")
    else:
        print("No gradient norm data available in log")

    # Learning rate analysis
    print("\n=== LEARNING RATE ANALYSIS ===")
    print(f"Final LR group 0: {metrics['lr_group_0'][-1]:.2e}")
    print(f"Final LR group 1: {metrics['lr_group_1'][-1]:.2e}")
    if metrics["lr_group_2"]:
        print(f"Final LR group 2: {metrics['lr_group_2'][-1]:.2e}")
    if metrics["lr_group_3"]:
        print(f"Final LR group 3: {metrics['lr_group_3'][-1]:.2e}")
    if metrics["lr_group_4"]:
        print(f"Final LR group 4: {metrics['lr_group_4'][-1]:.2e}")

    # Check for issues
    print("\n=== POTENTIAL ISSUES ===")

    # Check if LB loss is too high
    lb_too_high = metrics["lb_stage_0"][-1] > 1.1
    if metrics["lb_stage_1"]:
        lb_too_high = lb_too_high or metrics["lb_stage_1"][-1] > 1.1
    if lb_too_high:
        print("⚠️  WARNING: Load balancing loss is too high (>1.1)")
        print("   This indicates poor hierarchical routing decisions")

    # Check if perplexity is reasonable
    if metrics["perplexity"][-1] > 10:
        print("⚠️  WARNING: Final perplexity is very high (>10)")
        print("   This suggests the model hasn't learned well")

    # Check for gradient explosion (ignore initial spike, check recent behavior)
    if metrics["grad_norm"]:
        recent_grad_norms = (
            metrics["grad_norm"][-100:]
            if len(metrics["grad_norm"]) > 100
            else (
                metrics["grad_norm"][10:]
                if len(metrics["grad_norm"]) > 10
                else metrics["grad_norm"]
            )
        )
        avg_recent_grad = np.mean(recent_grad_norms) if recent_grad_norms else 0
        max_recent_grad = max(recent_grad_norms) if recent_grad_norms else 0

        if avg_recent_grad > 15 or max_recent_grad > 50:
            print("⚠️  WARNING: Gradient norms are very high")
            print(f"   Recent average: {avg_recent_grad:.2f}, Recent max: {max_recent_grad:.2f}")
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
    if metrics["lr_group_4"]:
        lr_ratio = metrics["lr_group_0"][-1] / metrics["lr_group_4"][-1]
        if lr_ratio > 5:
            print("⚠️  WARNING: Very high learning rate ratio between stages")
            print(f"   Group 0 LR is {lr_ratio:.1f}x higher than Group 4")


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
    compression_plot_path = f"{args.output_dir}/compression.png" if args.save_plots else None

    plot_loss_curves(metrics, loss_plot_path)
    plot_learning_rates(metrics, lr_plot_path)
    if _has_valid(metrics["stage_0_actual_compression"]):
        plot_compression_stats(metrics, compression_plot_path)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
