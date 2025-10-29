#!/usr/bin/env python3
"""
Training configuration analysis script for H-Net diagnostic review.

This script analyzes:
1. Hyperparameter settings
2. Optimizer configuration
3. Learning rate schedule
4. Gradient clipping
5. Batch size and accumulation
6. Data type and precision
"""

import math


def analyze_hyperparameters():
    """Analyze training hyperparameters."""
    print("=== TRAINING HYPERPARAMETERS ANALYSIS ===\n")

    # Default values from train.py
    hyperparams = {
        "learning_rate": 3e-4,
        "lr_multipliers": [3.0, 1.7, 0.9],
        "warmup_steps": 2000,
        "num_training_steps": 100000,
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "weight_decay": 0.1,
        "load_balancing_weight": 0.01,
        "initializer_range": 0.02,
        "dtype": "bfloat16",
    }

    print("Current hyperparameter settings:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")

    # Calculate effective learning rates
    print("\nEffective learning rates per stage:")
    for i, mult in enumerate(hyperparams["lr_multipliers"]):
        effective_lr = hyperparams["learning_rate"] * mult
        print(f"  Stage {i}: {effective_lr:.2e} ({mult}x base)")

    # Calculate warmup ratio
    warmup_ratio = hyperparams["warmup_steps"] / hyperparams["num_training_steps"]
    print("\nWarmup analysis:")
    print(f"  Warmup steps: {hyperparams['warmup_steps']}")
    print(f"  Total steps: {hyperparams['num_training_steps']}")
    print(f"  Warmup ratio: {warmup_ratio:.1%}")

    return hyperparams


def analyze_optimizer_config():
    """Analyze optimizer configuration."""
    print("\n=== OPTIMIZER CONFIGURATION ANALYSIS ===\n")

    print("Optimizer: AdamW")
    print("Betas: (0.9, 0.95)")
    print("Weight decay: 0.1")
    print("Gradient clipping: 1.0")

    print("\nParameter grouping:")
    print("  - Parameters grouped by lr_multiplier and weight_decay")
    print("  - Bias and norm parameters have weight_decay=0")
    print("  - Each stage gets different learning rate")

    print("\nGradient clipping:")
    print("  - Max norm: 1.0")
    print("  - Applied after gradient accumulation")
    print("  - This is quite aggressive for large models")


def analyze_learning_rate_schedule():
    """Analyze learning rate schedule."""
    print("\n=== LEARNING RATE SCHEDULE ANALYSIS ===\n")

    warmup_steps = 2000
    total_steps = 100000

    print("Schedule type: Cosine with warmup")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Total steps: {total_steps}")
    print("Cosine cycles: 0.5")

    # Calculate LR at different points
    def cosine_lr(step, warmup_steps, total_steps, base_lr=3e-4):
        if step < warmup_steps:
            return base_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    print("\nLearning rate at key points:")
    key_steps = [0, 1000, 2000, 10000, 50000, 100000]
    for step in key_steps:
        if step <= total_steps:
            lr = cosine_lr(step, warmup_steps, total_steps)
            print(f"  Step {step:6d}: {lr:.2e}")

    print("\nWarmup phase analysis:")
    print(f"  - Linear warmup from 0 to base_lr over {warmup_steps} steps")
    print(f"  - At step 1000: {cosine_lr(1000, warmup_steps, total_steps):.2e}")
    print(f"  - At step 2000: {cosine_lr(2000, warmup_steps, total_steps):.2e}")

    print("\nCosine decay phase:")
    print("  - Cosine decay from base_lr to 0 over remaining steps")
    print(f"  - At step 50000: {cosine_lr(50000, warmup_steps, total_steps):.2e}")
    print(f"  - At step 100000: {cosine_lr(100000, warmup_steps, total_steps):.2e}")


def analyze_batch_configuration():
    """Analyze batch size and gradient accumulation."""
    print("\n=== BATCH CONFIGURATION ANALYSIS ===\n")

    batch_size = 8
    gradient_accumulation_steps = 1
    num_gpus = 4  # From distributed training

    effective_batch_size = batch_size * gradient_accumulation_steps * num_gpus

    print(f"Batch size per device: {batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Effective batch size: {effective_batch_size}")

    print("\nBatch size analysis:")
    print("  - Per-device batch size is reasonable for 1.6B parameter model")
    print(f"  - Effective batch size of {effective_batch_size} is good for stable training")
    print("  - No gradient accumulation needed with 4 GPUs")

    # Check if batch size is appropriate for model size
    if effective_batch_size < 16:
        print(f"⚠️  WARNING: Effective batch size ({effective_batch_size}) may be too small")
        print("   Consider increasing batch size or gradient accumulation")
    else:
        print("✓ Effective batch size is appropriate for model size")


def analyze_precision_and_dtype():
    """Analyze data type and precision settings."""
    print("\n=== PRECISION AND DATA TYPE ANALYSIS ===\n")

    dtype = "bfloat16"
    print(f"Data type: {dtype}")

    print("\nPrecision analysis:")
    print("  - bfloat16 provides good numerical stability")
    print("  - Reduces memory usage compared to float32")
    print("  - Good for training large models")
    print("  - Compatible with modern hardware (A100, H100)")

    print("\nPotential issues:")
    print("  - bfloat16 has limited precision (7 bits mantissa)")
    print("  - May cause numerical instability in some operations")
    print("  - Gradient accumulation should use float32 for stability")


def analyze_loss_configuration():
    """Analyze loss function configuration."""
    print("\n=== LOSS CONFIGURATION ANALYSIS ===\n")

    load_balancing_weight = 0.01
    print(f"Load balancing weight: {load_balancing_weight}")

    print("\nLoss composition:")
    print("  - Language modeling loss: 1.0 (primary)")
    print(f"  - Load balancing loss: {load_balancing_weight} (auxiliary)")
    print(f"  - Total loss = ce_loss + {load_balancing_weight} * lb_loss")

    print("\nLoad balancing analysis:")
    print(f"  - Weight of {load_balancing_weight} is reasonable")
    print("  - Prevents load balancing from dominating training")
    print("  - Allows model to focus on language modeling")

    print("\nFrom training analysis:")
    print("  - Final CE loss: ~0.82")
    print("  - Final LB loss: ~2.01")
    print(f"  - LB contribution: {load_balancing_weight * 2.01:.3f}")
    print(f"  - LB contributes ~{load_balancing_weight * 2.01 / 0.82 * 100:.1f}% to total loss")


def analyze_potential_issues():
    """Analyze potential training issues."""
    print("\n=== POTENTIAL TRAINING ISSUES ===\n")

    print("1. LEARNING RATE MULTIPLIERS TOO AGGRESSIVE:")
    print("   - Stage 0 gets 3.0x base LR (9e-4)")
    print("   - Stage 2 gets 0.9x base LR (2.7e-4)")
    print("   - 3.3x ratio can cause training instability")
    print("   - Recommendation: Reduce to [2.0, 1.5, 1.0]")

    print("\n2. WARMUP PERIOD TOO SHORT:")
    print("   - Only 2% of total training steps")
    print("   - Large models need longer warmup")
    print("   - Recommendation: Increase to 5000+ steps (5%+)")

    print("\n3. GRADIENT CLIPPING TOO AGGRESSIVE:")
    print("   - Max norm of 1.0 is very restrictive")
    print("   - May prevent proper learning")
    print("   - Recommendation: Increase to 5.0 or use adaptive clipping")

    print("\n4. LEARNING RATE TOO HIGH:")
    print("   - Base LR of 3e-4 may be too high for 1.6B params")
    print("   - Combined with high multipliers causes instability")
    print("   - Recommendation: Reduce base LR to 2e-4")

    print("\n5. INSUFFICIENT TRAINING:")
    print("   - Only 32,800 steps completed out of 100,000")
    print("   - Model may need more training time")
    print("   - Recommendation: Continue training or increase total steps")


def recommend_fixes():
    """Provide specific recommendations for fixing issues."""
    print("\n=== RECOMMENDED FIXES ===\n")

    print("1. REDUCE LEARNING RATE MULTIPLIERS:")
    print("   --lr-multipliers 2.0,1.5,1.0")
    print("   - Reduces LR ratio from 3.3x to 2.0x")
    print("   - More stable training dynamics")

    print("\n2. INCREASE WARMUP STEPS:")
    print("   --warmup-steps 5000")
    print("   - Increases warmup ratio to 5%")
    print("   - Better for large model initialization")

    print("\n3. REDUCE BASE LEARNING RATE:")
    print("   --learning-rate 2e-4")
    print("   - More conservative learning rate")
    print("   - Reduces risk of gradient explosion")

    print("\n4. INCREASE GRADIENT CLIPPING:")
    print("   --max-grad-norm 5.0")
    print("   - Less restrictive clipping")
    print("   - Allows for better gradient flow")

    print("\n5. ADD GRADIENT MONITORING:")
    print("   - Log gradient norms per stage")
    print("   - Implement adaptive clipping")
    print("   - Monitor for gradient explosion")

    print("\n6. CONSIDER LEARNING RATE SCHEDULE ADJUSTMENTS:")
    print("   - Use linear warmup instead of cosine")
    print("   - Add learning rate decay at specific milestones")
    print("   - Consider warmup restart for better convergence")


def main():
    print("H-Net Training Configuration Diagnostic Analysis")
    print("=" * 60)

    # Analyze each component
    analyze_hyperparameters()
    analyze_optimizer_config()
    analyze_learning_rate_schedule()
    analyze_batch_configuration()
    analyze_precision_and_dtype()
    analyze_loss_configuration()
    analyze_potential_issues()
    recommend_fixes()

    print("\n=== SUMMARY ===")
    print("The training configuration has several issues that likely caused")
    print("the poor generation quality:")
    print("1. Aggressive learning rate multipliers (3.3x ratio)")
    print("2. Insufficient warmup period (2% of total steps)")
    print("3. Gradient explosion due to high learning rates")
    print("4. Training stopped too early (32.8k/100k steps)")
    print("\nThe recommended fixes should significantly improve training stability")
    print("and generation quality.")


if __name__ == "__main__":
    main()
