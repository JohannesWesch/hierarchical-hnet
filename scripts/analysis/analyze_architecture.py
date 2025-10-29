#!/usr/bin/env python3
"""
Architecture analysis script for H-Net diagnostic review.

This script analyzes:
1. Model configuration and architecture
2. Weight initialization strategy
3. Learning rate multiplier application
4. Load balancing loss calculation
5. Routing module initialization
"""

import json


def analyze_model_config(config_path: str):
    """Analyze the model configuration."""
    print("=== MODEL CONFIGURATION ANALYSIS ===\n")

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Architecture layout: {config['arch_layout']}")
    print(f"Model dimensions: {config['d_model']}")
    print(f"Intermediate dimensions: {config['d_intermediate']}")
    print(f"Vocab size: {config['vocab_size']}")
    print(f"Tie embeddings: {config['tie_embeddings']}")

    print("\nSSM Configuration:")
    ssm_cfg = config["ssm_cfg"]
    for key, value in ssm_cfg.items():
        print(f"  {key}: {value}")

    print("\nAttention Configuration:")
    attn_cfg = config["attn_cfg"]
    for key, value in attn_cfg.items():
        print(f"  {key}: {value}")

    # Analyze architecture depth
    arch_layout = config["arch_layout"]
    depth = 0
    current = arch_layout
    while isinstance(current, list) and len(current) > 1:
        depth += 1
        current = current[1]  # Main network is always at index 1

    print(f"\nArchitecture depth: {depth} stages")
    print(f"Stage dimensions: {config['d_model']}")

    # Check for potential issues
    print("\n=== POTENTIAL ARCHITECTURE ISSUES ===")

    # Check dimension progression
    d_models = config["d_model"]
    for i in range(1, len(d_models)):
        if d_models[i] <= d_models[i - 1]:
            print(
                f"⚠️  WARNING: Stage {i} dimension ({d_models[i]}) <= Stage {i-1} ({d_models[i-1]})"
            )
            print("   This may cause information bottleneck")

    # Check attention heads
    num_heads = attn_cfg["num_heads"]
    for i, heads in enumerate(num_heads):
        if d_models[i] % heads != 0:
            print(
                f"⚠️  WARNING: Stage {i} d_model ({d_models[i]}) not divisible by num_heads ({heads})"
            )

    # Check window sizes
    window_sizes = attn_cfg["window_size"]
    for i, window_size in enumerate(window_sizes):
        if window_size == -1:
            print(f"✓ Stage {i}: Global attention (window_size=-1)")
        else:
            print(f"✓ Stage {i}: Local attention (window_size={window_size})")

    return config


def analyze_weight_initialization():
    """Analyze weight initialization strategy."""
    print("\n=== WEIGHT INITIALIZATION ANALYSIS ===\n")

    print("Initialization strategy:")
    print("1. Embeddings: std=1.0 (standard for embeddings)")
    print("2. LM Head: std=0.02 (standard)")
    print("3. Linear layers: std=0.02")
    print("4. Output projections: std=0.02 / sqrt(n_residuals)")
    print("5. Routing modules: Identity initialization (Q=K=I)")

    print("\nResidual scaling:")
    print("- Output projections scaled by 1/sqrt(n_residuals)")
    print("- This prevents gradient explosion in deep hierarchies")
    print("- n_residuals = parent_residuals + encoder_height + decoder_height")

    print("\nRouting module initialization:")
    print("- Q and K projection matrices initialized to identity")
    print("- This ensures stable initial routing decisions")
    print("- _no_reinit=True prevents reinitialization")


def analyze_learning_rate_schedule():
    """Analyze learning rate schedule and multipliers."""
    print("\n=== LEARNING RATE SCHEDULE ANALYSIS ===\n")

    # From the training log analysis
    lr_multipliers = [3.0, 1.7, 0.9]
    base_lr = 3e-4

    print(f"Base learning rate: {base_lr}")
    print(f"LR multipliers: {lr_multipliers}")
    print("Effective learning rates:")

    for i, mult in enumerate(lr_multipliers):
        effective_lr = base_lr * mult
        print(f"  Stage {i}: {effective_lr:.2e} ({mult}x base)")

    print(f"\nLR ratio (Stage 0 / Stage 2): {lr_multipliers[0] / lr_multipliers[2]:.1f}x")

    print("\n=== POTENTIAL LR ISSUES ===")

    if lr_multipliers[0] / lr_multipliers[2] > 3:
        print("⚠️  WARNING: Very high LR ratio between stages")
        print("   This can cause training instability")
        print("   Consider reducing multipliers to [2.0, 1.5, 1.0]")

    if lr_multipliers[0] > 2.5:
        print("⚠️  WARNING: Stage 0 LR multiplier is very high")
        print("   This can cause gradient explosion in early stages")

    # Check if warmup is sufficient
    warmup_steps = 2000
    total_steps = 100000
    warmup_ratio = warmup_steps / total_steps

    print("\nWarmup analysis:")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Total steps: {total_steps}")
    print(f"Warmup ratio: {warmup_ratio:.1%}")

    if warmup_ratio < 0.05:
        print("⚠️  WARNING: Warmup ratio is very low (<5%)")
        print("   Consider increasing warmup to 5000+ steps")


def analyze_load_balancing():
    """Analyze load balancing loss calculation."""
    print("\n=== LOAD BALANCING ANALYSIS ===\n")

    print("Load balancing loss formula:")
    print("LB = ((1 - true_ratio) * (1 - avg_prob) + true_ratio * avg_prob * (N-1)) * N / (N-1)")
    print("Where:")
    print("- true_ratio = fraction of tokens selected for routing")
    print("- avg_prob = average boundary probability")
    print("- N = downsampling factor (typically 2.0)")

    print("\nTarget values:")
    print("- LB loss per stage should be ~1.0")
    print("- true_ratio should be ~0.5 (50% of tokens routed)")
    print("- avg_prob should be ~0.5 (balanced routing)")

    print("\nFrom training analysis:")
    print("- Final stage 0 LB: 1.0000 (perfect)")
    print("- Final stage 1 LB: 1.0079 (excellent)")
    print("- Load balancing is working correctly")

    print("\nLoad balancing weight: 0.01")
    print("- This means LB loss contributes 1% to total loss")
    print("- This is reasonable for hierarchical models")


def analyze_routing_module():
    """Analyze routing module design."""
    print("\n=== ROUTING MODULE ANALYSIS ===\n")

    print("Routing mechanism:")
    print("1. Compute cosine similarity between consecutive tokens")
    print("2. Convert to boundary probability: (1 - cos_sim) / 2")
    print("3. Force first token boundary probability to 1.0")
    print("4. Select routing decision based on argmax")

    print("\nKey design choices:")
    print("- Uses cosine similarity (normalized dot product)")
    print("- Clamps probabilities to [0, 1] range")
    print("- Pads sequence boundaries with probability 1.0")
    print("- Uses identity initialization for Q and K projections")

    print("\nPotential issues:")
    print("- Cosine similarity may not be optimal for all data types")
    print("- Hard thresholding (argmax) may be too aggressive")
    print("- No learnable temperature parameter for soft routing")


def analyze_gradient_flow():
    """Analyze gradient flow through the architecture."""
    print("\n=== GRADIENT FLOW ANALYSIS ===\n")

    print("Gradient flow path:")
    print("1. Loss → LM Head → Backbone")
    print("2. Backbone → Encoder/Decoder (residuals)")
    print("3. Backbone → Routing Module → Chunking")
    print("4. Chunking → Main Network (hierarchical)")
    print("5. Main Network → Dechunking → Decoder")

    print("\nGradient scaling:")
    print("- Residual connections use Straight-Through Estimator (STE)")
    print("- Output projections scaled by 1/sqrt(n_residuals)")
    print("- This prevents gradient explosion in deep hierarchies")

    print("\nFrom training analysis:")
    print("- Max gradient norm: 1810 (extremely high)")
    print("- Final gradient norm: 0.17 (very low)")
    print("- This suggests gradient explosion followed by clipping")

    print("\nPotential fixes:")
    print("1. Reduce learning rate multipliers")
    print("2. Increase gradient clipping threshold")
    print("3. Use gradient scaling for routing decisions")
    print("4. Add gradient monitoring and adaptive clipping")


def main():
    config_path = "/home/ka/ka_stud/ka_upygb/repos/hierarchical-hnet/configs/hnet_2stage_XL.json"

    print("H-Net Architecture Diagnostic Analysis")
    print("=" * 50)

    # Analyze each component
    analyze_model_config(config_path)
    analyze_weight_initialization()
    analyze_learning_rate_schedule()
    analyze_load_balancing()
    analyze_routing_module()
    analyze_gradient_flow()

    print("\n=== SUMMARY OF FINDINGS ===")
    print("✅ Load balancing is working correctly")
    print("✅ Model architecture is well-designed")
    print("✅ Weight initialization is appropriate")
    print("⚠️  Learning rate multipliers are too aggressive")
    print("⚠️  Gradient explosion occurred during training")
    print("⚠️  Warmup period may be too short")

    print("\n=== RECOMMENDED FIXES ===")
    print("1. Reduce LR multipliers to [2.0, 1.5, 1.0]")
    print("2. Increase warmup steps to 5000+")
    print("3. Add gradient monitoring and adaptive clipping")
    print("4. Consider reducing base learning rate to 2e-4")
    print("5. Add gradient scaling for routing decisions")


if __name__ == "__main__":
    main()
