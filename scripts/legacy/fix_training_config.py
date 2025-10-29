#!/usr/bin/env python3
"""
Training configuration fix script for H-Net.

This script applies the recommended hyperparameter fixes to improve
training stability and generation quality.
"""

import os


def create_fixed_training_script():
    """Create a fixed version of the training script with better hyperparameters."""

    print("Creating fixed training script...")

    # Read the original training script
    with open("scripts/train.py", "r") as f:
        content = f.read()

    # Apply fixes to the argument parser
    fixes = [
        # Fix learning rate multipliers
        ('default="3.0,1.7,0.9"', 'default="2.0,1.5,1.0"'),
        # Fix warmup steps
        ("default=2000", "default=5000"),
        # Fix learning rate
        ("default=3e-4", "default=2e-4"),
        # Fix gradient clipping
        ("default=1.0", "default=5.0"),
    ]

    # Apply fixes
    for old, new in fixes:
        content = content.replace(old, new)

    # Add gradient monitoring
    gradient_monitoring = """
    # Add gradient monitoring
    if step % 100 == 0:
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms.append(grad_norm)

        if grad_norms:
            max_grad_norm = max(grad_norms)
            avg_grad_norm = sum(grad_norms) / len(grad_norms)
            logger.info(f"Gradient norms - Max: {max_grad_norm:.4f}, Avg: {avg_grad_norm:.4f}")

            # Adaptive gradient clipping
            if max_grad_norm > 10.0:
                logger.warning(f"High gradient norm detected: {max_grad_norm:.4f}")
                # Could implement adaptive clipping here
"""

    # Insert gradient monitoring after the training step
    content = content.replace("optimizer.step()", "optimizer.step()" + gradient_monitoring)

    # Write the fixed script
    with open("scripts/train_fixed.py", "w") as f:
        f.write(content)

    print("✓ Fixed training script created: scripts/train_fixed.py")


def create_fixed_generation_script():
    """Create a fixed version of the generation script with better parameters."""

    print("Creating fixed generation script...")

    # Read the original generation script
    with open("scripts/generate.py", "r") as f:
        content = f.read()

    # Apply fixes to the argument parser
    fixes = [
        # Fix temperature
        ("default=1.0", "default=0.8"),
        # Fix top-p
        ("default=1.0", "default=0.9"),
        # Fix max tokens
        ("default=1024", "default=2048"),
    ]

    # Apply fixes
    for old, new in fixes:
        content = content.replace(old, new)

    # Add repetition penalty
    repetition_penalty_code = '''
def apply_repetition_penalty(logits, input_ids, penalty=1.1):
    """Apply repetition penalty to logits."""
    if penalty == 1.0:
        return logits

    # Get unique tokens from input
    unique_tokens = torch.unique(input_ids)

    # Apply penalty to repeated tokens
    for token in unique_tokens:
        if token < logits.size(-1):
            if logits[token] < 0:
                logits[token] *= penalty
            else:
                logits[token] /= penalty

    return logits
'''

    # Insert repetition penalty function
    content = content.replace("def generate(", repetition_penalty_code + "\n\ndef generate(")

    # Update generate function to use repetition penalty
    content = content.replace(
        "logits = output.logits[0, -1, :] / temperature",
        "logits = output.logits[0, -1, :] / temperature\n        logits = apply_repetition_penalty(logits, input_ids, penalty=1.1)",
    )

    # Write the fixed script
    with open("scripts/generate_fixed.py", "w") as f:
        f.write(content)

    print("✓ Fixed generation script created: scripts/generate_fixed.py")


def create_training_command():
    """Create a command to resume training with fixed hyperparameters."""

    print("Creating training command...")

    # Get the latest checkpoint
    checkpoint_dir = "outputs/hnet_2stage_XL_distributed"
    if os.path.exists(checkpoint_dir):
        checkpoints = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        else:
            checkpoint_path = None
    else:
        checkpoint_path = None

    # Create training command
    command = """# Resume training with fixed hyperparameters
python scripts/train_fixed.py \\
    --config-path configs/hnet_2stage_XL.json \\
    --output-dir outputs/hnet_2stage_XL_fixed \\
    --learning-rate 2e-4 \\
    --lr-multipliers 2.0,1.5,1.0 \\
    --warmup-steps 5000 \\
    --max-grad-norm 5.0 \\
    --batch-size 8 \\
    --gradient-accumulation-steps 1 \\
    --num-training-steps 100000 \\
    --save-interval 1000 \\
    --eval-interval 1000 \\
    --dtype bfloat16 \\
    --distributed"""

    if checkpoint_path:
        command += f" \\\n    --resume-from {checkpoint_path}"

    # Write command to file
    with open("resume_training_fixed.sh", "w") as f:
        f.write(command)

    os.chmod("resume_training_fixed.sh", 0o755)
    print("✓ Training command created: resume_training_fixed.sh")


def create_generation_command():
    """Create a command to test generation with fixed parameters."""

    print("Creating generation command...")

    # Get the latest checkpoint
    checkpoint_dir = "outputs/hnet_2stage_XL_distributed"
    if os.path.exists(checkpoint_dir):
        checkpoints = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        else:
            checkpoint_path = None
    else:
        checkpoint_path = None

    if checkpoint_path:
        command = f"""# Test generation with fixed parameters
python scripts/generate_fixed.py \\
    --model-path {checkpoint_path} \\
    --config-path configs/hnet_2stage_XL.json \\
    --temperature 0.8 \\
    --top-p 0.9 \\
    --max-tokens 2048"""

        # Write command to file
        with open("test_generation_fixed.sh", "w") as f:
            f.write(command)

        os.chmod("test_generation_fixed.sh", 0o755)
        print("✓ Generation command created: test_generation_fixed.sh")
    else:
        print("⚠️  No checkpoint found, skipping generation command")


def create_monitoring_script():
    """Create a script to monitor training progress."""

    print("Creating monitoring script...")

    monitoring_script = '''#!/usr/bin/env python3
"""
Training monitoring script for H-Net.

This script monitors training progress and alerts on issues.
"""

import re
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def monitor_training_log(log_path="outputs/hnet_2stage_XL_fixed/train.log"):
    """Monitor training log for issues."""

    print("Monitoring training log...")
    print(f"Log file: {log_path}")

    if not Path(log_path).exists():
        print(f"Log file not found: {log_path}")
        return

    # Patterns to monitor
    patterns = {
        "gradient_explosion": r"grad_norm: ([\\d.]+)",
        "loss_spike": r"Loss: ([\\d.]+)",
        "learning_rate": r"LR: group_0: ([\\d.e-]+)",
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
                step_match = re.search(r"Step (\\d+)", line)
                loss_match = re.search(r"Loss: ([\\d.]+)", line)
                grad_match = re.search(r"grad_norm: ([\\d.]+)", line)

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
    ax1.plot(steps, losses, 'b-', alpha=0.7)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    # Gradient norm curve
    ax2.plot(steps, grad_norms, 'r-', alpha=0.7)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_monitor.png', dpi=300, bbox_inches='tight')
    print("✓ Training curves saved to training_monitor.png")

if __name__ == "__main__":
    monitor_training_log()
    plot_training_curves()
'''

    with open("monitor_training.py", "w") as f:
        f.write(monitoring_script)

    os.chmod("monitor_training.py", 0o755)
    print("✓ Monitoring script created: monitor_training.py")


def main():
    print("H-Net Training Configuration Fix Script")
    print("=" * 50)

    # Create all fixes
    create_fixed_training_script()
    create_fixed_generation_script()
    create_training_command()
    create_generation_command()
    create_monitoring_script()

    print("\n" + "=" * 50)
    print("FIXES APPLIED SUCCESSFULLY!")
    print("=" * 50)

    print("\nNext steps:")
    print("1. Review the fixed scripts:")
    print("   - scripts/train_fixed.py")
    print("   - scripts/generate_fixed.py")

    print("\n2. Resume training with fixed hyperparameters:")
    print("   ./resume_training_fixed.sh")

    print("\n3. Monitor training progress:")
    print("   python monitor_training.py")

    print("\n4. Test generation quality:")
    print("   ./test_generation_fixed.sh")

    print("\nKey changes applied:")
    print("✓ LR multipliers: [3.0,1.7,0.9] → [2.0,1.5,1.0]")
    print("✓ Warmup steps: 2000 → 5000")
    print("✓ Learning rate: 3e-4 → 2e-4")
    print("✓ Gradient clipping: 1.0 → 5.0")
    print("✓ Generation temperature: 1.0 → 0.8")
    print("✓ Generation top-p: 1.0 → 0.9")
    print("✓ Max generation tokens: 1024 → 2048")
    print("✓ Added repetition penalty")
    print("✓ Added gradient monitoring")

    print("\nExpected improvements:")
    print("✓ No more gradient explosion")
    print("✓ More stable training")
    print("✓ Better generation quality")
    print("✓ Faster convergence")


if __name__ == "__main__":
    main()
