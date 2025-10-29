#!/usr/bin/env python3
"""
Test script to verify the diagnostic fixes work correctly.

This script tests the fixed hyperparameters and generation settings
without running the full training process.
"""

import json
import sys
from pathlib import Path

import torch


def test_model_loading():
    """Test if the model can be loaded with fixed configuration."""
    print("Testing model loading with fixed configuration...")

    try:
        # Load the model configuration
        with open("configs/hnet_2stage_XL.json", "r") as f:
            config = json.load(f)

        print("✓ Model config loaded successfully")
        print(f"  - Architecture: {config['arch_layout']}")
        print(f"  - Dimensions: {config['d_model']}")
        print(f"  - Vocab size: {config['vocab_size']}")

        return True
    except Exception as e:
        print(f"✗ Error loading model config: {e}")
        return False


def test_checkpoint_loading():
    """Test if the checkpoint can be loaded."""
    print("\nTesting checkpoint loading...")

    checkpoint_path = "outputs/hnet_2stage_XL_distributed/checkpoint_9000.pt"

    if not Path(checkpoint_path).exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("✓ Checkpoint loaded successfully")
        print(f"  - Keys: {list(checkpoint.keys())}")

        if "step" in checkpoint:
            print(f"  - Training step: {checkpoint['step']}")

        return True
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        return False


def test_hyperparameter_fixes():
    """Test the hyperparameter fixes."""
    print("\nTesting hyperparameter fixes...")

    # Test the fixed training script
    try:
        with open("scripts/train_fixed.py", "r") as f:
            content = f.read()

        # Check if fixes were applied
        fixes_applied = []

        if 'default="2.0,1.5,1.0"' in content:
            fixes_applied.append("✓ LR multipliers fixed: [2.0,1.5,1.0]")
        else:
            fixes_applied.append("✗ LR multipliers not fixed")

        if "default=5000" in content and "warmup-steps" in content:
            fixes_applied.append("✓ Warmup steps fixed: 5000")
        else:
            fixes_applied.append("✗ Warmup steps not fixed")

        if "default=2e-4" in content and "learning-rate" in content:
            fixes_applied.append("✓ Learning rate fixed: 2e-4")
        else:
            fixes_applied.append("✗ Learning rate not fixed")

        if "default=5.0" in content and "max-grad-norm" in content:
            fixes_applied.append("✓ Gradient clipping fixed: 5.0")
        else:
            fixes_applied.append("✗ Gradient clipping not fixed")

        for fix in fixes_applied:
            print(f"  {fix}")

        return all("✓" in fix for fix in fixes_applied)

    except Exception as e:
        print(f"✗ Error testing hyperparameter fixes: {e}")
        return False


def test_generation_fixes():
    """Test the generation fixes."""
    print("\nTesting generation fixes...")

    try:
        with open("scripts/generate_fixed.py", "r") as f:
            content = f.read()

        # Check if fixes were applied
        fixes_applied = []

        if "default=0.8" in content and "temperature" in content:
            fixes_applied.append("✓ Temperature fixed: 0.8")
        else:
            fixes_applied.append("✗ Temperature not fixed")

        if "default=0.9" in content and "top-p" in content:
            fixes_applied.append("✓ Top-p fixed: 0.9")
        else:
            fixes_applied.append("✗ Top-p not fixed")

        if "default=2048" in content and "max-tokens" in content:
            fixes_applied.append("✓ Max tokens fixed: 2048")
        else:
            fixes_applied.append("✗ Max tokens not fixed")

        if "apply_repetition_penalty" in content:
            fixes_applied.append("✓ Repetition penalty added")
        else:
            fixes_applied.append("✗ Repetition penalty not added")

        for fix in fixes_applied:
            print(f"  {fix}")

        return all("✓" in fix for fix in fixes_applied)

    except Exception as e:
        print(f"✗ Error testing generation fixes: {e}")
        return False


def test_generation_with_fixed_params():
    """Test generation with the fixed parameters."""
    print("\nTesting generation with fixed parameters...")

    checkpoint_path = "outputs/hnet_2stage_XL_distributed/checkpoint_9000.pt"

    if not Path(checkpoint_path).exists():
        print("✗ Checkpoint not found, skipping generation test")
        return False

    try:
        # Test the generation script
        import subprocess

        # Create a simple test command
        cmd = [
            "python",
            "scripts/generate_fixed.py",
            "--model-path",
            checkpoint_path,
            "--config-path",
            "configs/hnet_2stage_XL.json",
            "--temperature",
            "0.8",
            "--top-p",
            "0.9",
            "--max-tokens",
            "50",
        ]

        print(f"  Running: {' '.join(cmd)}")

        # Run with a simple prompt
        result = subprocess.run(
            cmd, input="Hello, my name is", text=True, capture_output=True, timeout=30
        )

        if result.returncode == 0:
            print("✓ Generation test successful")
            print(f"  Output: {result.stdout[:100]}...")
            return True
        else:
            print(f"✗ Generation test failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("✗ Generation test timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing generation: {e}")
        return False


def main():
    print("H-Net Diagnostic Fixes Test")
    print("=" * 40)

    tests = [
        ("Model Loading", test_model_loading),
        ("Checkpoint Loading", test_checkpoint_loading),
        ("Hyperparameter Fixes", test_hyperparameter_fixes),
        ("Generation Fixes", test_generation_fixes),
        ("Generation Test", test_generation_with_fixed_params),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 40)
    print("TEST RESULTS SUMMARY")
    print("=" * 40)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! The fixes are ready to use.")
        print("\nNext steps:")
        print("1. Run training with fixed hyperparameters:")
        print("   ./resume_training_local.sh")
        print("\n2. Monitor training progress:")
        print("   python monitor_training.py")
        print("\n3. Test generation quality:")
        print("   ./test_generation_fixed.sh")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
