#!/usr/bin/env python3
"""
Simple test script to verify the diagnostic fixes were applied correctly.

This script tests the fixed hyperparameters without requiring torch.
"""

import json
from pathlib import Path


def test_hyperparameter_fixes():
    """Test the hyperparameter fixes."""
    print("Testing hyperparameter fixes...")

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


def test_model_config():
    """Test if the model configuration is valid."""
    print("\nTesting model configuration...")

    try:
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


def test_checkpoint_exists():
    """Test if checkpoint exists."""
    print("\nTesting checkpoint availability...")

    checkpoint_path = "outputs/hnet_2stage_XL_distributed/checkpoint_9000.pt"

    if Path(checkpoint_path).exists():
        print(f"✓ Checkpoint found: {checkpoint_path}")
        return True
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False


def test_script_files():
    """Test if all script files were created."""
    print("\nTesting script files...")

    files_to_check = [
        "scripts/train_fixed.py",
        "scripts/generate_fixed.py",
        "resume_training_local.sh",
        "test_generation_fixed.sh",
        "monitor_training.py",
        "diagnostic_analysis/diagnostic_report.md",
    ]

    all_exist = True
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            all_exist = False

    return all_exist


def main():
    print("H-Net Diagnostic Fixes Test (Simple)")
    print("=" * 45)

    tests = [
        ("Model Configuration", test_model_config),
        ("Checkpoint Availability", test_checkpoint_exists),
        ("Script Files", test_script_files),
        ("Hyperparameter Fixes", test_hyperparameter_fixes),
        ("Generation Fixes", test_generation_fixes),
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

    print("\n" + "=" * 45)
    print("TEST RESULTS SUMMARY")
    print("=" * 45)

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
        print("\n4. Read the full diagnostic report:")
        print("   cat diagnostic_report.md")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
