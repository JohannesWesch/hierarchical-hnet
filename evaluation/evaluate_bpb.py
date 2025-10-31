"""
Bits-per-byte (BPB) evaluation for H-Net models.

This script evaluates H-Net models on the bits-per-byte metric,
which is the standard evaluation for byte-level language models.
"""

import argparse
import json
import math
from typing import Any, Dict

import torch
from datasets import load_dataset
from tqdm import tqdm

from evaluation.utils_eval import get_device, load_model_for_eval
from hnet.utils.tokenizers import ByteTokenizer


def compute_bpb(
    model,
    dataset,
    tokenizer: ByteTokenizer,
    max_samples: int = 100,
    max_seq_length: int = 8192,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, Any]:
    """
    Compute bits-per-byte on a dataset.

    Args:
        model: H-Net model
        dataset: HuggingFace dataset
        tokenizer: ByteTokenizer instance
        max_samples: Maximum number of samples to evaluate
        max_seq_length: Maximum sequence length
        device: Device to run on

    Returns:
        Dictionary with BPB, perplexity, and other metrics
    """
    model.eval()

    total_loss = 0.0
    total_bytes = 0
    num_samples = 0

    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, desc="Evaluating BPB", total=max_samples)):
            if i >= max_samples:
                break

            text = example["text"]
            if not text or len(text.strip()) == 0:
                continue

            # Encode text to bytes
            encoded = tokenizer.encode([text], add_bos=True)[0]
            input_ids = encoded["input_ids"]

            # Truncate if necessary
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]

            # Skip very short sequences
            if len(input_ids) < 2:
                continue

            # Convert to tensor
            input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

            # Forward pass
            try:
                output = model(input_tensor)
                logits = output.logits

                # Compute cross-entropy loss
                # Shift logits and input_ids for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_tensor[:, 1:].contiguous()

                # Compute loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # Track statistics
                num_bytes = shift_labels.numel()
                total_loss += loss.item()
                total_bytes += num_bytes
                num_samples += 1

            except RuntimeError as e:
                print(f"Warning: Skipping sample {i} due to error: {e}")
                continue

    # Compute metrics
    if total_bytes == 0:
        raise ValueError("No valid samples processed")

    avg_loss_per_byte = total_loss / total_bytes
    bpb = avg_loss_per_byte / math.log(2)  # Convert from nats to bits
    perplexity = math.exp(avg_loss_per_byte)

    results = {
        "bits_per_byte": bpb,
        "perplexity": perplexity,
        "avg_loss_per_byte": avg_loss_per_byte,
        "total_bytes": total_bytes,
        "num_samples": num_samples,
    }

    return results


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate H-Net model on bits-per-byte metric")

    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to model configuration JSON"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset name",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")

    args = parser.parse_args()

    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model_for_eval(
        args.model_path, args.config_path, device=str(device), dtype=torch.bfloat16
    )
    print("Model loaded successfully")

    # Load dataset
    print(f"Loading dataset {args.dataset} (split: {args.split})")
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,  # Use streaming for large datasets
        trust_remote_code=True,
    )
    print("Dataset loaded successfully")

    # Initialize tokenizer
    tokenizer = ByteTokenizer()

    # Evaluate
    print(f"Starting BPB evaluation on {args.max_samples} samples...")
    results = compute_bpb(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
        device=device,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Bits per byte (BPB):     {results['bits_per_byte']:.4f}")
    print(f"Perplexity:              {results['perplexity']:.4f}")
    print(f"Avg loss per byte:       {results['avg_loss_per_byte']:.4f}")
    print(f"Total bytes evaluated:   {results['total_bytes']:,}")
    print(f"Number of samples:       {results['num_samples']}")
    print("=" * 60)

    # Save results if output path provided
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
