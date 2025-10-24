"""
Evaluation script for H-Net language models.

This script evaluates trained H-Net models on validation/test datasets.
Computes metrics like perplexity, token accuracy, etc.
"""

import argparse
import json
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.models.mixer_seq import HNetForCausalLM
from training.data import PackedDataCollator
from training.losses import get_routing_statistics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate H-Net language model")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to model configuration JSON",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to evaluation data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )

    return parser.parse_args()


def load_model(
    model_path: str,
    config_path: str,
    device: torch.device,
) -> HNetForCausalLM:
    """
    Load model from checkpoint.

    Args:
        model_path: Path to checkpoint
        config_path: Path to config JSON
        device: Device to load to

    Returns:
        Loaded model
    """
    # Load configuration
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Create config objects
    attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    # Create model
    model = HNetForCausalLM(hnet_cfg, device=device, dtype=torch.bfloat16)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


def evaluate_model(
    model: HNetForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate model on dataset.

    Args:
        model: H-Net model
        dataloader: Evaluation dataloader
        device: Device

    Returns:
        Dictionary of evaluation metrics
    """

    model.eval()

    all_losses = []
    all_ce_losses = []
    all_lb_losses = []
    all_routing_stats = []

    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            cu_seqlens = batch.get("cu_seqlens", None)
            batch.get("max_seqlen", None)

            if cu_seqlens is not None:
                cu_seqlens = cu_seqlens.to(device)

            # Forward pass
            output = model(
                input_ids.unsqueeze(0),
                mask=None,
            )

            logits = output.logits
            bpred_outputs = output.bpred_output

            # Compute losses
            from training.losses import HierarchicalLoadBalancingLoss, LanguageModelingLoss

            lm_loss = LanguageModelingLoss()
            lb_loss_fn = HierarchicalLoadBalancingLoss()

            ce_loss = lm_loss(logits, targets.unsqueeze(0))
            lb_result = lb_loss_fn(bpred_outputs)

            all_losses.append(ce_loss.item())
            all_ce_losses.append(ce_loss.item())
            all_lb_losses.append(lb_result["total"].item())

            # Routing statistics
            if len(bpred_outputs) > 0:
                routing_stats = get_routing_statistics(bpred_outputs)
                all_routing_stats.append(routing_stats)

            total_tokens += input_ids.numel()

    # Aggregate results
    results = {
        "avg_loss": sum(all_losses) / len(all_losses),
        "avg_ce_loss": sum(all_ce_losses) / len(all_ce_losses),
        "avg_lb_loss": sum(all_lb_losses) / len(all_lb_losses),
        "perplexity": torch.exp(torch.tensor(sum(all_ce_losses) / len(all_ce_losses))).item(),
        "total_tokens": total_tokens,
        "num_batches": len(all_losses),
    }

    # Average routing statistics
    if all_routing_stats:
        avg_routing_stats = {}
        for key in all_routing_stats[0].keys():
            values = [
                stats[key].item() if torch.is_tensor(stats[key]) else stats[key]
                for stats in all_routing_stats
                if key in stats
            ]
            avg_routing_stats[key] = sum(values) / len(values)
        results["routing_stats"] = avg_routing_stats

    return results


def compute_detailed_metrics(
    all_losses: list,
    all_logits: list,
    all_targets: list,
    all_routing_outputs: list,
) -> Dict[str, float]:
    """
    Compute detailed evaluation metrics.

    Args:
        all_losses: List of loss values
        all_logits: List of model predictions
        all_targets: List of targets
        all_routing_outputs: List of routing outputs

    Returns:
        Dictionary of metrics
    """
    # Average loss
    avg_loss = sum(all_losses) / len(all_losses)

    # Perplexity
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    metrics = {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
    }

    return metrics


def save_results(
    results: Dict[str, Any],
    output_path: str,
):
    """
    Save evaluation results to file.

    Args:
        results: Results dictionary
        output_path: Path to save to
    """

    # Convert any tensors to floats
    def convert_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    serializable_results = convert_to_serializable(results)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {output_path}")

    # Pretty print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in serializable_results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


def main():
    """Main evaluation entry point."""
    # Parse arguments
    args = parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.config_path, device)
    print("Model loaded successfully")

    # Setup data
    print(f"Loading evaluation data from {args.data_path}")
    from training.data import TextDataset

    dataset = TextDataset(
        args.data_path,
        max_length=args.max_seq_length,
        add_bos=True,
    )

    collator = PackedDataCollator(max_seq_length=args.max_seq_length)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False,
    )

    print(f"Loaded {len(dataset)} examples")

    # Evaluate
    print("Starting evaluation...")
    results = evaluate_model(model, dataloader, device)

    # Save results
    if args.output_path:
        save_results(results, args.output_path)
    else:
        print("\nEvaluation Results:")
        print("=" * 50)
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            else:
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
