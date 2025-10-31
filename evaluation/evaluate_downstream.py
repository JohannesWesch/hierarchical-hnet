"""
Downstream zero-shot task evaluation for H-Net models.

This script evaluates H-Net models on standard NLP benchmarks
using the lm-evaluation-harness library.
"""

import argparse
import json
from typing import List

import lm_eval

from evaluation.hnet_lm_wrapper import HNetLM


def run_downstream_eval(
    model_path: str,
    config_path: str,
    tasks: List[str],
    output_path: str = "eval_results.json",
    batch_size: int = 1,
    device: str = "cuda",
    limit: int = None,
) -> dict:
    """
    Run lm-eval harness on specified tasks.

    Args:
        model_path: Path to model checkpoint
        config_path: Path to model config
        tasks: List of task names
        output_path: Path to save results
        batch_size: Batch size for evaluation
        device: Device to use
        limit: Limit number of examples per task (for testing)

    Returns:
        Dictionary of evaluation results
    """
    print(f"Initializing H-Net model from {model_path}")
    model = HNetLM(
        model_path=model_path,
        config_path=config_path,
        device=device,
        batch_size=batch_size,
    )
    print("Model initialized successfully")

    print(f"\nRunning evaluation on tasks: {', '.join(tasks)}")
    print(f"Batch size: {batch_size}")
    if limit:
        print(f"Limiting to {limit} examples per task")

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=batch_size,
        device=device,
        limit=limit,
    )

    # Save results
    print(f"\nSaving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_results(results: dict):
    """
    Pretty print evaluation results.

    Args:
        results: Results dictionary from lm_eval
    """
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)

    if "results" in results:
        for task_name, task_results in results["results"].items():
            print(f"\n{task_name}:")
            print("-" * 70)
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, float):
                    print(f"  {metric_name:30s}: {metric_value:.4f}")
                else:
                    print(f"  {metric_name:30s}: {metric_value}")

    print("=" * 70)


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate H-Net model on downstream zero-shot tasks"
    )

    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to model configuration JSON"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa",
        help="Comma-separated list of tasks to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="downstream_results.json",
        help="Path to save results JSON",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for testing)",
    )

    args = parser.parse_args()

    # Parse tasks
    tasks = [task.strip() for task in args.tasks.split(",")]

    # Run evaluation
    results = run_downstream_eval(
        model_path=args.model_path,
        config_path=args.config_path,
        tasks=tasks,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )

    # Print results
    print_results(results)

    print(f"\n✓ Evaluation complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
