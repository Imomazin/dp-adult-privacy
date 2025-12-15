"""
Parameter sweep script for DP training experiments.

Runs DP-SGD training across multiple epsilon values and datasets,
then evaluates with membership inference attacks and saves results.
"""

import argparse
import os
import csv
from datetime import datetime

import torch

from data_loader import get_data_loaders
from model import create_model, count_parameters
from train_dp import train_dp
from train_baseline import train_baseline
from mia_evaluation import run_mia_evaluation


# Default sweep parameters
DEFAULT_EPSILONS = [2, 4, 8, 16]
DEFAULT_DATASETS = ["adult", "bank"]
DEFAULT_SEED = 42


def run_sweep(
    datasets: list[str] = None,
    epsilons: list[float] = None,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    hidden_dim: int = 128,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    seed: int = DEFAULT_SEED,
    output_dir: str = "results",
    checkpoint_dir: str = "checkpoints"
) -> list[dict]:
    """
    Run parameter sweep across datasets and epsilon values.

    Args:
        datasets: List of dataset names to sweep
        epsilons: List of epsilon values to sweep
        epochs: Training epochs per run
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_dim: Model hidden dimension
        delta: Delta for DP guarantee
        max_grad_norm: Gradient clipping norm
        seed: Random seed for reproducibility
        output_dir: Directory to save results
        checkpoint_dir: Directory for model checkpoints

    Returns:
        List of result dictionaries
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS
    if epsilons is None:
        epsilons = DEFAULT_EPSILONS

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    results = []
    total_runs = len(datasets) * (len(epsilons) + 1)  # +1 for baseline per dataset
    current_run = 0

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*60}")

        # Train baseline model first
        current_run += 1
        print(f"\n[{current_run}/{total_runs}] Training baseline for {dataset}...")

        baseline_save_dir = os.path.join(checkpoint_dir, f"{dataset}_baseline")
        baseline_results = train_baseline(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            seed=seed,
            save_dir=baseline_save_dir
        )

        baseline_path = os.path.join(baseline_save_dir, "baseline_final.pt")

        # Run MIA on baseline
        baseline_mia = run_mia_evaluation(
            dataset=dataset,
            baseline_path=baseline_path,
            dp_path=None,
            batch_size=batch_size,
            seed=seed
        )

        # Record baseline result
        baseline_record = {
            "dataset": dataset,
            "model_type": "baseline",
            "epsilon": None,
            "delta": None,
            "test_accuracy": baseline_results["test_accuracy"],
            "best_val_accuracy": baseline_results["best_val_accuracy"],
            "mia_threshold_auc": baseline_mia["baseline"]["threshold_attack"]["auc"],
            "mia_threshold_accuracy": baseline_mia["baseline"]["threshold_attack"]["accuracy"],
            "mia_threshold_advantage": baseline_mia["baseline"]["threshold_attack"]["advantage"],
            "mia_loss_auc": baseline_mia["baseline"]["loss_attack"]["auc"],
            "seed": seed,
            "epochs": epochs
        }
        results.append(baseline_record)

        # Sweep over epsilon values
        for epsilon in epsilons:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Training DP (ε={epsilon}) for {dataset}...")

            dp_save_dir = os.path.join(checkpoint_dir, f"{dataset}_dp_eps{epsilon}")
            dp_results = train_dp(
                dataset=dataset,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                hidden_dim=hidden_dim,
                target_epsilon=epsilon,
                target_delta=delta,
                max_grad_norm=max_grad_norm,
                seed=seed,
                save_dir=dp_save_dir
            )

            dp_path = os.path.join(dp_save_dir, "dp_final.pt")

            # Run MIA on DP model
            dp_mia = run_mia_evaluation(
                dataset=dataset,
                baseline_path=None,
                dp_path=dp_path,
                batch_size=batch_size,
                seed=seed
            )

            # Record DP result
            dp_record = {
                "dataset": dataset,
                "model_type": "dp",
                "epsilon": dp_results["final_epsilon"],
                "delta": delta,
                "test_accuracy": dp_results["test_accuracy"],
                "best_val_accuracy": dp_results["best_val_accuracy"],
                "mia_threshold_auc": dp_mia["dp"]["threshold_attack"]["auc"],
                "mia_threshold_accuracy": dp_mia["dp"]["threshold_attack"]["accuracy"],
                "mia_threshold_advantage": dp_mia["dp"]["threshold_attack"]["advantage"],
                "mia_loss_auc": dp_mia["dp"]["loss_attack"]["auc"],
                "seed": seed,
                "epochs": epochs
            }
            results.append(dp_record)

    return results


def save_results(results: list[dict], output_path: str) -> None:
    """Save results to CSV file."""
    if not results:
        print("No results to save.")
        return

    fieldnames = [
        "dataset", "model_type", "epsilon", "delta",
        "test_accuracy", "best_val_accuracy",
        "mia_threshold_auc", "mia_threshold_accuracy", "mia_threshold_advantage",
        "mia_loss_auc", "seed", "epochs"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")


def print_summary(results: list[dict]) -> None:
    """Print summary table of results."""
    print("\n" + "="*80)
    print("SWEEP RESULTS SUMMARY")
    print("="*80)
    print(f"{'Dataset':<10} {'Model':<12} {'Epsilon':<10} {'Test Acc':<12} {'MIA AUC':<10}")
    print("-"*80)

    for r in results:
        eps_str = f"{r['epsilon']:.1f}" if r['epsilon'] else "N/A"
        print(f"{r['dataset']:<10} {r['model_type']:<12} {eps_str:<10} "
              f"{r['test_accuracy']:.4f}      {r['mia_threshold_auc']:.4f}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Run DP training sweep")
    parser.add_argument("--datasets", type=str, nargs="+", default=DEFAULT_DATASETS,
                        choices=["adult", "bank"], help="Datasets to sweep")
    parser.add_argument("--epsilons", type=float, nargs="+", default=DEFAULT_EPSILONS,
                        help="Epsilon values to sweep (default: 2 4 8 16)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--delta", type=float, default=1e-5, help="Delta for DP")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")

    args = parser.parse_args()

    print("="*60)
    print("DP TRAINING PARAMETER SWEEP")
    print("="*60)
    print(f"Datasets: {args.datasets}")
    print(f"Epsilons: {args.epsilons}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: {args.seed}")
    print("="*60)

    # Run sweep
    results = run_sweep(
        datasets=args.datasets,
        epsilons=args.epsilons,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir
    )

    # Save results
    output_path = os.path.join(args.output_dir, "results.csv")
    save_results(results, output_path)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
