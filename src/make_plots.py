"""
Plotting script for DP training results.

Generates privacy-utility and privacy-leakage frontier plots
from sweep results.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_results(results_path: str) -> pd.DataFrame:
    """Load results CSV file."""
    df = pd.read_csv(results_path)
    return df


def plot_privacy_utility_frontier(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot privacy-utility frontier: epsilon vs test accuracy.

    Args:
        df: Results DataFrame
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = df["dataset"].unique()

    for dataset in datasets:
        # Get DP results for this dataset
        dp_data = df[(df["dataset"] == dataset) & (df["model_type"] == "dp")]
        dp_data = dp_data.sort_values("epsilon")

        # Get baseline for this dataset
        baseline = df[(df["dataset"] == dataset) & (df["model_type"] == "baseline")]

        # Plot DP frontier
        if len(dp_data) > 0:
            ax.plot(dp_data["epsilon"], dp_data["test_accuracy"],
                    marker="o", label=f"{dataset} (DP)", linewidth=2, markersize=8)

        # Plot baseline as horizontal line
        if len(baseline) > 0:
            baseline_acc = baseline["test_accuracy"].values[0]
            ax.axhline(y=baseline_acc, linestyle="--", alpha=0.7,
                       label=f"{dataset} (baseline)")

    ax.set_xlabel("Privacy Budget (ε)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Privacy-Utility Frontier", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "privacy_utility_frontier.png"),
                dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "privacy_utility_frontier.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    print(f"Saved privacy-utility frontier plots to {output_dir}")


def plot_privacy_leakage_frontier(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot privacy-leakage frontier: epsilon vs MIA AUC.

    Args:
        df: Results DataFrame
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = df["dataset"].unique()

    for dataset in datasets:
        # Get DP results for this dataset
        dp_data = df[(df["dataset"] == dataset) & (df["model_type"] == "dp")]
        dp_data = dp_data.sort_values("epsilon")

        # Get baseline for this dataset
        baseline = df[(df["dataset"] == dataset) & (df["model_type"] == "baseline")]

        # Plot DP frontier
        if len(dp_data) > 0:
            ax.plot(dp_data["epsilon"], dp_data["mia_threshold_auc"],
                    marker="s", label=f"{dataset} (DP)", linewidth=2, markersize=8)

        # Plot baseline as horizontal line
        if len(baseline) > 0:
            baseline_auc = baseline["mia_threshold_auc"].values[0]
            ax.axhline(y=baseline_auc, linestyle="--", alpha=0.7,
                       label=f"{dataset} (baseline)")

    # Add reference line at 0.5 (random guessing)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random (0.5)")

    ax.set_xlabel("Privacy Budget (ε)", fontsize=12)
    ax.set_ylabel("MIA AUC", fontsize=12)
    ax.set_title("Privacy Leakage Frontier", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "privacy_leakage_frontier.png"),
                dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "privacy_leakage_frontier.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    print(f"Saved privacy-leakage frontier plots to {output_dir}")


def plot_combined_frontier(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot combined figure with both frontiers side by side.

    Args:
        df: Results DataFrame
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    datasets = df["dataset"].unique()

    # Left plot: Privacy-Utility
    ax = axes[0]
    for dataset in datasets:
        dp_data = df[(df["dataset"] == dataset) & (df["model_type"] == "dp")]
        dp_data = dp_data.sort_values("epsilon")
        baseline = df[(df["dataset"] == dataset) & (df["model_type"] == "baseline")]

        if len(dp_data) > 0:
            ax.plot(dp_data["epsilon"], dp_data["test_accuracy"],
                    marker="o", label=f"{dataset}", linewidth=2, markersize=8)

        if len(baseline) > 0:
            baseline_acc = baseline["test_accuracy"].values[0]
            ax.axhline(y=baseline_acc, linestyle="--", alpha=0.5)

    ax.set_xlabel("Privacy Budget (ε)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Privacy-Utility Frontier", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    # Right plot: Privacy-Leakage
    ax = axes[1]
    for dataset in datasets:
        dp_data = df[(df["dataset"] == dataset) & (df["model_type"] == "dp")]
        dp_data = dp_data.sort_values("epsilon")
        baseline = df[(df["dataset"] == dataset) & (df["model_type"] == "baseline")]

        if len(dp_data) > 0:
            ax.plot(dp_data["epsilon"], dp_data["mia_threshold_auc"],
                    marker="s", label=f"{dataset}", linewidth=2, markersize=8)

        if len(baseline) > 0:
            baseline_auc = baseline["mia_threshold_auc"].values[0]
            ax.axhline(y=baseline_auc, linestyle="--", alpha=0.5)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
    ax.set_xlabel("Privacy Budget (ε)", fontsize=12)
    ax.set_ylabel("MIA AUC", fontsize=12)
    ax.set_title("Privacy Leakage Frontier", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    plt.tight_layout()

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "combined_frontier.png"),
                dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "combined_frontier.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined frontier plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from sweep results")
    parser.add_argument("--results", type=str, default="results/results.csv",
                        help="Path to results CSV file")
    parser.add_argument("--output-dir", type=str, default="results/figures",
                        help="Output directory for plots")

    args = parser.parse_args()

    # Load results
    if not os.path.exists(args.results):
        print(f"Error: Results file not found at {args.results}")
        print("Run run_sweep.py first to generate results.")
        return

    print(f"Loading results from {args.results}")
    df = load_results(args.results)

    print(f"Found {len(df)} result rows")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Model types: {df['model_type'].unique()}")

    # Generate plots
    plot_privacy_utility_frontier(df, args.output_dir)
    plot_privacy_leakage_frontier(df, args.output_dir)
    plot_combined_frontier(df, args.output_dir)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
