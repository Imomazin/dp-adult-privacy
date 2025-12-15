"""
Plotting script for privacy-preserving ML experiments.
Generates privacy-utility and privacy-leakage frontier plots from results.

Reads results from results/results.csv and generates:
- Privacy-utility frontier (epsilon vs test AUC/accuracy)
- Privacy-leakage frontier (epsilon vs MIA AUC)
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def load_results(results_path: str = "results/results.csv") -> pd.DataFrame:
    """
    Load experiment results from CSV file.

    Expected columns:
    - epsilon: Privacy budget (0 for baseline)
    - test_accuracy: Test set accuracy
    - test_auc_roc: Test set AUC-ROC
    - mia_auc: Membership inference attack AUC
    - dataset: Dataset name (optional)
    - model_type: "baseline" or "dp" (optional)

    Args:
        results_path: Path to results CSV file

    Returns:
        DataFrame with results
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Results file not found: {results_path}\n"
            "Run experiments first to generate results."
        )

    df = pd.read_csv(results_path)

    # Ensure required columns exist
    required_cols = ["epsilon"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def plot_privacy_utility_frontier(df: pd.DataFrame, output_dir: str = "results/figures",
                                   metric: str = "test_auc_roc"):
    """
    Plot privacy-utility frontier (epsilon vs test performance).

    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
        metric: Metric to plot ("test_auc_roc" or "test_accuracy")
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Get unique datasets if present
    if "dataset" in df.columns:
        datasets = df["dataset"].unique()
    else:
        datasets = ["default"]
        df = df.copy()
        df["dataset"] = "default"

    # Plot for each dataset
    for dataset in datasets:
        subset = df[df["dataset"] == dataset].copy()

        # Sort by epsilon for line plot
        subset = subset.sort_values("epsilon")

        # Handle baseline (epsilon=0 or inf)
        dp_data = subset[subset["epsilon"] > 0]

        if metric in subset.columns:
            ax.plot(dp_data["epsilon"], dp_data[metric],
                    marker='o', label=f"{dataset}")

            # Add baseline point if available
            baseline = subset[subset["epsilon"] == 0]
            if len(baseline) > 0 and metric in baseline.columns:
                ax.axhline(y=baseline[metric].values[0], linestyle='--',
                          alpha=0.7, label=f"{dataset} (baseline)")

    metric_label = "AUC-ROC" if "auc" in metric.lower() else "Accuracy"
    ax.set_xlabel("Privacy Budget (epsilon)", fontsize=12)
    ax.set_ylabel(f"Test {metric_label}", fontsize=12)
    ax.set_title("Privacy-Utility Frontier", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set x-axis to log scale for better visualization
    if len(df[df["epsilon"] > 0]) > 0:
        ax.set_xscale('log')

    # Save plots
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "privacy_utility_frontier.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "privacy_utility_frontier.pdf"))
    plt.close()

    print(f"Saved privacy-utility frontier to {output_dir}/")


def plot_privacy_leakage_frontier(df: pd.DataFrame, output_dir: str = "results/figures"):
    """
    Plot privacy-leakage frontier (epsilon vs MIA AUC).

    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check for MIA columns
    mia_cols = [c for c in df.columns if "mia" in c.lower()]
    if not mia_cols:
        print("No MIA columns found in results. Skipping privacy-leakage plot.")
        return

    mia_col = mia_cols[0]  # Use first MIA column found

    fig, ax = plt.subplots(figsize=(8, 6))

    # Get unique datasets if present
    if "dataset" in df.columns:
        datasets = df["dataset"].unique()
    else:
        datasets = ["default"]
        df = df.copy()
        df["dataset"] = "default"

    # Plot for each dataset
    for dataset in datasets:
        subset = df[df["dataset"] == dataset].copy()
        subset = subset.sort_values("epsilon")

        # Only plot DP results (epsilon > 0)
        dp_data = subset[subset["epsilon"] > 0]

        if mia_col in dp_data.columns and len(dp_data) > 0:
            ax.plot(dp_data["epsilon"], dp_data[mia_col],
                    marker='s', label=f"{dataset}")

            # Add baseline point
            baseline = subset[subset["epsilon"] == 0]
            if len(baseline) > 0 and mia_col in baseline.columns:
                ax.axhline(y=baseline[mia_col].values[0], linestyle='--',
                          alpha=0.7, label=f"{dataset} (baseline)")

    # Add random guessing line
    ax.axhline(y=0.5, linestyle=':', color='gray', alpha=0.7, label="Random guess")

    ax.set_xlabel("Privacy Budget (epsilon)", fontsize=12)
    ax.set_ylabel("MIA Attack AUC", fontsize=12)
    ax.set_title("Privacy-Leakage Frontier", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set x-axis to log scale
    if len(df[df["epsilon"] > 0]) > 0:
        ax.set_xscale('log')

    # Save plots
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "privacy_leakage_frontier.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "privacy_leakage_frontier.pdf"))
    plt.close()

    print(f"Saved privacy-leakage frontier to {output_dir}/")


def plot_combined_frontier(df: pd.DataFrame, output_dir: str = "results/figures"):
    """
    Plot combined privacy-utility and privacy-leakage in subplots.

    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check available metrics
    has_utility = "test_auc_roc" in df.columns or "test_accuracy" in df.columns
    mia_cols = [c for c in df.columns if "mia" in c.lower()]
    has_mia = len(mia_cols) > 0

    if not has_utility and not has_mia:
        print("No plottable metrics found in results.")
        return

    n_plots = int(has_utility) + int(has_mia)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Get datasets
    if "dataset" in df.columns:
        datasets = df["dataset"].unique()
    else:
        datasets = ["default"]
        df = df.copy()
        df["dataset"] = "default"

    # Utility plot
    if has_utility:
        ax = axes[plot_idx]
        metric = "test_auc_roc" if "test_auc_roc" in df.columns else "test_accuracy"

        for dataset in datasets:
            subset = df[df["dataset"] == dataset].sort_values("epsilon")
            dp_data = subset[subset["epsilon"] > 0]

            if metric in dp_data.columns:
                ax.plot(dp_data["epsilon"], dp_data[metric], marker='o', label=dataset)

                baseline = subset[subset["epsilon"] == 0]
                if len(baseline) > 0:
                    ax.axhline(y=baseline[metric].values[0], linestyle='--', alpha=0.5)

        metric_label = "AUC-ROC" if "auc" in metric else "Accuracy"
        ax.set_xlabel("Epsilon")
        ax.set_ylabel(f"Test {metric_label}")
        ax.set_title("Privacy-Utility Tradeoff")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if len(df[df["epsilon"] > 0]) > 0:
            ax.set_xscale('log')
        plot_idx += 1

    # MIA plot
    if has_mia:
        ax = axes[plot_idx]
        mia_col = mia_cols[0]

        for dataset in datasets:
            subset = df[df["dataset"] == dataset].sort_values("epsilon")
            dp_data = subset[subset["epsilon"] > 0]

            if mia_col in dp_data.columns and len(dp_data) > 0:
                ax.plot(dp_data["epsilon"], dp_data[mia_col], marker='s', label=dataset)

                baseline = subset[subset["epsilon"] == 0]
                if len(baseline) > 0:
                    ax.axhline(y=baseline[mia_col].values[0], linestyle='--', alpha=0.5)

        ax.axhline(y=0.5, linestyle=':', color='gray', alpha=0.7, label="Random")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("MIA AUC")
        ax.set_title("Privacy-Leakage Tradeoff")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if len(df[df["epsilon"] > 0]) > 0:
            ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_frontier.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "combined_frontier.pdf"))
    plt.close()

    print(f"Saved combined frontier plot to {output_dir}/")


def create_sample_results():
    """
    Create a sample results.csv file for testing.
    """
    os.makedirs("results", exist_ok=True)

    # Sample data
    data = {
        "epsilon": [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        "test_accuracy": [0.85, 0.70, 0.78, 0.80, 0.82, 0.83, 0.84],
        "test_auc_roc": [0.90, 0.72, 0.82, 0.85, 0.87, 0.88, 0.89],
        "mia_auc": [0.58, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56],
        "dataset": ["adult"] * 7,
        "model_type": ["baseline", "dp", "dp", "dp", "dp", "dp", "dp"],
    }

    df = pd.DataFrame(data)
    df.to_csv("results/results.csv", index=False)
    print("Created sample results at results/results.csv")
    return df


def main():
    """Command-line interface for plotting."""
    parser = argparse.ArgumentParser(
        description="Generate privacy frontier plots from experiment results"
    )
    parser.add_argument("--results", type=str, default="results/results.csv",
                        help="Path to results CSV file")
    parser.add_argument("--output", type=str, default="results/figures",
                        help="Output directory for plots")
    parser.add_argument("--create-sample", action="store_true",
                        help="Create sample results.csv for testing")
    args = parser.parse_args()

    if args.create_sample:
        df = create_sample_results()
    else:
        df = load_results(args.results)

    print(f"Loaded {len(df)} result rows")
    print(f"Columns: {list(df.columns)}")

    # Generate all plots
    plot_privacy_utility_frontier(df, args.output)
    plot_privacy_leakage_frontier(df, args.output)
    plot_combined_frontier(df, args.output)

    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
