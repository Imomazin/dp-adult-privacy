"""
Fairness evaluation for privacy-preserving ML models.
Computes subgroup metrics on Adult dataset by sex and race.

Metrics computed:
- Subgroup AUC/accuracy
- TPR gap (equal opportunity difference)
- Comparison between baseline and DP models at various epsilon values
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

import config
from data import get_dataset_with_sensitive_attrs, TabularDataset
from model import create_model


def compute_subgroup_metrics(labels: np.ndarray, probs: np.ndarray,
                             sensitive_attr: np.ndarray, attr_name: str):
    """
    Compute metrics for each subgroup defined by a sensitive attribute.

    Args:
        labels: True labels (N,)
        probs: Predicted probabilities (N,)
        sensitive_attr: Binary sensitive attribute values (N,)
        attr_name: Name of the sensitive attribute

    Returns:
        metrics: Dictionary with subgroup metrics
    """
    predictions = (probs > 0.5).astype(int)

    # Split by sensitive attribute
    group_0_mask = sensitive_attr == 0
    group_1_mask = sensitive_attr == 1

    metrics = {}

    # Overall metrics
    metrics["overall_accuracy"] = accuracy_score(labels, predictions)
    try:
        metrics["overall_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["overall_auc"] = 0.5

    # Group 0 metrics
    if group_0_mask.sum() > 0:
        g0_labels = labels[group_0_mask]
        g0_probs = probs[group_0_mask]
        g0_preds = predictions[group_0_mask]

        metrics[f"{attr_name}_0_accuracy"] = accuracy_score(g0_labels, g0_preds)
        try:
            metrics[f"{attr_name}_0_auc"] = roc_auc_score(g0_labels, g0_probs)
        except ValueError:
            metrics[f"{attr_name}_0_auc"] = 0.5

        # TPR for group 0 (only on positive samples)
        g0_pos_mask = g0_labels == 1
        if g0_pos_mask.sum() > 0:
            metrics[f"{attr_name}_0_tpr"] = g0_preds[g0_pos_mask].mean()
        else:
            metrics[f"{attr_name}_0_tpr"] = 0.0

        metrics[f"{attr_name}_0_count"] = int(group_0_mask.sum())

    # Group 1 metrics
    if group_1_mask.sum() > 0:
        g1_labels = labels[group_1_mask]
        g1_probs = probs[group_1_mask]
        g1_preds = predictions[group_1_mask]

        metrics[f"{attr_name}_1_accuracy"] = accuracy_score(g1_labels, g1_preds)
        try:
            metrics[f"{attr_name}_1_auc"] = roc_auc_score(g1_labels, g1_probs)
        except ValueError:
            metrics[f"{attr_name}_1_auc"] = 0.5

        # TPR for group 1
        g1_pos_mask = g1_labels == 1
        if g1_pos_mask.sum() > 0:
            metrics[f"{attr_name}_1_tpr"] = g1_preds[g1_pos_mask].mean()
        else:
            metrics[f"{attr_name}_1_tpr"] = 0.0

        metrics[f"{attr_name}_1_count"] = int(group_1_mask.sum())

    # TPR gap (equal opportunity difference)
    if f"{attr_name}_0_tpr" in metrics and f"{attr_name}_1_tpr" in metrics:
        metrics[f"{attr_name}_tpr_gap"] = abs(
            metrics[f"{attr_name}_1_tpr"] - metrics[f"{attr_name}_0_tpr"]
        )

    # Accuracy gap
    if f"{attr_name}_0_accuracy" in metrics and f"{attr_name}_1_accuracy" in metrics:
        metrics[f"{attr_name}_accuracy_gap"] = abs(
            metrics[f"{attr_name}_1_accuracy"] - metrics[f"{attr_name}_0_accuracy"]
        )

    return metrics


def evaluate_model_fairness(model, test_features: np.ndarray, test_labels: np.ndarray,
                            test_sensitive: dict, device):
    """
    Evaluate model fairness across all sensitive attributes.

    Args:
        model: Trained model
        test_features: Test feature matrix
        test_labels: Test labels
        test_sensitive: Dictionary of sensitive attribute arrays
        device: Device to run on

    Returns:
        metrics: Dictionary with all fairness metrics
    """
    model.eval()

    # Get predictions
    with torch.no_grad():
        features_t = torch.tensor(test_features, dtype=torch.float32).to(device)
        logits = model(features_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    all_metrics = {}

    # Compute metrics for each sensitive attribute
    for attr_name, attr_values in test_sensitive.items():
        attr_metrics = compute_subgroup_metrics(
            test_labels, probs, attr_values, attr_name
        )
        all_metrics.update(attr_metrics)

    return all_metrics


def train_model_for_fairness(train_features: np.ndarray, train_labels: np.ndarray,
                             input_dim: int, epochs: int, device,
                             use_dp: bool = False, epsilon: float = None):
    """
    Train a model for fairness evaluation.

    Args:
        train_features: Training features
        train_labels: Training labels
        input_dim: Input feature dimension
        epochs: Number of epochs
        device: Device to train on
        use_dp: Whether to use differential privacy
        epsilon: Privacy budget if using DP

    Returns:
        model: Trained model
    """
    from torch.utils.data import TensorDataset, DataLoader

    model = create_model(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Create DataLoader
    features_t = torch.tensor(train_features, dtype=torch.float32)
    labels_t = torch.tensor(train_labels, dtype=torch.float32)
    dataset = TensorDataset(features_t, labels_t)
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    if use_dp:
        from opacus import PrivacyEngine

        if epsilon is None:
            epsilon = config.EPSILON

        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=config.DELTA,
            max_grad_norm=config.MAX_GRAD_NORM,
        )

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    if use_dp:
        return model._module
    return model


def run_fairness_evaluation(epsilons: list = None, epochs: int = None,
                            output_path: str = "results/fairness.csv",
                            verbose: bool = True):
    """
    Run fairness evaluation for baseline and DP models at various epsilon values.

    Args:
        epsilons: List of epsilon values to evaluate
        epochs: Training epochs
        output_path: Path to save results CSV
        verbose: Whether to print progress

    Returns:
        results_df: DataFrame with all fairness results
    """
    if epsilons is None:
        epsilons = [0.5, 1.0, 2.0, 5.0, 10.0]
    if epochs is None:
        epochs = config.EPOCHS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    if verbose:
        print(f"Using device: {device}")
        print("\nLoading Adult dataset with sensitive attributes...")

    # Load data with sensitive attributes
    (train_features, train_labels, test_features, test_labels,
     train_sensitive, test_sensitive, input_dim) = get_dataset_with_sensitive_attrs("adult")

    if verbose:
        print(f"Training samples: {len(train_labels)}")
        print(f"Test samples: {len(test_labels)}")
        print(f"Sensitive attributes: {list(test_sensitive.keys())}")

    all_results = []

    # Evaluate baseline model
    if verbose:
        print("\n" + "="*60)
        print("Training and evaluating BASELINE model...")
        print("="*60)

    baseline_model = train_model_for_fairness(
        train_features, train_labels, input_dim, epochs, device,
        use_dp=False
    )

    baseline_metrics = evaluate_model_fairness(
        baseline_model, test_features, test_labels, test_sensitive, device
    )
    baseline_metrics["model_type"] = "baseline"
    baseline_metrics["epsilon"] = 0.0
    all_results.append(baseline_metrics)

    if verbose:
        print(f"\nBaseline Results:")
        print(f"  Overall Accuracy: {baseline_metrics['overall_accuracy']:.4f}")
        print(f"  Overall AUC: {baseline_metrics['overall_auc']:.4f}")
        print(f"  Sex TPR Gap: {baseline_metrics.get('sex_tpr_gap', 'N/A'):.4f}")
        print(f"  Race TPR Gap: {baseline_metrics.get('race_tpr_gap', 'N/A'):.4f}")

    # Evaluate DP models at various epsilon values
    for eps in epsilons:
        if verbose:
            print("\n" + "="*60)
            print(f"Training and evaluating DP model (epsilon={eps})...")
            print("="*60)

        dp_model = train_model_for_fairness(
            train_features, train_labels, input_dim, epochs, device,
            use_dp=True, epsilon=eps
        )

        dp_metrics = evaluate_model_fairness(
            dp_model, test_features, test_labels, test_sensitive, device
        )
        dp_metrics["model_type"] = "dp"
        dp_metrics["epsilon"] = eps
        all_results.append(dp_metrics)

        if verbose:
            print(f"\nDP (eps={eps}) Results:")
            print(f"  Overall Accuracy: {dp_metrics['overall_accuracy']:.4f}")
            print(f"  Overall AUC: {dp_metrics['overall_auc']:.4f}")
            print(f"  Sex TPR Gap: {dp_metrics.get('sex_tpr_gap', 'N/A'):.4f}")
            print(f"  Race TPR Gap: {dp_metrics.get('race_tpr_gap', 'N/A'):.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)

    if verbose:
        print("\n" + "="*60)
        print("FAIRNESS EVALUATION SUMMARY")
        print("="*60)
        print(f"\nResults saved to {output_path}")

        # Print summary table
        summary_cols = ["model_type", "epsilon", "overall_accuracy", "overall_auc",
                       "sex_tpr_gap", "race_tpr_gap"]
        available_cols = [c for c in summary_cols if c in results_df.columns]

        print("\n" + results_df[available_cols].to_string(index=False))

        # Analyze fairness trends
        print("\n" + "="*60)
        print("FAIRNESS ANALYSIS")
        print("="*60)

        baseline_row = results_df[results_df["model_type"] == "baseline"].iloc[0]
        dp_rows = results_df[results_df["model_type"] == "dp"]

        for attr in ["sex", "race"]:
            gap_col = f"{attr}_tpr_gap"
            if gap_col in results_df.columns:
                baseline_gap = baseline_row[gap_col]
                print(f"\n{attr.upper()} TPR Gap:")
                print(f"  Baseline: {baseline_gap:.4f}")

                for _, row in dp_rows.iterrows():
                    eps = row["epsilon"]
                    dp_gap = row[gap_col]
                    change = dp_gap - baseline_gap
                    direction = "increased" if change > 0 else "decreased"
                    print(f"  DP (eps={eps}): {dp_gap:.4f} ({direction} by {abs(change):.4f})")

    return results_df


def main():
    """Command-line interface for fairness evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate model fairness across sensitive subgroups"
    )
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.5, 1.0, 2.0, 5.0, 10.0],
                        help="Epsilon values to evaluate")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Training epochs")
    parser.add_argument("--output", type=str, default="results/fairness.csv",
                        help="Output path for results CSV")
    args = parser.parse_args()

    run_fairness_evaluation(
        epsilons=args.epsilons,
        epochs=args.epochs,
        output_path=args.output,
        verbose=True
    )


if __name__ == "__main__":
    main()
