"""
Fairness evaluation script for Adult dataset.

Evaluates subgroup performance (by sex and race) and computes:
- Subgroup AUC and accuracy
- TPR gap (equal opportunity difference)

Compares baseline vs DP models at different epsilon values.
"""

import argparse
import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from model import create_model


# Adult dataset configuration (duplicated for standalone use)
ADULT_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

ADULT_COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

ADULT_CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

ADULT_NUMERICAL_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]


def load_adult_with_demographics(data_dir: str = "data"):
    """
    Load Adult dataset while preserving demographic attributes for fairness analysis.

    Returns:
        Tuple of (X_test, y_test, demographics_test) where demographics_test
        contains 'sex' and 'race' columns.
    """
    import urllib.request

    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "adult.data")
    test_path = os.path.join(data_dir, "adult.test")

    # Download if needed
    if not os.path.exists(train_path):
        urllib.request.urlretrieve(ADULT_TRAIN_URL, train_path)
    if not os.path.exists(test_path):
        urllib.request.urlretrieve(ADULT_TEST_URL, test_path)

    # Load test data
    test_df = pd.read_csv(
        test_path, names=ADULT_COLUMN_NAMES,
        sep=r",\s*", engine="python", na_values="?", skiprows=1
    )

    # Also load train for fitting scalers
    train_df = pd.read_csv(
        train_path, names=ADULT_COLUMN_NAMES,
        sep=r",\s*", engine="python", na_values="?"
    )

    # Clean income labels
    train_df["income"] = train_df["income"].str.strip().str.rstrip(".")
    test_df["income"] = test_df["income"].str.strip().str.rstrip(".")

    # Drop NA
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Save demographics before encoding
    demographics_test = test_df[["sex", "race"]].copy()

    # Encode target
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["income"])
    y_test = label_encoder.transform(test_df["income"])

    # Remove target
    train_features = train_df.drop("income", axis=1)
    test_features = test_df.drop("income", axis=1)

    # One-hot encode
    train_encoded = pd.get_dummies(train_features, columns=ADULT_CATEGORICAL_COLS)
    test_encoded = pd.get_dummies(test_features, columns=ADULT_CATEGORICAL_COLS)

    # Align columns
    train_encoded, test_encoded = train_encoded.align(
        test_encoded, join="left", axis=1, fill_value=0
    )

    # Scale numerical features
    scaler = StandardScaler()
    train_encoded[ADULT_NUMERICAL_COLS] = scaler.fit_transform(train_encoded[ADULT_NUMERICAL_COLS])
    test_encoded[ADULT_NUMERICAL_COLS] = scaler.transform(test_encoded[ADULT_NUMERICAL_COLS])

    X_test = test_encoded.values.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return X_test, y_test, demographics_test


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def get_predictions(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    """Get model predictions."""
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor).cpu().numpy().flatten()
    return outputs


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_mask: np.ndarray
) -> dict:
    """
    Compute metrics for a subgroup.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        group_mask: Boolean mask for the subgroup

    Returns:
        Dictionary with subgroup metrics
    """
    y_true_group = y_true[group_mask]
    y_pred_group = y_pred[group_mask]

    if len(y_true_group) == 0 or len(np.unique(y_true_group)) < 2:
        return {"auc": np.nan, "accuracy": np.nan, "tpr": np.nan, "n_samples": 0}

    # AUC
    auc = roc_auc_score(y_true_group, y_pred_group)

    # Accuracy
    y_pred_binary = (y_pred_group >= 0.5).astype(int)
    accuracy = accuracy_score(y_true_group, y_pred_binary)

    # TPR (True Positive Rate) - for equal opportunity
    positive_mask = y_true_group == 1
    if positive_mask.sum() > 0:
        tpr = y_pred_binary[positive_mask].mean()
    else:
        tpr = np.nan

    return {
        "auc": auc,
        "accuracy": accuracy,
        "tpr": tpr,
        "n_samples": len(y_true_group)
    }


def evaluate_fairness(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    demographics: pd.DataFrame,
    device: torch.device
) -> dict:
    """
    Evaluate fairness metrics across demographic groups.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        demographics: DataFrame with 'sex' and 'race' columns
        device: Device to run on

    Returns:
        Dictionary with fairness metrics
    """
    y_pred = get_predictions(model, X_test, device)

    results = {
        "overall": compute_subgroup_metrics(y_test, y_pred, np.ones(len(y_test), dtype=bool))
    }

    # Evaluate by sex
    sex_groups = demographics["sex"].unique()
    for sex in sex_groups:
        mask = (demographics["sex"] == sex).values
        results[f"sex_{sex}"] = compute_subgroup_metrics(y_test, y_pred, mask)

    # Compute TPR gap by sex
    if len(sex_groups) >= 2:
        tpr_values = [results[f"sex_{sex}"]["tpr"] for sex in sex_groups if not np.isnan(results[f"sex_{sex}"]["tpr"])]
        if len(tpr_values) >= 2:
            results["tpr_gap_sex"] = max(tpr_values) - min(tpr_values)
        else:
            results["tpr_gap_sex"] = np.nan
    else:
        results["tpr_gap_sex"] = np.nan

    # Evaluate by race
    race_groups = demographics["race"].unique()
    for race in race_groups:
        mask = (demographics["race"] == race).values
        results[f"race_{race}"] = compute_subgroup_metrics(y_test, y_pred, mask)

    # Compute TPR gap by race
    tpr_values = [results[f"race_{race}"]["tpr"] for race in race_groups if not np.isnan(results.get(f"race_{race}", {}).get("tpr", np.nan))]
    if len(tpr_values) >= 2:
        results["tpr_gap_race"] = max(tpr_values) - min(tpr_values)
    else:
        results["tpr_gap_race"] = np.nan

    return results


def run_fairness_evaluation(
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "results",
    data_dir: str = "data"
) -> list[dict]:
    """
    Run fairness evaluation on baseline and DP models.

    Args:
        checkpoint_dir: Directory with model checkpoints
        output_dir: Directory to save results
        data_dir: Directory with data

    Returns:
        List of result dictionaries
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Adult dataset with demographics
    print("\n--- Loading Adult Dataset ---")
    X_test, y_test, demographics = load_adult_with_demographics(data_dir)
    print(f"Test samples: {len(X_test)}")
    print(f"Sex groups: {demographics['sex'].unique()}")
    print(f"Race groups: {demographics['race'].unique()}")

    results = []

    # Find all Adult checkpoints
    checkpoint_patterns = [
        ("baseline", "adult_baseline/baseline_final.pt", None),
    ]

    # Add DP checkpoints for different epsilons
    for epsilon in [2, 4, 8, 16]:
        dp_path = f"adult_dp_eps{epsilon}/dp_final.pt"
        checkpoint_patterns.append(("dp", dp_path, epsilon))

    for model_type, checkpoint_path, epsilon in checkpoint_patterns:
        full_path = os.path.join(checkpoint_dir, checkpoint_path)

        if not os.path.exists(full_path):
            print(f"Checkpoint not found: {full_path}")
            continue

        print(f"\n--- Evaluating {model_type} (epsilon={epsilon}) ---")

        model = load_model(full_path, device)
        fairness_results = evaluate_fairness(model, X_test, y_test, demographics, device)

        # Create result record
        record = {
            "model_type": model_type,
            "epsilon": epsilon,
            "overall_auc": fairness_results["overall"]["auc"],
            "overall_accuracy": fairness_results["overall"]["accuracy"],
            "overall_tpr": fairness_results["overall"]["tpr"],
            "tpr_gap_sex": fairness_results.get("tpr_gap_sex", np.nan),
            "tpr_gap_race": fairness_results.get("tpr_gap_race", np.nan),
        }

        # Add sex-specific metrics
        for sex in demographics["sex"].unique():
            key = f"sex_{sex}"
            if key in fairness_results:
                record[f"{key}_auc"] = fairness_results[key]["auc"]
                record[f"{key}_accuracy"] = fairness_results[key]["accuracy"]
                record[f"{key}_tpr"] = fairness_results[key]["tpr"]

        # Add race-specific metrics (top groups)
        for race in ["White", "Black"]:
            key = f"race_{race}"
            if key in fairness_results:
                record[f"{key}_auc"] = fairness_results[key]["auc"]
                record[f"{key}_accuracy"] = fairness_results[key]["accuracy"]
                record[f"{key}_tpr"] = fairness_results[key]["tpr"]

        results.append(record)

        # Print summary
        print(f"  Overall AUC: {record['overall_auc']:.4f}")
        print(f"  TPR Gap (Sex): {record['tpr_gap_sex']:.4f}")
        print(f"  TPR Gap (Race): {record['tpr_gap_race']:.4f}")

    return results


def save_results(results: list[dict], output_path: str) -> None:
    """Save fairness results to CSV."""
    if not results:
        print("No results to save.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get all unique keys
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())

    fieldnames = sorted(all_keys)
    # Ensure important fields come first
    priority_fields = ["model_type", "epsilon", "overall_auc", "overall_accuracy",
                       "tpr_gap_sex", "tpr_gap_race"]
    fieldnames = [f for f in priority_fields if f in fieldnames] + \
                 [f for f in fieldnames if f not in priority_fields]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nFairness results saved to {output_path}")


def print_summary(results: list[dict]) -> None:
    """Print summary table."""
    print("\n" + "="*80)
    print("FAIRNESS EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model':<12} {'Epsilon':<10} {'Overall AUC':<14} {'TPR Gap Sex':<14} {'TPR Gap Race':<14}")
    print("-"*80)

    for r in results:
        eps_str = f"{r['epsilon']:.1f}" if r['epsilon'] else "N/A"
        tpr_sex = f"{r['tpr_gap_sex']:.4f}" if not np.isnan(r.get('tpr_gap_sex', np.nan)) else "N/A"
        tpr_race = f"{r['tpr_gap_race']:.4f}" if not np.isnan(r.get('tpr_gap_race', np.nan)) else "N/A"
        print(f"{r['model_type']:<12} {eps_str:<10} {r['overall_auc']:.4f}        {tpr_sex:<14} {tpr_race:<14}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Fairness Evaluation on Adult Dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory with model checkpoints")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory")

    args = parser.parse_args()

    print("="*60)
    print("FAIRNESS EVALUATION")
    print("="*60)

    results = run_fairness_evaluation(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        data_dir=args.data_dir
    )

    # Save results
    output_path = os.path.join(args.output_dir, "fairness.csv")
    save_results(results, output_path)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
