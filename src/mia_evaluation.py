"""
Membership Inference Attack (MIA) evaluation script.

This script implements membership inference attacks to evaluate the privacy
leakage of trained models. It compares baseline vs DP-trained models.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm

from data_loader import get_data_loaders
from model import create_model


def get_model_confidence(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get model prediction confidence for all samples.

    Args:
        model: Trained model
        data_loader: DataLoader with samples
        device: Device to run inference on

    Returns:
        Tuple of (confidences, true_labels) as numpy arrays
    """
    model.eval()
    all_confidences = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).cpu().numpy()

            # Confidence = predicted probability for true class
            y_np = y_batch.numpy().flatten()
            conf = np.where(y_np == 1, outputs.flatten(), 1 - outputs.flatten())

            all_confidences.extend(conf)
            all_labels.extend(y_np)

    return np.array(all_confidences), np.array(all_labels)


def threshold_attack(
    member_confidences: np.ndarray,
    nonmember_confidences: np.ndarray
) -> dict:
    """
    Simple threshold-based membership inference attack.

    The attack assumes members have higher prediction confidence than non-members.

    Args:
        member_confidences: Confidence scores for training members
        nonmember_confidences: Confidence scores for non-members (test set)

    Returns:
        Dictionary with attack metrics
    """
    # Create labels: 1 for members, 0 for non-members
    y_true = np.concatenate([
        np.ones(len(member_confidences)),
        np.zeros(len(nonmember_confidences))
    ])

    # Use confidence as attack score (higher = more likely member)
    attack_scores = np.concatenate([member_confidences, nonmember_confidences])

    # Calculate AUC-ROC
    auc = roc_auc_score(y_true, attack_scores)

    # Find optimal threshold (Youden's J statistic)
    thresholds = np.linspace(0, 1, 100)
    best_acc = 0
    best_threshold = 0.5

    for thresh in thresholds:
        predictions = (attack_scores >= thresh).astype(int)
        acc = accuracy_score(y_true, predictions)
        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh

    # Calculate metrics at best threshold
    predictions = (attack_scores >= best_threshold).astype(int)

    # True positive rate (members correctly identified)
    tpr = np.mean(predictions[y_true == 1])
    # False positive rate (non-members incorrectly identified as members)
    fpr = np.mean(predictions[y_true == 0])

    return {
        "auc": auc,
        "accuracy": best_acc,
        "threshold": best_threshold,
        "tpr": tpr,
        "fpr": fpr,
        "advantage": tpr - fpr  # Membership advantage
    }


def loss_attack(
    model: nn.Module,
    member_loader: DataLoader,
    nonmember_loader: DataLoader,
    device: torch.device
) -> dict:
    """
    Loss-based membership inference attack.

    Uses the model's loss as the attack signal (lower loss = more likely member).

    Args:
        model: Trained model
        member_loader: DataLoader for training members
        nonmember_loader: DataLoader for non-members
        device: Device to run on

    Returns:
        Dictionary with attack metrics
    """
    model.eval()
    criterion = nn.BCELoss(reduction='none')

    def get_losses(loader):
        losses = []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                batch_losses = criterion(outputs, y_batch).cpu().numpy()
                losses.extend(batch_losses.flatten())
        return np.array(losses)

    member_losses = get_losses(member_loader)
    nonmember_losses = get_losses(nonmember_loader)

    # Create labels: 1 for members, 0 for non-members
    y_true = np.concatenate([
        np.ones(len(member_losses)),
        np.zeros(len(nonmember_losses))
    ])

    # Use negative loss as attack score (lower loss = higher score = more likely member)
    attack_scores = -np.concatenate([member_losses, nonmember_losses])

    # Calculate AUC-ROC
    auc = roc_auc_score(y_true, attack_scores)

    # Find optimal threshold
    thresholds = np.percentile(attack_scores, np.linspace(0, 100, 100))
    best_acc = 0

    for thresh in thresholds:
        predictions = (attack_scores >= thresh).astype(int)
        acc = accuracy_score(y_true, predictions)
        if acc > best_acc:
            best_acc = acc

    return {
        "auc": auc,
        "accuracy": best_acc,
        "member_loss_mean": np.mean(member_losses),
        "nonmember_loss_mean": np.mean(nonmember_losses)
    }


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_mia_evaluation(
    baseline_path: str = None,
    dp_path: str = None,
    batch_size: int = 256,
    seed: int = 42
) -> dict:
    """
    Run membership inference attack evaluation on trained models.

    Args:
        baseline_path: Path to baseline model checkpoint
        dp_path: Path to DP model checkpoint
        batch_size: Batch size for evaluation
        seed: Random seed

    Returns:
        Dictionary with MIA results for each model
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader, input_dim = get_data_loaders(
        batch_size=batch_size, seed=seed
    )

    results = {}

    # Evaluate baseline model
    if baseline_path and os.path.exists(baseline_path):
        print(f"\n--- Evaluating Baseline Model ---")
        print(f"Loading from: {baseline_path}")

        model = load_model(baseline_path, device)

        # Get confidence scores
        member_conf, _ = get_model_confidence(model, train_loader, device)
        nonmember_conf, _ = get_model_confidence(model, test_loader, device)

        # Run attacks
        threshold_results = threshold_attack(member_conf, nonmember_conf)
        loss_results = loss_attack(model, train_loader, test_loader, device)

        results["baseline"] = {
            "threshold_attack": threshold_results,
            "loss_attack": loss_results
        }

        print(f"\nThreshold Attack:")
        print(f"  AUC: {threshold_results['auc']:.4f}")
        print(f"  Accuracy: {threshold_results['accuracy']:.4f}")
        print(f"  Advantage: {threshold_results['advantage']:.4f}")

        print(f"\nLoss Attack:")
        print(f"  AUC: {loss_results['auc']:.4f}")
        print(f"  Accuracy: {loss_results['accuracy']:.4f}")

    # Evaluate DP model
    if dp_path and os.path.exists(dp_path):
        print(f"\n--- Evaluating DP Model ---")
        print(f"Loading from: {dp_path}")

        model = load_model(dp_path, device)

        # Get confidence scores
        member_conf, _ = get_model_confidence(model, train_loader, device)
        nonmember_conf, _ = get_model_confidence(model, test_loader, device)

        # Run attacks
        threshold_results = threshold_attack(member_conf, nonmember_conf)
        loss_results = loss_attack(model, train_loader, test_loader, device)

        results["dp"] = {
            "threshold_attack": threshold_results,
            "loss_attack": loss_results
        }

        print(f"\nThreshold Attack:")
        print(f"  AUC: {threshold_results['auc']:.4f}")
        print(f"  Accuracy: {threshold_results['accuracy']:.4f}")
        print(f"  Advantage: {threshold_results['advantage']:.4f}")

        print(f"\nLoss Attack:")
        print(f"  AUC: {loss_results['auc']:.4f}")
        print(f"  Accuracy: {loss_results['accuracy']:.4f}")

    # Compare results
    if "baseline" in results and "dp" in results:
        print("\n--- Comparison: Baseline vs DP ---")
        print(f"Threshold Attack AUC: {results['baseline']['threshold_attack']['auc']:.4f} -> {results['dp']['threshold_attack']['auc']:.4f}")
        print(f"Loss Attack AUC: {results['baseline']['loss_attack']['auc']:.4f} -> {results['dp']['loss_attack']['auc']:.4f}")

        auc_reduction = results['baseline']['threshold_attack']['auc'] - results['dp']['threshold_attack']['auc']
        print(f"Privacy improvement (AUC reduction): {auc_reduction:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Membership Inference Attack Evaluation")
    parser.add_argument("--baseline", type=str, default="checkpoints/baseline_final.pt",
                        help="Path to baseline model checkpoint")
    parser.add_argument("--dp", type=str, default="checkpoints/dp_final.pt",
                        help="Path to DP model checkpoint")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    results = run_mia_evaluation(
        baseline_path=args.baseline,
        dp_path=args.dp,
        batch_size=args.batch_size,
        seed=args.seed
    )

    print("\n--- MIA Evaluation Complete ---")


if __name__ == "__main__":
    main()
