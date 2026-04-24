"""
Membership Inference Attack (MIA) evaluation script.

This script implements membership inference attacks to evaluate the privacy
leakage of trained models. It compares baseline vs DP-trained models.

Attack methods:
- Threshold attack (confidence-based)
- Loss attack
- Shadow model attack (trains k shadow models + logistic regression classifier)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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


def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Precision-Recall AUC."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def train_shadow_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device
) -> nn.Module:
    """Train a single shadow model."""
    model = create_model(input_dim, hidden_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model


def get_attack_features(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device
) -> np.ndarray:
    """
    Extract attack features from model predictions.

    Returns feature vector: [confidence, loss, prediction_correct]
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        criterion = nn.BCELoss(reduction='none')
        losses = criterion(outputs, y_tensor).cpu().numpy().flatten()

    outputs_np = outputs.cpu().numpy().flatten()

    # Confidence for true class
    confidence = np.where(y == 1, outputs_np, 1 - outputs_np)

    # Prediction correctness
    predictions = (outputs_np >= 0.5).astype(float)
    correct = (predictions == y).astype(float)

    # Feature vector
    features = np.column_stack([confidence, losses, correct])

    return features


def shadow_model_attack(
    target_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    input_dim: int,
    hidden_dim: int,
    device: torch.device,
    n_shadow: int = 5,
    shadow_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    seed: int = 42
) -> dict:
    """
    Shadow model-based membership inference attack.

    Trains k shadow models, extracts features from their predictions,
    and trains a logistic regression attack classifier.

    Args:
        target_model: The model to attack
        train_loader: Training data loader (members)
        test_loader: Test data loader (non-members)
        input_dim: Input dimension for shadow models
        hidden_dim: Hidden dimension for shadow models
        device: Device to run on
        n_shadow: Number of shadow models to train
        shadow_epochs: Training epochs per shadow model
        batch_size: Batch size for training
        learning_rate: Learning rate for shadow models
        seed: Random seed

    Returns:
        Dictionary with attack metrics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Extract all data from loaders
    X_members, y_members = [], []
    for X_batch, y_batch in train_loader:
        X_members.append(X_batch.numpy())
        y_members.append(y_batch.numpy().flatten())
    X_members = np.concatenate(X_members)
    y_members = np.concatenate(y_members)

    X_nonmembers, y_nonmembers = [], []
    for X_batch, y_batch in test_loader:
        X_nonmembers.append(X_batch.numpy())
        y_nonmembers.append(y_batch.numpy().flatten())
    X_nonmembers = np.concatenate(X_nonmembers)
    y_nonmembers = np.concatenate(y_nonmembers)

    # Combine all data for shadow model training
    X_all = np.concatenate([X_members, X_nonmembers])
    y_all = np.concatenate([y_members, y_nonmembers])

    # Train shadow models and collect attack training data
    attack_features = []
    attack_labels = []

    print(f"  Training {n_shadow} shadow models...")

    for i in range(n_shadow):
        # Random split for this shadow model
        n_samples = len(X_all)
        indices = np.random.permutation(n_samples)
        split = n_samples // 2

        shadow_train_idx = indices[:split]
        shadow_test_idx = indices[split:]

        X_shadow_train = X_all[shadow_train_idx]
        y_shadow_train = y_all[shadow_train_idx]
        X_shadow_test = X_all[shadow_test_idx]
        y_shadow_test = y_all[shadow_test_idx]

        # Train shadow model
        shadow_model = train_shadow_model(
            X_shadow_train, y_shadow_train,
            input_dim, hidden_dim, shadow_epochs,
            batch_size, learning_rate, device
        )

        # Get features for members (shadow train set)
        member_features = get_attack_features(
            shadow_model, X_shadow_train, y_shadow_train, device
        )
        attack_features.append(member_features)
        attack_labels.append(np.ones(len(member_features)))

        # Get features for non-members (shadow test set)
        nonmember_features = get_attack_features(
            shadow_model, X_shadow_test, y_shadow_test, device
        )
        attack_features.append(nonmember_features)
        attack_labels.append(np.zeros(len(nonmember_features)))

    # Combine attack training data
    X_attack = np.concatenate(attack_features)
    y_attack = np.concatenate(attack_labels)

    # Train attack classifier
    attack_classifier = LogisticRegression(max_iter=1000, random_state=seed)
    attack_classifier.fit(X_attack, y_attack)

    # Get features from target model for actual members and non-members
    target_member_features = get_attack_features(
        target_model, X_members, y_members, device
    )
    target_nonmember_features = get_attack_features(
        target_model, X_nonmembers, y_nonmembers, device
    )

    # Predict membership
    member_probs = attack_classifier.predict_proba(target_member_features)[:, 1]
    nonmember_probs = attack_classifier.predict_proba(target_nonmember_features)[:, 1]

    # Evaluate attack
    y_true = np.concatenate([
        np.ones(len(member_probs)),
        np.zeros(len(nonmember_probs))
    ])
    attack_scores = np.concatenate([member_probs, nonmember_probs])

    auc_roc = roc_auc_score(y_true, attack_scores)
    pr_auc = compute_pr_auc(y_true, attack_scores)

    # Find best threshold
    best_acc = 0
    for thresh in np.linspace(0, 1, 100):
        predictions = (attack_scores >= thresh).astype(int)
        acc = accuracy_score(y_true, predictions)
        if acc > best_acc:
            best_acc = acc

    return {
        "auc": auc_roc,
        "pr_auc": pr_auc,
        "accuracy": best_acc,
        "n_shadow_models": n_shadow
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = create_model(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_mia_evaluation(
    dataset: str = "adult",
    baseline_path: str = None,
    dp_path: str = None,
    batch_size: int = 256,
    seed: int = 42,
    n_shadow: int = 5,
    run_shadow_attack: bool = False
) -> dict:
    """
    Run membership inference attack evaluation on trained models.

    Args:
        dataset: Dataset name ('adult', 'bank', or 'credit_default')
        baseline_path: Path to baseline model checkpoint
        dp_path: Path to DP model checkpoint
        batch_size: Batch size for evaluation
        seed: Random seed
        n_shadow: Number of shadow models for shadow attack
        run_shadow_attack: Whether to run shadow model attack

    Returns:
        Dictionary with MIA results for each model
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"\n--- Loading Data ({dataset}) ---")
    train_loader, val_loader, test_loader, input_dim = get_data_loaders(
        dataset=dataset, batch_size=batch_size, seed=seed
    )

    results = {}

    # Evaluate baseline model
    if baseline_path and os.path.exists(baseline_path):
        print(f"\n--- Evaluating Baseline Model ---")
        print(f"Loading from: {baseline_path}")

        model = load_model(baseline_path, device)
        checkpoint = torch.load(baseline_path, map_location=device, weights_only=False)
        hidden_dim = checkpoint.get("hidden_dim", 128)

        # Get confidence scores
        member_conf, _ = get_model_confidence(model, train_loader, device)
        nonmember_conf, _ = get_model_confidence(model, test_loader, device)

        # Run attacks
        threshold_results = threshold_attack(member_conf, nonmember_conf)
        loss_results = loss_attack(model, train_loader, test_loader, device)

        # Add PR-AUC to threshold results
        y_true = np.concatenate([np.ones(len(member_conf)), np.zeros(len(nonmember_conf))])
        scores = np.concatenate([member_conf, nonmember_conf])
        threshold_results["pr_auc"] = compute_pr_auc(y_true, scores)

        results["baseline"] = {
            "threshold_attack": threshold_results,
            "loss_attack": loss_results
        }

        print(f"\nThreshold Attack:")
        print(f"  AUC-ROC: {threshold_results['auc']:.4f}")
        print(f"  PR-AUC: {threshold_results['pr_auc']:.4f}")
        print(f"  Accuracy: {threshold_results['accuracy']:.4f}")
        print(f"  Advantage: {threshold_results['advantage']:.4f}")

        print(f"\nLoss Attack:")
        print(f"  AUC: {loss_results['auc']:.4f}")
        print(f"  Accuracy: {loss_results['accuracy']:.4f}")

        # Shadow model attack
        if run_shadow_attack:
            print(f"\nShadow Model Attack (k={n_shadow}):")
            shadow_results = shadow_model_attack(
                model, train_loader, test_loader,
                input_dim, hidden_dim, device,
                n_shadow=n_shadow, seed=seed
            )
            results["baseline"]["shadow_attack"] = shadow_results
            print(f"  AUC-ROC: {shadow_results['auc']:.4f}")
            print(f"  PR-AUC: {shadow_results['pr_auc']:.4f}")
            print(f"  Accuracy: {shadow_results['accuracy']:.4f}")

    # Evaluate DP model
    if dp_path and os.path.exists(dp_path):
        print(f"\n--- Evaluating DP Model ---")
        print(f"Loading from: {dp_path}")

        model = load_model(dp_path, device)
        checkpoint = torch.load(dp_path, map_location=device, weights_only=False)
        hidden_dim = checkpoint.get("hidden_dim", 128)

        # Get confidence scores
        member_conf, _ = get_model_confidence(model, train_loader, device)
        nonmember_conf, _ = get_model_confidence(model, test_loader, device)

        # Run attacks
        threshold_results = threshold_attack(member_conf, nonmember_conf)
        loss_results = loss_attack(model, train_loader, test_loader, device)

        # Add PR-AUC to threshold results
        y_true = np.concatenate([np.ones(len(member_conf)), np.zeros(len(nonmember_conf))])
        scores = np.concatenate([member_conf, nonmember_conf])
        threshold_results["pr_auc"] = compute_pr_auc(y_true, scores)

        results["dp"] = {
            "threshold_attack": threshold_results,
            "loss_attack": loss_results
        }

        print(f"\nThreshold Attack:")
        print(f"  AUC-ROC: {threshold_results['auc']:.4f}")
        print(f"  PR-AUC: {threshold_results['pr_auc']:.4f}")
        print(f"  Accuracy: {threshold_results['accuracy']:.4f}")
        print(f"  Advantage: {threshold_results['advantage']:.4f}")

        print(f"\nLoss Attack:")
        print(f"  AUC: {loss_results['auc']:.4f}")
        print(f"  Accuracy: {loss_results['accuracy']:.4f}")

        # Shadow model attack
        if run_shadow_attack:
            print(f"\nShadow Model Attack (k={n_shadow}):")
            shadow_results = shadow_model_attack(
                model, train_loader, test_loader,
                input_dim, hidden_dim, device,
                n_shadow=n_shadow, seed=seed
            )
            results["dp"]["shadow_attack"] = shadow_results
            print(f"  AUC-ROC: {shadow_results['auc']:.4f}")
            print(f"  PR-AUC: {shadow_results['pr_auc']:.4f}")
            print(f"  Accuracy: {shadow_results['accuracy']:.4f}")

    # Compare results
    if "baseline" in results and "dp" in results:
        print("\n--- Comparison: Baseline vs DP ---")
        print(f"Threshold Attack AUC: {results['baseline']['threshold_attack']['auc']:.4f} -> {results['dp']['threshold_attack']['auc']:.4f}")
        print(f"Loss Attack AUC: {results['baseline']['loss_attack']['auc']:.4f} -> {results['dp']['loss_attack']['auc']:.4f}")

        if run_shadow_attack:
            print(f"Shadow Attack AUC: {results['baseline']['shadow_attack']['auc']:.4f} -> {results['dp']['shadow_attack']['auc']:.4f}")

        auc_reduction = results['baseline']['threshold_attack']['auc'] - results['dp']['threshold_attack']['auc']
        print(f"Privacy improvement (AUC reduction): {auc_reduction:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Membership Inference Attack Evaluation")
    parser.add_argument("--dataset", type=str, default="adult",
                        choices=["adult", "bank", "credit_default"],
                        help="Dataset to use (default: adult)")
    parser.add_argument("--baseline", type=str, default="checkpoints/baseline_final.pt",
                        help="Path to baseline model checkpoint")
    parser.add_argument("--dp", type=str, default="checkpoints/dp_final.pt",
                        help="Path to DP model checkpoint")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shadow-models", type=int, default=5,
                        help="Number of shadow models for shadow attack (default: 5)")
    parser.add_argument("--run-shadow", action="store_true",
                        help="Run shadow model attack (slower but more powerful)")

    args = parser.parse_args()

    results = run_mia_evaluation(
        dataset=args.dataset,
        baseline_path=args.baseline,
        dp_path=args.dp,
        batch_size=args.batch_size,
        seed=args.seed,
        n_shadow=args.shadow_models,
        run_shadow_attack=args.run_shadow
    )

    print("\n--- MIA Evaluation Complete ---")


if __name__ == "__main__":
    main()
