"""
Membership Inference Attack (MIA) evaluation.
Tests how well an attacker can determine if a sample was used for training.

Attack strategy (threshold-based):
- Train a target model on member data
- Compute prediction confidence for members and non-members
- Members typically have higher confidence (overfitting)
- Use confidence threshold to classify member/non-member
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from tqdm import tqdm

import config
from data import get_member_nonmember_split, AdultDataset
from model import create_model, count_parameters


def get_prediction_confidence(model, data_loader, device):
    """
    Get prediction confidence scores for all samples.

    For binary classification, confidence is defined as:
    - P(correct class) = sigmoid(logit) if label=1
    - P(correct class) = 1 - sigmoid(logit) if label=0

    Higher confidence suggests the model has "memorized" the sample.

    Args:
        model: Trained classifier
        data_loader: DataLoader with samples to evaluate
        device: Device to run on

    Returns:
        confidences: Array of confidence scores
        labels: Array of true labels
    """
    model.eval()
    all_confidences = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)

            # Get model predictions (logits)
            logits = model(features)
            probs = torch.sigmoid(logits)

            # Confidence = P(correct class)
            # For label=1: confidence = prob
            # For label=0: confidence = 1 - prob
            confidence = probs * labels + (1 - probs) * (1 - labels)

            all_confidences.extend(confidence.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_confidences), np.array(all_labels)


def get_loss_scores(model, data_loader, device):
    """
    Get per-sample loss values.

    Lower loss typically indicates membership (model fits these better).

    Args:
        model: Trained classifier
        data_loader: DataLoader with samples to evaluate
        device: Device to run on

    Returns:
        losses: Array of per-sample losses
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    all_losses = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)

            logits = model(features)
            losses = criterion(logits, labels)

            all_losses.extend(losses.cpu().numpy())

    return np.array(all_losses)


def train_target_model(member_dataset, input_dim, epochs=None, device=None):
    """
    Train a target model on member data only.

    Args:
        member_dataset: Dataset of members (training data for target)
        input_dim: Input feature dimension
        epochs: Training epochs
        device: Device to train on

    Returns:
        model: Trained target model
    """
    if epochs is None:
        epochs = config.EPOCHS
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and training setup
    model = create_model(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Create DataLoader for members
    train_loader = DataLoader(
        member_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )

    # Training loop
    model.train()
    for epoch in range(epochs):
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


def evaluate_mia(model, member_dataset, nonmember_dataset, device, verbose=True):
    """
    Evaluate membership inference attack on a trained model.

    Metrics:
    - Attack accuracy: How well can attacker distinguish members
    - Attack AUC: Area under ROC curve
    - Advantage: Attack accuracy - 0.5 (improvement over random guessing)

    Args:
        model: Trained target model to attack
        member_dataset: Dataset of training members
        nonmember_dataset: Dataset of non-members
        device: Device to run on
        verbose: Whether to print results

    Returns:
        results: Dictionary with attack metrics
    """
    # Create DataLoaders
    member_loader = DataLoader(
        member_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    nonmember_loader = DataLoader(
        nonmember_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    # Get confidence scores for both sets
    member_conf, _ = get_prediction_confidence(model, member_loader, device)
    nonmember_conf, _ = get_prediction_confidence(model, nonmember_loader, device)

    # Also get loss-based scores (lower loss = more likely member)
    member_loss = get_loss_scores(model, member_loader, device)
    nonmember_loss = get_loss_scores(model, nonmember_loader, device)

    # Create labels: 1 = member, 0 = non-member
    n_members = len(member_conf)
    n_nonmembers = len(nonmember_conf)

    # Combine scores and labels
    all_conf = np.concatenate([member_conf, nonmember_conf])
    all_loss = np.concatenate([member_loss, nonmember_loss])
    membership_labels = np.concatenate([
        np.ones(n_members),
        np.zeros(n_nonmembers)
    ])

    # Confidence-based attack: higher confidence -> predict member
    conf_auc = roc_auc_score(membership_labels, all_conf)

    # Loss-based attack: lower loss -> predict member (negate for AUC)
    loss_auc = roc_auc_score(membership_labels, -all_loss)

    # Find optimal threshold for confidence-based attack
    fpr, tpr, thresholds = roc_curve(membership_labels, all_conf)
    # Optimal threshold maximizes TPR - FPR (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Compute accuracy at optimal threshold
    conf_predictions = (all_conf >= optimal_threshold).astype(int)
    attack_accuracy = accuracy_score(membership_labels, conf_predictions)

    # Advantage over random guessing
    advantage = attack_accuracy - 0.5

    results = {
        "attack_accuracy": attack_accuracy,
        "attack_advantage": advantage,
        "confidence_auc": conf_auc,
        "loss_auc": loss_auc,
        "optimal_threshold": optimal_threshold,
        "n_members": n_members,
        "n_nonmembers": n_nonmembers,
        "member_conf_mean": member_conf.mean(),
        "nonmember_conf_mean": nonmember_conf.mean(),
        "member_loss_mean": member_loss.mean(),
        "nonmember_loss_mean": nonmember_loss.mean(),
    }

    if verbose:
        print("\n" + "="*50)
        print("Membership Inference Attack Results")
        print("="*50)
        print(f"Members: {n_members}, Non-members: {n_nonmembers}")
        print(f"\nConfidence-based attack:")
        print(f"  Member mean confidence:     {member_conf.mean():.4f}")
        print(f"  Non-member mean confidence: {nonmember_conf.mean():.4f}")
        print(f"  AUC: {conf_auc:.4f}")
        print(f"\nLoss-based attack:")
        print(f"  Member mean loss:     {member_loss.mean():.4f}")
        print(f"  Non-member mean loss: {nonmember_loss.mean():.4f}")
        print(f"  AUC: {loss_auc:.4f}")
        print(f"\nOverall attack performance:")
        print(f"  Accuracy: {attack_accuracy:.4f}")
        print(f"  Advantage over random: {advantage:.4f}")
        print("="*50)

    return results


def run_mia_evaluation(model_path=None, train_new=True, epochs=None,
                       use_dp=False, epsilon=None, verbose=True):
    """
    Run complete MIA evaluation pipeline.

    Either loads a pre-trained model or trains a new one,
    then evaluates membership inference attack.

    Args:
        model_path: Path to pre-trained model (optional)
        train_new: Whether to train a new model
        epochs: Training epochs for new model
        use_dp: Whether to use DP training for new model
        epsilon: Privacy budget if using DP
        verbose: Whether to print progress

    Returns:
        results: Dictionary with attack metrics
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.RANDOM_SEED)

    if verbose:
        print(f"Using device: {device}")
        print("\nPreparing member/non-member split...")

    # Get member/non-member split
    member_dataset, nonmember_dataset, test_dataset, input_dim = \
        get_member_nonmember_split(member_ratio=0.5)

    if verbose:
        print(f"Members: {len(member_dataset)}")
        print(f"Non-members: {len(nonmember_dataset)}")

    if model_path and os.path.exists(model_path):
        # Load pre-trained model
        if verbose:
            print(f"\nLoading model from {model_path}...")

        checkpoint = torch.load(model_path, map_location=device)
        model = create_model(checkpoint["input_dim"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Train new model
        if verbose:
            print(f"\nTraining target model on member data...")

        if use_dp:
            # Import DP training
            from train_dp import train_dp
            from torch.utils.data import DataLoader

            # Create DataLoader from member dataset
            member_loader = DataLoader(
                member_dataset, batch_size=config.BATCH_SIZE, shuffle=True
            )

            # For DP training, we need to modify the training function
            # This is a simplified version - train on members only
            if verbose:
                print(f"  Using DP-SGD with epsilon={epsilon or config.EPSILON}")

            # Train with DP on member data
            model = train_target_model_dp(
                member_dataset, input_dim, epochs=epochs,
                epsilon=epsilon, device=device
            )
        else:
            if verbose:
                print("  Using standard (non-private) training")

            model = train_target_model(
                member_dataset, input_dim, epochs=epochs, device=device
            )

    # Run attack evaluation
    if verbose:
        print("\nRunning membership inference attack...")

    results = evaluate_mia(
        model, member_dataset, nonmember_dataset, device, verbose=verbose
    )

    return results


def train_target_model_dp(member_dataset, input_dim, epochs=None,
                          epsilon=None, device=None):
    """
    Train a target model with differential privacy on member data.

    Args:
        member_dataset: Dataset of members
        input_dim: Input feature dimension
        epochs: Training epochs
        epsilon: Privacy budget
        device: Device to train on

    Returns:
        model: Trained DP model
    """
    from opacus import PrivacyEngine

    if epochs is None:
        epochs = config.EPOCHS
    if epsilon is None:
        epsilon = config.EPSILON
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and training setup
    model = create_model(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Create DataLoader for members
    train_loader = DataLoader(
        member_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )

    # Wrap with PrivacyEngine
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
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Return the underlying model
    return model._module


def compare_baseline_vs_dp(epochs=None, epsilon=None, verbose=True):
    """
    Compare MIA vulnerability between baseline and DP models.

    Args:
        epochs: Training epochs
        epsilon: Privacy budget for DP model
        verbose: Whether to print results

    Returns:
        comparison: Dictionary with results for both models
    """
    if verbose:
        print("="*60)
        print("Comparing Membership Inference Attack Vulnerability")
        print("="*60)

    # Evaluate baseline model
    if verbose:
        print("\n[1/2] Evaluating BASELINE (non-private) model...")

    baseline_results = run_mia_evaluation(
        train_new=True, epochs=epochs, use_dp=False, verbose=verbose
    )

    # Evaluate DP model
    if verbose:
        print("\n[2/2] Evaluating DP-SGD (private) model...")

    dp_results = run_mia_evaluation(
        train_new=True, epochs=epochs, use_dp=True,
        epsilon=epsilon, verbose=verbose
    )

    # Comparison summary
    if verbose:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Metric':<30} {'Baseline':>12} {'DP-SGD':>12}")
        print("-"*60)
        print(f"{'Attack Accuracy':<30} {baseline_results['attack_accuracy']:>12.4f} {dp_results['attack_accuracy']:>12.4f}")
        print(f"{'Attack Advantage':<30} {baseline_results['attack_advantage']:>12.4f} {dp_results['attack_advantage']:>12.4f}")
        print(f"{'Confidence AUC':<30} {baseline_results['confidence_auc']:>12.4f} {dp_results['confidence_auc']:>12.4f}")
        print(f"{'Loss AUC':<30} {baseline_results['loss_auc']:>12.4f} {dp_results['loss_auc']:>12.4f}")
        print("="*60)

        if dp_results['attack_advantage'] < baseline_results['attack_advantage']:
            reduction = baseline_results['attack_advantage'] - dp_results['attack_advantage']
            print(f"\nDP training reduced attack advantage by {reduction:.4f}")
        else:
            print("\nNote: DP model shows similar or higher vulnerability")

    return {
        "baseline": baseline_results,
        "dp": dp_results,
    }


def main():
    """Command-line interface for MIA evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate membership inference attack vulnerability"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to pre-trained model (optional)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Training epochs for target model")
    parser.add_argument("--compare", action="store_true",
                        help="Compare baseline vs DP model")
    parser.add_argument("--epsilon", type=float, default=config.EPSILON,
                        help="Privacy budget for DP model")
    parser.add_argument("--use-dp", action="store_true",
                        help="Use DP training for target model")
    args = parser.parse_args()

    if args.compare:
        compare_baseline_vs_dp(epochs=args.epochs, epsilon=args.epsilon)
    else:
        run_mia_evaluation(
            model_path=args.model,
            train_new=args.model is None,
            epochs=args.epochs,
            use_dp=args.use_dp,
            epsilon=args.epsilon,
            verbose=True
        )


if __name__ == "__main__":
    main()
