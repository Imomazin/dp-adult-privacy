"""
Membership Inference Attack (MIA) evaluation.
Tests how well an attacker can determine if a sample was used for training.

Implements two attack strategies:
1. Threshold-based attack: Uses prediction confidence or loss as signal
2. Shadow-model attack: Trains shadow models to learn attack classifier

Supports multiple datasets: adult, credit_default.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.linear_model import LogisticRegression

import config
from data import get_member_nonmember_split, TabularDataset, SUPPORTED_DATASETS
from model import create_model, count_parameters


# =============================================================================
# Feature Extraction for MIA
# =============================================================================
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
        probs: Array of raw prediction probabilities
    """
    model.eval()
    all_confidences = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)

            logits = model(features)
            probs = torch.sigmoid(logits)

            # Confidence = P(correct class)
            confidence = probs * labels + (1 - probs) * (1 - labels)

            all_confidences.extend(confidence.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_confidences), np.array(all_labels), np.array(all_probs)


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


def get_attack_features(model, data_loader, device):
    """
    Extract features for the attack model.

    Features include:
    - Prediction probability
    - Prediction confidence (P(correct class))
    - Loss value
    - Prediction correctness

    Args:
        model: Target model
        data_loader: DataLoader
        device: Device

    Returns:
        features: (N, 4) array of attack features
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    all_features = []

    with torch.no_grad():
        for features, labels in data_loader:
            features_t, labels_t = features.to(device), labels.to(device)

            logits = model(features_t)
            probs = torch.sigmoid(logits)
            losses = criterion(logits, labels_t)

            # Confidence = P(correct class)
            confidence = probs * labels_t + (1 - probs) * (1 - labels_t)

            # Prediction correctness
            predictions = (probs > 0.5).float()
            correct = (predictions == labels_t).float()

            # Stack features
            batch_features = torch.stack([
                probs, confidence, losses, correct
            ], dim=1)

            all_features.append(batch_features.cpu().numpy())

    return np.vstack(all_features)


# =============================================================================
# Target Model Training
# =============================================================================
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

    model = create_model(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_loader = DataLoader(
        member_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )

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

    model = create_model(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_loader = DataLoader(
        member_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )

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

    model.train()
    for epoch in range(epochs):
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model._module


# =============================================================================
# Threshold-based MIA
# =============================================================================
def evaluate_threshold_mia(model, member_dataset, nonmember_dataset, device, verbose=True):
    """
    Evaluate threshold-based membership inference attack.

    Uses prediction confidence and loss as attack signals.

    Args:
        model: Trained target model to attack
        member_dataset: Dataset of training members
        nonmember_dataset: Dataset of non-members
        device: Device to run on
        verbose: Whether to print results

    Returns:
        results: Dictionary with attack metrics
    """
    member_loader = DataLoader(
        member_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    nonmember_loader = DataLoader(
        nonmember_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    # Get scores for both sets
    member_conf, _, _ = get_prediction_confidence(model, member_loader, device)
    nonmember_conf, _, _ = get_prediction_confidence(model, nonmember_loader, device)

    member_loss = get_loss_scores(model, member_loader, device)
    nonmember_loss = get_loss_scores(model, nonmember_loader, device)

    n_members = len(member_conf)
    n_nonmembers = len(nonmember_conf)

    # Combine scores and labels
    all_conf = np.concatenate([member_conf, nonmember_conf])
    all_loss = np.concatenate([member_loss, nonmember_loss])
    membership_labels = np.concatenate([
        np.ones(n_members),
        np.zeros(n_nonmembers)
    ])

    # Confidence-based attack
    conf_auc = roc_auc_score(membership_labels, all_conf)

    # Loss-based attack (negate for AUC)
    loss_auc = roc_auc_score(membership_labels, -all_loss)

    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(membership_labels, all_conf)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    conf_predictions = (all_conf >= optimal_threshold).astype(int)
    attack_accuracy = accuracy_score(membership_labels, conf_predictions)
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
        print("Threshold-based MIA Results")
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


# =============================================================================
# Shadow Model MIA
# =============================================================================
def train_shadow_models(dataset, input_dim, n_shadows=5, epochs=None, device=None):
    """
    Train shadow models for the shadow-model attack.

    Each shadow model is trained on a random subset of the data,
    similar to how the target model was trained.

    Args:
        dataset: Full dataset to sample from
        input_dim: Input feature dimension
        n_shadows: Number of shadow models to train
        epochs: Training epochs per model
        device: Device to train on

    Returns:
        shadow_models: List of trained shadow models
        shadow_member_indices: List of member indices for each shadow
    """
    if epochs is None:
        epochs = config.EPOCHS
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_samples = len(dataset)
    shadow_size = n_samples // 2  # Each shadow trained on half the data

    shadow_models = []
    shadow_member_indices = []

    for i in range(n_shadows):
        # Random split for this shadow model
        np.random.seed(config.RANDOM_SEED + i + 100)
        indices = np.random.permutation(n_samples)
        member_idx = indices[:shadow_size]
        shadow_member_indices.append(set(member_idx))

        # Create subset dataset
        member_subset = Subset(dataset, member_idx)

        # Train shadow model
        model = train_target_model(member_subset, input_dim, epochs=epochs, device=device)
        shadow_models.append(model)

    return shadow_models, shadow_member_indices


def shadow_model_attack(target_model, shadow_models, shadow_member_indices,
                        dataset, member_indices, nonmember_indices,
                        device, verbose=True):
    """
    Perform shadow-model based membership inference attack.

    1. For each shadow model, extract attack features for its members/non-members
    2. Train a logistic regression attack model on shadow features
    3. Apply attack model to target model's predictions

    Args:
        target_model: The target model to attack
        shadow_models: List of trained shadow models
        shadow_member_indices: Member indices for each shadow model
        dataset: Full dataset
        member_indices: Indices of actual members (for target model)
        nonmember_indices: Indices of actual non-members
        device: Device to run on
        verbose: Whether to print results

    Returns:
        results: Dictionary with attack metrics
    """
    # Collect attack training data from shadow models
    attack_X = []
    attack_y = []

    for shadow_model, shadow_members in zip(shadow_models, shadow_member_indices):
        # All indices
        all_indices = set(range(len(dataset)))

        # For each sample, determine if it's a member of this shadow
        for idx in all_indices:
            sample_subset = Subset(dataset, [idx])
            sample_loader = DataLoader(sample_subset, batch_size=1, shuffle=False)

            # Get attack features for this sample
            features = get_attack_features(shadow_model, sample_loader, device)
            attack_X.append(features[0])
            attack_y.append(1 if idx in shadow_members else 0)

    attack_X = np.array(attack_X)
    attack_y = np.array(attack_y)

    # Train attack model (logistic regression)
    attack_model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)
    attack_model.fit(attack_X, attack_y)

    # Evaluate attack on target model
    # Get features for members
    member_subset = Subset(dataset, list(member_indices))
    member_loader = DataLoader(member_subset, batch_size=config.BATCH_SIZE, shuffle=False)
    member_features = get_attack_features(target_model, member_loader, device)

    # Get features for non-members
    nonmember_subset = Subset(dataset, list(nonmember_indices))
    nonmember_loader = DataLoader(nonmember_subset, batch_size=config.BATCH_SIZE, shuffle=False)
    nonmember_features = get_attack_features(target_model, nonmember_loader, device)

    # Combine and predict
    test_X = np.vstack([member_features, nonmember_features])
    test_y = np.concatenate([
        np.ones(len(member_features)),
        np.zeros(len(nonmember_features))
    ])

    # Get attack predictions
    attack_probs = attack_model.predict_proba(test_X)[:, 1]
    attack_preds = attack_model.predict(test_X)

    # Compute metrics
    attack_auc = roc_auc_score(test_y, attack_probs)
    attack_accuracy = accuracy_score(test_y, attack_preds)
    attack_advantage = attack_accuracy - 0.5

    results = {
        "shadow_attack_auc": attack_auc,
        "shadow_attack_accuracy": attack_accuracy,
        "shadow_attack_advantage": attack_advantage,
        "n_shadow_models": len(shadow_models),
    }

    if verbose:
        print("\n" + "="*50)
        print("Shadow-Model MIA Results")
        print("="*50)
        print(f"Number of shadow models: {len(shadow_models)}")
        print(f"Attack AUC: {attack_auc:.4f}")
        print(f"Attack Accuracy: {attack_accuracy:.4f}")
        print(f"Attack Advantage: {attack_advantage:.4f}")
        print("="*50)

    return results


# =============================================================================
# Main Evaluation Functions
# =============================================================================
def run_mia_evaluation(dataset="adult", model_path=None, train_new=True,
                       epochs=None, use_dp=False, epsilon=None,
                       use_shadow=False, n_shadows=5, verbose=True):
    """
    Run complete MIA evaluation pipeline.

    Args:
        dataset: Dataset name
        model_path: Path to pre-trained model (optional)
        train_new: Whether to train a new model
        epochs: Training epochs for new model
        use_dp: Whether to use DP training for new model
        epsilon: Privacy budget if using DP
        use_shadow: Whether to run shadow-model attack
        n_shadows: Number of shadow models for shadow attack
        verbose: Whether to print progress

    Returns:
        results: Dictionary with attack metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    if verbose:
        print(f"Using device: {device}")
        print(f"\nPreparing member/non-member split for {dataset}...")

    # Get member/non-member split
    member_dataset, nonmember_dataset, test_dataset, input_dim = \
        get_member_nonmember_split(dataset=dataset, member_ratio=0.5)

    if verbose:
        print(f"Members: {len(member_dataset)}")
        print(f"Non-members: {len(nonmember_dataset)}")

    if model_path and os.path.exists(model_path):
        if verbose:
            print(f"\nLoading model from {model_path}...")

        checkpoint = torch.load(model_path, map_location=device)
        model = create_model(checkpoint["input_dim"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if verbose:
            print(f"\nTraining target model on member data...")

        if use_dp:
            if verbose:
                print(f"  Using DP-SGD with epsilon={epsilon or config.EPSILON}")
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

    # Run threshold-based attack
    if verbose:
        print("\nRunning threshold-based MIA...")

    results = evaluate_threshold_mia(
        model, member_dataset, nonmember_dataset, device, verbose=verbose
    )

    # Run shadow-model attack if requested
    if use_shadow:
        if verbose:
            print(f"\nRunning shadow-model MIA with {n_shadows} shadow models...")

        # Get full dataset for shadow training
        full_data = member_dataset.dataset if hasattr(member_dataset, 'dataset') else member_dataset

        # Get member/nonmember indices
        if hasattr(member_dataset, 'indices'):
            member_idx = set(member_dataset.indices)
            nonmember_idx = set(nonmember_dataset.indices)
        else:
            # Fallback for non-Subset datasets
            member_idx = set(range(len(member_dataset)))
            nonmember_idx = set(range(len(member_dataset), len(member_dataset) + len(nonmember_dataset)))

        # Train shadow models
        shadow_models, shadow_member_indices = train_shadow_models(
            full_data, input_dim, n_shadows=n_shadows, epochs=epochs, device=device
        )

        # Run shadow attack
        shadow_results = shadow_model_attack(
            model, shadow_models, shadow_member_indices,
            full_data, member_idx, nonmember_idx,
            device, verbose=verbose
        )

        results.update(shadow_results)

    return results


def compare_baseline_vs_dp(dataset="adult", epochs=None, epsilon=None,
                           use_shadow=False, n_shadows=5, verbose=True):
    """
    Compare MIA vulnerability between baseline and DP models.

    Args:
        dataset: Dataset name
        epochs: Training epochs
        epsilon: Privacy budget for DP model
        use_shadow: Whether to include shadow-model attack
        n_shadows: Number of shadow models
        verbose: Whether to print results

    Returns:
        comparison: Dictionary with results for both models
    """
    if verbose:
        print("="*60)
        print(f"Comparing MIA Vulnerability on {dataset}")
        print("="*60)

    # Evaluate baseline model
    if verbose:
        print("\n[1/2] Evaluating BASELINE (non-private) model...")

    baseline_results = run_mia_evaluation(
        dataset=dataset, train_new=True, epochs=epochs, use_dp=False,
        use_shadow=use_shadow, n_shadows=n_shadows, verbose=verbose
    )

    # Evaluate DP model
    if verbose:
        print("\n[2/2] Evaluating DP-SGD (private) model...")

    dp_results = run_mia_evaluation(
        dataset=dataset, train_new=True, epochs=epochs, use_dp=True,
        epsilon=epsilon, use_shadow=use_shadow, n_shadows=n_shadows,
        verbose=verbose
    )

    # Comparison summary
    if verbose:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Metric':<30} {'Baseline':>12} {'DP-SGD':>12}")
        print("-"*60)
        print(f"{'Threshold Attack Accuracy':<30} {baseline_results['attack_accuracy']:>12.4f} {dp_results['attack_accuracy']:>12.4f}")
        print(f"{'Threshold Attack Advantage':<30} {baseline_results['attack_advantage']:>12.4f} {dp_results['attack_advantage']:>12.4f}")
        print(f"{'Confidence AUC':<30} {baseline_results['confidence_auc']:>12.4f} {dp_results['confidence_auc']:>12.4f}")
        print(f"{'Loss AUC':<30} {baseline_results['loss_auc']:>12.4f} {dp_results['loss_auc']:>12.4f}")

        if use_shadow:
            print(f"{'Shadow Attack AUC':<30} {baseline_results['shadow_attack_auc']:>12.4f} {dp_results['shadow_attack_auc']:>12.4f}")
            print(f"{'Shadow Attack Accuracy':<30} {baseline_results['shadow_attack_accuracy']:>12.4f} {dp_results['shadow_attack_accuracy']:>12.4f}")

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


# Backward compatibility alias
evaluate_mia = evaluate_threshold_mia


def main():
    """Command-line interface for MIA evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate membership inference attack vulnerability"
    )
    parser.add_argument("--dataset", type=str, default="adult",
                        choices=SUPPORTED_DATASETS,
                        help="Dataset to use (default: adult)")
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
    parser.add_argument("--shadow", action="store_true",
                        help="Run shadow-model attack")
    parser.add_argument("--n-shadows", type=int, default=5,
                        help="Number of shadow models (default: 5)")
    args = parser.parse_args()

    if args.compare:
        compare_baseline_vs_dp(
            dataset=args.dataset, epochs=args.epochs, epsilon=args.epsilon,
            use_shadow=args.shadow, n_shadows=args.n_shadows
        )
    else:
        run_mia_evaluation(
            dataset=args.dataset,
            model_path=args.model,
            train_new=args.model is None,
            epochs=args.epochs,
            use_dp=args.use_dp,
            epsilon=args.epsilon,
            use_shadow=args.shadow,
            n_shadows=args.n_shadows,
            verbose=True
        )


if __name__ == "__main__":
    main()
