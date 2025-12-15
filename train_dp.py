"""
Differentially private training using DP-SGD with Opacus.
Trains a neural network with formal privacy guarantees.
Supports multiple datasets: adult, credit_default.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import roc_auc_score, average_precision_score

import config
from data import get_data_loaders, SUPPORTED_DATASETS
from model import create_model, count_parameters


def train_epoch_dp(model, train_loader, criterion, optimizer, device,
                   max_physical_batch_size=128):
    """
    Train for one epoch with differential privacy.

    Uses BatchMemoryManager to handle large logical batches while
    keeping physical batches small enough to fit in memory.

    Args:
        model: Neural network model (wrapped by PrivacyEngine)
        train_loader: Training data loader
        criterion: Loss function
        optimizer: DP optimizer (wrapped by PrivacyEngine)
        device: Device to train on
        max_physical_batch_size: Maximum batch size for memory

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    # BatchMemoryManager handles gradient accumulation for large batches
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer
    ) as memory_safe_loader:
        for features, labels in memory_safe_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset with comprehensive metrics.

    Args:
        model: Neural network model
        data_loader: Data loader to evaluate on
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        metrics: Dictionary with loss, accuracy, auc_roc, pr_auc
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)
            probs = torch.sigmoid(outputs)

            total_loss += loss.item() * len(labels)
            predictions = (probs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Compute AUC-ROC and PR-AUC
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_roc = 0.5

    try:
        pr_auc = average_precision_score(all_labels, all_probs)
    except ValueError:
        pr_auc = all_labels.mean()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
    }


def train_dp(dataset="adult", epsilon=None, delta=None, max_grad_norm=None,
             epochs=None, lr=None, save_path=None, verbose=True):
    """
    Main training function with differential privacy.

    Uses Opacus PrivacyEngine to:
    1. Clip per-sample gradients
    2. Add calibrated Gaussian noise
    3. Track privacy budget (epsilon) spent

    Args:
        dataset: Dataset name ("adult" or "credit_default")
        epsilon: Target privacy budget (defaults to config.EPSILON)
        delta: Privacy parameter delta (defaults to config.DELTA)
        max_grad_norm: Gradient clipping bound (defaults to config.MAX_GRAD_NORM)
        epochs: Number of training epochs (defaults to config.EPOCHS)
        lr: Learning rate (defaults to config.LEARNING_RATE)
        save_path: Path to save trained model (optional)
        verbose: Whether to print progress

    Returns:
        model: Trained model
        history: Dictionary with training history
        final_epsilon: Actual epsilon spent
    """
    # Use defaults from config if not specified
    if epsilon is None:
        epsilon = config.EPSILON
    if delta is None:
        delta = config.DELTA
    if max_grad_norm is None:
        max_grad_norm = config.MAX_GRAD_NORM
    if epochs is None:
        epochs = config.EPOCHS
    if lr is None:
        lr = config.LEARNING_RATE

    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    # Load data
    if verbose:
        print(f"\nLoading {dataset} data...")
    train_loader, val_loader, test_loader, input_dim = get_data_loaders(dataset=dataset)

    # Create model
    model = create_model(input_dim).to(device)
    if verbose:
        print(f"Model parameters: {count_parameters(model):,}")

    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Wrap model and optimizer with PrivacyEngine
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=max_grad_norm,
    )

    if verbose:
        print(f"\nPrivacy parameters:")
        print(f"  Target epsilon: {epsilon}")
        print(f"  Delta: {delta}")
        print(f"  Max grad norm: {max_grad_norm}")
        print(f"  Noise multiplier: {optimizer.noise_multiplier:.4f}")

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc_roc": [],
        "val_pr_auc": [],
        "epsilon": [],
    }

    # Training loop
    if verbose:
        print(f"\nTraining for {epochs} epochs with DP-SGD...")

    for epoch in range(epochs):
        # Train with DP
        train_loss = train_epoch_dp(
            model, train_loader, criterion, optimizer, device
        )

        # Evaluate on validation set (no privacy cost)
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Get current epsilon spent
        current_epsilon = privacy_engine.get_epsilon(delta=delta)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auc_roc"].append(val_metrics["auc_roc"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])
        history["epsilon"].append(current_epsilon)

        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                  f"eps: {current_epsilon:.2f}")

    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    final_epsilon = privacy_engine.get_epsilon(delta=delta)

    if verbose:
        print(f"\n{'='*50}")
        print("Test Set Results:")
        print(f"  Loss:    {test_metrics['loss']:.4f}")
        print(f"  Acc:     {test_metrics['accuracy']:.4f}")
        print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"  PR-AUC:  {test_metrics['pr_auc']:.4f}")
        print(f"  Final privacy: (eps={final_epsilon:.2f}, delta={delta})")
        print(f"{'='*50}")

    history["test_loss"] = test_metrics["loss"]
    history["test_acc"] = test_metrics["accuracy"]
    history["test_auc_roc"] = test_metrics["auc_roc"]
    history["test_pr_auc"] = test_metrics["pr_auc"]
    history["final_epsilon"] = final_epsilon
    history["delta"] = delta
    history["dataset"] = dataset

    # Save model if path provided
    # Note: We save the underlying model, not the GradSampleModule wrapper
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": model._module.state_dict(),
            "input_dim": input_dim,
            "history": history,
            "epsilon": final_epsilon,
            "delta": delta,
            "dataset": dataset,
        }, save_path)
        if verbose:
            print(f"\nModel saved to {save_path}")

    return model, history, final_epsilon


def main():
    """Command-line interface for DP training."""
    parser = argparse.ArgumentParser(
        description="Train differentially private model"
    )
    parser.add_argument("--dataset", type=str, default="adult",
                        choices=SUPPORTED_DATASETS,
                        help="Dataset to use (default: adult)")
    parser.add_argument("--epsilon", type=float, default=config.EPSILON,
                        help="Target privacy budget epsilon")
    parser.add_argument("--delta", type=float, default=config.DELTA,
                        help="Privacy parameter delta")
    parser.add_argument("--max-grad-norm", type=float, default=config.MAX_GRAD_NORM,
                        help="Gradient clipping bound")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--save", type=str, default="checkpoints/dp_model.pt",
                        help="Path to save trained model")
    args = parser.parse_args()

    train_dp(
        dataset=args.dataset,
        epsilon=args.epsilon,
        delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save,
        verbose=True
    )


if __name__ == "__main__":
    main()
