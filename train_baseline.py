"""
Baseline (non-private) training script for tabular classification.
Trains a standard neural network without differential privacy.
Supports multiple datasets: adult, credit_default.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score

import config
from data import get_data_loaders, SUPPORTED_DATASETS
from model import create_model, count_parameters


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for features, labels in train_loader:
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
        auc_roc = 0.5  # Default if only one class present

    try:
        pr_auc = average_precision_score(all_labels, all_probs)
    except ValueError:
        pr_auc = all_labels.mean()  # Default to positive class ratio

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "auc_roc": auc_roc,
        "pr_auc": pr_auc,
    }


def train_baseline(dataset="adult", epochs=None, lr=None, save_path=None, verbose=True):
    """
    Main training function for baseline (non-private) model.

    Args:
        dataset: Dataset name ("adult" or "credit_default")
        epochs: Number of training epochs (defaults to config.EPOCHS)
        lr: Learning rate (defaults to config.LEARNING_RATE)
        save_path: Path to save trained model (optional)
        verbose: Whether to print progress

    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    # Use defaults from config if not specified
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

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc_roc": [],
        "val_pr_auc": [],
    }

    # Training loop
    if verbose:
        print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_auc_roc"].append(val_metrics["auc_roc"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])

        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val AUC: {val_metrics['auc_roc']:.4f}")

    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    if verbose:
        print(f"\n{'='*50}")
        print("Test Set Results:")
        print(f"  Loss:    {test_metrics['loss']:.4f}")
        print(f"  Acc:     {test_metrics['accuracy']:.4f}")
        print(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"  PR-AUC:  {test_metrics['pr_auc']:.4f}")
        print(f"{'='*50}")

    history["test_loss"] = test_metrics["loss"]
    history["test_acc"] = test_metrics["accuracy"]
    history["test_auc_roc"] = test_metrics["auc_roc"]
    history["test_pr_auc"] = test_metrics["pr_auc"]
    history["dataset"] = dataset

    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "history": history,
            "dataset": dataset,
        }, save_path)
        if verbose:
            print(f"\nModel saved to {save_path}")

    return model, history


def main():
    """Command-line interface for baseline training."""
    parser = argparse.ArgumentParser(
        description="Train baseline (non-private) model"
    )
    parser.add_argument("--dataset", type=str, default="adult",
                        choices=SUPPORTED_DATASETS,
                        help="Dataset to use (default: adult)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--save", type=str, default="checkpoints/baseline.pt",
                        help="Path to save trained model")
    args = parser.parse_args()

    train_baseline(
        dataset=args.dataset,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save,
        verbose=True
    )


if __name__ == "__main__":
    main()
