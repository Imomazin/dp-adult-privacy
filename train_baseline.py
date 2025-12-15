"""
Baseline (non-private) training script for Adult Census classification.
Trains a standard neural network without differential privacy.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from data import get_data_loaders
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
    Evaluate model on a dataset.

    Args:
        model: Neural network model
        data_loader: Data loader to evaluate on
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        loss: Average loss
        accuracy: Classification accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def train_baseline(epochs=None, lr=None, save_path=None, verbose=True):
    """
    Main training function for baseline (non-private) model.

    Args:
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

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    # Load data
    if verbose:
        print("\nLoading data...")
    train_loader, val_loader, test_loader, input_dim = get_data_loaders()

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
    }

    # Training loop
    if verbose:
        print(f"\nTraining for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

    # Final evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    if verbose:
        print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    history["test_loss"] = test_loss
    history["test_acc"] = test_acc

    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "history": history,
        }, save_path)
        if verbose:
            print(f"\nModel saved to {save_path}")

    return model, history


def main():
    """Command-line interface for baseline training."""
    parser = argparse.ArgumentParser(
        description="Train baseline (non-private) model on Adult dataset"
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--save", type=str, default="checkpoints/baseline.pt",
                        help="Path to save trained model")
    args = parser.parse_args()

    train_baseline(
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save,
        verbose=True
    )


if __name__ == "__main__":
    main()
