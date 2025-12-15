"""
Baseline (non-private) training script for the Adult dataset.

This script trains a standard neural network without differential privacy
to establish a baseline for comparison with DP-SGD training.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_loader import get_data_loaders
from model import create_model, count_parameters


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Train for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: Neural network model
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            # Calculate accuracy (threshold at 0.5)
            predictions = (outputs >= 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def train_baseline(
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    hidden_dim: int = 128,
    seed: int = 42,
    save_dir: str = "checkpoints"
) -> dict:
    """
    Train a baseline (non-private) model.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for optimizer
        hidden_dim: Hidden dimension of the model
        seed: Random seed for reproducibility
        save_dir: Directory to save checkpoints

    Returns:
        Dictionary with training history and final metrics
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader, input_dim = get_data_loaders(
        batch_size=batch_size, seed=seed
    )

    # Create model
    print("\n--- Creating Model ---")
    model = create_model(input_dim, hidden_dim).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    # Training loop
    print("\n--- Training ---")
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "baseline_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim
            }, save_path)

    # Final test evaluation
    print("\n--- Final Test Evaluation ---")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Save final model
    final_path = os.path.join(save_dir, "baseline_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "test_accuracy": test_acc
    }, final_path)
    print(f"Model saved to {final_path}")

    return {
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc
    }


def main():
    parser = argparse.ArgumentParser(description="Train baseline (non-private) model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")

    args = parser.parse_args()

    results = train_baseline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        save_dir=args.save_dir
    )

    print("\n--- Training Complete ---")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Final test accuracy: {results['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
