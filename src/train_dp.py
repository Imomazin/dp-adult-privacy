"""
Differentially Private (DP-SGD) training script.

This script trains a neural network with differential privacy guarantees
using Opacus, implementing the DP-SGD algorithm.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm

from data_loader import get_data_loaders
from model import create_model, count_parameters


def train_epoch_dp(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_physical_batch_size: int = 128
) -> float:
    """
    Train for one epoch with differential privacy.

    Uses BatchMemoryManager to handle large logical batches with smaller
    physical batches (gradient accumulation for memory efficiency).

    Args:
        model: Neural network model (wrapped by PrivacyEngine)
        train_loader: Training data loader
        optimizer: DP optimizer (wrapped by PrivacyEngine)
        criterion: Loss function
        device: Device to train on
        max_physical_batch_size: Maximum physical batch size for memory

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Use BatchMemoryManager for memory-efficient training with large batches
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer
    ) as memory_safe_loader:
        for X_batch, y_batch in memory_safe_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


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


def train_dp(
    dataset: str = "adult",
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    hidden_dim: int = 128,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    save_dir: str = "checkpoints"
) -> dict:
    """
    Train a differentially private model using DP-SGD.

    Args:
        dataset: Dataset name ('adult' or 'bank')
        epochs: Number of training epochs
        batch_size: Batch size (logical batch size for DP)
        learning_rate: Learning rate for optimizer
        hidden_dim: Hidden dimension of the model
        target_epsilon: Target epsilon for (epsilon, delta)-DP
        target_delta: Target delta for (epsilon, delta)-DP
        max_grad_norm: Maximum gradient norm for clipping
        seed: Random seed for reproducibility
        save_dir: Directory to save checkpoints

    Returns:
        Dictionary with training history, final metrics, and privacy spent
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"\n--- Loading Data ({dataset}) ---")
    train_loader, val_loader, test_loader, input_dim = get_data_loaders(
        dataset=dataset, batch_size=batch_size, seed=seed
    )

    # Create model
    print("\n--- Creating Model ---")
    model = create_model(input_dim, hidden_dim).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Setup differential privacy with Opacus
    print("\n--- Setting up Differential Privacy ---")
    print(f"Target epsilon: {target_epsilon}")
    print(f"Target delta: {target_delta}")
    print(f"Max gradient norm: {max_grad_norm}")

    privacy_engine = PrivacyEngine()

    # Make model, optimizer, and data_loader private
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )

    print(f"Using noise multiplier: {optimizer.noise_multiplier:.4f}")

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "epsilon": []
    }

    # Training loop
    print("\n--- Training with DP-SGD ---")
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch_dp(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Get current privacy spent
        epsilon = privacy_engine.get_epsilon(delta=target_delta)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["epsilon"].append(epsilon)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"ε: {epsilon:.2f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "dp_best.pt")
            # Save the underlying module (unwrap GradSampleModule)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model._module.state_dict(),
                "val_accuracy": val_acc,
                "epsilon": epsilon,
                "input_dim": input_dim,
                "hidden_dim": hidden_dim
            }, save_path)

    # Final privacy accounting
    final_epsilon = privacy_engine.get_epsilon(delta=target_delta)
    print(f"\n--- Final Privacy Guarantee ---")
    print(f"(ε, δ)-DP: ({final_epsilon:.2f}, {target_delta})")

    # Final test evaluation
    print("\n--- Final Test Evaluation ---")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Save final model
    final_path = os.path.join(save_dir, "dp_final.pt")
    torch.save({
        "model_state_dict": model._module.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "test_accuracy": test_acc,
        "final_epsilon": final_epsilon,
        "target_delta": target_delta
    }, final_path)
    print(f"Model saved to {final_path}")

    return {
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "final_epsilon": final_epsilon,
        "target_delta": target_delta
    }


def main():
    parser = argparse.ArgumentParser(description="Train DP model with DP-SGD")
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult", "bank"],
                        help="Dataset to use (default: adult)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--epsilon", type=float, default=8.0, help="Target epsilon")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target delta")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")

    args = parser.parse_args()

    results = train_dp(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        save_dir=args.save_dir
    )

    print("\n--- Training Complete ---")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Final test accuracy: {results['test_accuracy']:.4f}")
    print(f"Privacy guarantee: (ε={results['final_epsilon']:.2f}, δ={results['target_delta']})")


if __name__ == "__main__":
    main()
