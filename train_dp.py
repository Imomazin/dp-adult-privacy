"""
Differentially private training using DP-SGD with Opacus.
Trains a neural network with formal privacy guarantees.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm

import config
from data import get_data_loaders
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


def train_dp(epsilon=None, delta=None, max_grad_norm=None,
             epochs=None, lr=None, save_path=None, verbose=True):
    """
    Main training function with differential privacy.

    Uses Opacus PrivacyEngine to:
    1. Clip per-sample gradients
    2. Add calibrated Gaussian noise
    3. Track privacy budget (epsilon) spent

    Args:
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
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Get current epsilon spent
        current_epsilon = privacy_engine.get_epsilon(delta=delta)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epsilon"].append(current_epsilon)

        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"ε: {current_epsilon:.2f}")

    # Final evaluation on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    final_epsilon = privacy_engine.get_epsilon(delta=delta)

    if verbose:
        print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(f"Final privacy spent: (ε={final_epsilon:.2f}, δ={delta})")

    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    history["final_epsilon"] = final_epsilon
    history["delta"] = delta

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
        }, save_path)
        if verbose:
            print(f"\nModel saved to {save_path}")

    return model, history, final_epsilon


def main():
    """Command-line interface for DP training."""
    parser = argparse.ArgumentParser(
        description="Train differentially private model on Adult dataset"
    )
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
