"""
Neural network model for Adult Census classification.
Simple MLP architecture compatible with Opacus DP-SGD.
"""

import torch
import torch.nn as nn

import config


class AdultMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for binary classification.

    Architecture:
        Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid

    Note: Uses nn.Linear layers which are compatible with Opacus.
    Batch normalization is avoided as it can leak privacy.
    """

    def __init__(self, input_dim: int, hidden_dim: int = None):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension (defaults to config.HIDDEN_DIM)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = config.HIDDEN_DIM

        # Simple 2-hidden-layer MLP
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            # No sigmoid here - use BCEWithLogitsLoss for numerical stability
        )

    def forward(self, x):
        """Forward pass returning logits."""
        return self.network(x).squeeze(-1)


def create_model(input_dim: int) -> AdultMLP:
    """
    Factory function to create a new model instance.

    Args:
        input_dim: Number of input features

    Returns:
        Initialized AdultMLP model
    """
    model = AdultMLP(input_dim=input_dim)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    test_input_dim = 108  # Approximate dimension for Adult dataset

    model = create_model(test_input_dim)
    print(f"Model architecture:\n{model}")
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, test_input_dim)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
