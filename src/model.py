"""
Neural network model for the Adult dataset classification task.

Simple fully-connected network compatible with Opacus for DP training.
"""

import torch
import torch.nn as nn


class AdultClassifier(nn.Module):
    """
    Simple fully-connected neural network for binary classification.

    Architecture:
        Input -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(1) -> Sigmoid

    Note: This architecture is compatible with Opacus DP-SGD. We avoid:
        - BatchNorm (not compatible with DP)
        - In-place operations
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """
        Initialize the classifier.

        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units in each layer
        """
        super(AdultClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) with probabilities
        """
        return self.network(x)


def create_model(input_dim: int, hidden_dim: int = 128) -> AdultClassifier:
    """
    Factory function to create the model.

    Args:
        input_dim: Number of input features
        hidden_dim: Hidden layer dimension

    Returns:
        Initialized AdultClassifier model
    """
    return AdultClassifier(input_dim, hidden_dim)


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
