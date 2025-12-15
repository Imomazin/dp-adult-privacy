"""
Data loading and preprocessing for the Adult Census Income dataset.

This script downloads, preprocesses, and splits the Adult dataset for
privacy-preserving machine learning experiments.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader

# URLs for the Adult dataset from UCI ML Repository
TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# Column names for the Adult dataset
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Categorical columns that need encoding
CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

# Numerical columns that need scaling
NUMERICAL_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]


def download_data(data_dir: str = "data") -> tuple[str, str]:
    """
    Download the Adult dataset if not already present.

    Args:
        data_dir: Directory to store the downloaded data

    Returns:
        Tuple of paths to train and test files
    """
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "adult.data")
    test_path = os.path.join(data_dir, "adult.test")

    # Download training data
    if not os.path.exists(train_path):
        print(f"Downloading training data to {train_path}...")
        urllib.request.urlretrieve(TRAIN_URL, train_path)
        print("Download complete.")
    else:
        print(f"Training data already exists at {train_path}")

    # Download test data
    if not os.path.exists(test_path):
        print(f"Downloading test data to {test_path}...")
        urllib.request.urlretrieve(TEST_URL, test_path)
        print("Download complete.")
    else:
        print(f"Test data already exists at {test_path}")

    return train_path, test_path


def load_raw_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSV data into pandas DataFrames.

    Args:
        train_path: Path to training data file
        test_path: Path to test data file

    Returns:
        Tuple of training and test DataFrames
    """
    # Load training data
    train_df = pd.read_csv(
        train_path,
        names=COLUMN_NAMES,
        sep=r",\s*",
        engine="python",
        na_values="?"
    )

    # Load test data (skip first row which contains a comment)
    test_df = pd.read_csv(
        test_path,
        names=COLUMN_NAMES,
        sep=r",\s*",
        engine="python",
        na_values="?",
        skiprows=1
    )

    # Clean income labels (test set has trailing period)
    train_df["income"] = train_df["income"].str.strip().str.rstrip(".")
    test_df["income"] = test_df["income"].str.strip().str.rstrip(".")

    return train_df, test_df


def preprocess_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the Adult dataset: handle missing values, encode categoricals,
    and scale numerical features.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as numpy arrays
    """
    # Combine for consistent preprocessing
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Drop rows with missing values
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    print(f"Training samples after dropping NA: {len(train_df)}")
    print(f"Test samples after dropping NA: {len(test_df)}")

    # Encode target variable: <=50K -> 0, >50K -> 1
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["income"])
    y_test = label_encoder.transform(test_df["income"])

    # Remove target from features
    train_df = train_df.drop("income", axis=1)
    test_df = test_df.drop("income", axis=1)

    # Encode categorical columns using one-hot encoding
    train_encoded = pd.get_dummies(train_df, columns=CATEGORICAL_COLS)
    test_encoded = pd.get_dummies(test_df, columns=CATEGORICAL_COLS)

    # Align columns (some categories may only appear in train or test)
    train_encoded, test_encoded = train_encoded.align(
        test_encoded, join="left", axis=1, fill_value=0
    )

    # Scale numerical features
    scaler = StandardScaler()
    train_encoded[NUMERICAL_COLS] = scaler.fit_transform(train_encoded[NUMERICAL_COLS])
    test_encoded[NUMERICAL_COLS] = scaler.transform(test_encoded[NUMERICAL_COLS])

    X_train = train_encoded.values.astype(np.float32)
    X_test = test_encoded.values.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    print(f"Feature dimension: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def create_data_loaders(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 256,
    val_split: float = 0.1,
    seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training, validation, and test sets.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        batch_size: Batch size for DataLoaders
        val_split: Fraction of training data to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=seed, stratify=y_train
    )

    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def get_data_loaders(
    data_dir: str = "data",
    batch_size: int = 256,
    val_split: float = 0.1,
    seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Main function to download, preprocess, and create DataLoaders.

    Args:
        data_dir: Directory for data storage
        batch_size: Batch size for DataLoaders
        val_split: Fraction for validation split
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader, input_dim)
    """
    # Download data
    train_path, test_path = download_data(data_dir)

    # Load raw data
    train_df, test_df = load_raw_data(train_path, test_path)

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_test, y_train, y_test,
        batch_size=batch_size, val_split=val_split, seed=seed
    )

    input_dim = X_train.shape[1]

    return train_loader, val_loader, test_loader, input_dim


if __name__ == "__main__":
    # Test the data loading pipeline
    print("Testing data loading pipeline...")
    train_loader, val_loader, test_loader, input_dim = get_data_loaders()

    # Print sample batch info
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nSample batch shape: X={X_batch.shape}, y={y_batch.shape}")
    print(f"Input dimension: {input_dim}")
    print(f"Label distribution in batch: {y_batch.mean().item():.3f} positive")
