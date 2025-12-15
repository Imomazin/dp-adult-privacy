"""
Data loading and preprocessing for privacy-preserving ML experiments.

Supported datasets:
- Adult Census Income (default)
- UCI Bank Marketing
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader

# =============================================================================
# Adult Dataset Configuration
# =============================================================================
ADULT_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

ADULT_COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

ADULT_CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

ADULT_NUMERICAL_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week"
]

# =============================================================================
# Bank Marketing Dataset Configuration
# =============================================================================
BANK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"

BANK_COLUMN_NAMES = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome", "y"
]

BANK_CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome"
]

BANK_NUMERICAL_COLS = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous"
]


# =============================================================================
# Adult Dataset Functions
# =============================================================================

def download_adult_data(data_dir: str = "data") -> tuple[str, str]:
    """Download the Adult dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "adult.data")
    test_path = os.path.join(data_dir, "adult.test")

    if not os.path.exists(train_path):
        print(f"Downloading Adult training data...")
        urllib.request.urlretrieve(ADULT_TRAIN_URL, train_path)

    if not os.path.exists(test_path):
        print(f"Downloading Adult test data...")
        urllib.request.urlretrieve(ADULT_TEST_URL, test_path)

    return train_path, test_path


def load_adult_data(data_dir: str = "data") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the Adult dataset."""
    train_path, test_path = download_adult_data(data_dir)

    # Load training data
    train_df = pd.read_csv(
        train_path, names=ADULT_COLUMN_NAMES,
        sep=r",\s*", engine="python", na_values="?"
    )

    # Load test data (skip first row which contains a comment)
    test_df = pd.read_csv(
        test_path, names=ADULT_COLUMN_NAMES,
        sep=r",\s*", engine="python", na_values="?", skiprows=1
    )

    # Clean income labels (test set has trailing period)
    train_df["income"] = train_df["income"].str.strip().str.rstrip(".")
    test_df["income"] = test_df["income"].str.strip().str.rstrip(".")

    # Drop rows with missing values
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    print(f"Adult dataset - Train: {len(train_df)}, Test: {len(test_df)}")

    # Encode target variable
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["income"])
    y_test = label_encoder.transform(test_df["income"])

    # Remove target from features
    train_df = train_df.drop("income", axis=1)
    test_df = test_df.drop("income", axis=1)

    # One-hot encode categorical columns
    train_encoded = pd.get_dummies(train_df, columns=ADULT_CATEGORICAL_COLS)
    test_encoded = pd.get_dummies(test_df, columns=ADULT_CATEGORICAL_COLS)

    # Align columns
    train_encoded, test_encoded = train_encoded.align(
        test_encoded, join="left", axis=1, fill_value=0
    )

    # Scale numerical features
    scaler = StandardScaler()
    train_encoded[ADULT_NUMERICAL_COLS] = scaler.fit_transform(train_encoded[ADULT_NUMERICAL_COLS])
    test_encoded[ADULT_NUMERICAL_COLS] = scaler.transform(test_encoded[ADULT_NUMERICAL_COLS])

    X_train = train_encoded.values.astype(np.float32)
    X_test = test_encoded.values.astype(np.float32)

    print(f"Feature dimension: {X_train.shape[1]}")

    return X_train, X_test, y_train.astype(np.float32), y_test.astype(np.float32)


# =============================================================================
# Bank Marketing Dataset Functions
# =============================================================================

def download_bank_data(data_dir: str = "data") -> str:
    """Download the Bank Marketing dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "bank-full.csv")
    zip_path = os.path.join(data_dir, "bank.zip")

    if not os.path.exists(csv_path):
        print(f"Downloading Bank Marketing data...")
        urllib.request.urlretrieve(BANK_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

    return csv_path


def load_bank_data(data_dir: str = "data", test_size: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the Bank Marketing dataset."""
    csv_path = download_bank_data(data_dir)

    # Load data
    df = pd.read_csv(csv_path, sep=";")

    print(f"Bank dataset - Total samples: {len(df)}")

    # Encode target variable: no -> 0, yes -> 1
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["y"])

    # Remove target from features
    df = df.drop("y", axis=1)

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=BANK_CATEGORICAL_COLS)

    # Scale numerical features
    scaler = StandardScaler()
    df_encoded[BANK_NUMERICAL_COLS] = scaler.fit_transform(df_encoded[BANK_NUMERICAL_COLS])

    X = df_encoded.values.astype(np.float32)
    y = y.astype(np.float32)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"Bank dataset - Train: {len(X_train)}, Test: {len(X_test)}")
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
    dataset: str = "adult",
    data_dir: str = "data",
    batch_size: int = 256,
    val_split: float = 0.1,
    seed: int = 42
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Main function to download, preprocess, and create DataLoaders.

    Args:
        dataset: Dataset name ('adult' or 'bank')
        data_dir: Directory for data storage
        batch_size: Batch size for DataLoaders
        val_split: Fraction for validation split
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader, input_dim)
    """
    # Load data based on dataset choice
    if dataset == "adult":
        X_train, X_test, y_train, y_test = load_adult_data(data_dir)
    elif dataset == "bank":
        X_train, X_test, y_train, y_test = load_bank_data(data_dir, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'adult' or 'bank'.")

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_test, y_train, y_test,
        batch_size=batch_size, val_split=val_split, seed=seed
    )

    input_dim = X_train.shape[1]

    return train_loader, val_loader, test_loader, input_dim


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test data loading")
    parser.add_argument("--dataset", type=str, default="adult", choices=["adult", "bank"])
    args = parser.parse_args()

    print(f"Testing {args.dataset} dataset loading...")
    train_loader, val_loader, test_loader, input_dim = get_data_loaders(dataset=args.dataset)

    X_batch, y_batch = next(iter(train_loader))
    print(f"\nSample batch shape: X={X_batch.shape}, y={y_batch.shape}")
    print(f"Input dimension: {input_dim}")
    print(f"Label distribution in batch: {y_batch.mean().item():.3f} positive")
