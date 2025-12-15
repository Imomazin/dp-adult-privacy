"""
Data loading and preprocessing for Adult Census Income dataset.
Handles downloading, cleaning, encoding, and creating PyTorch DataLoaders.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import config


# Column names for the Adult dataset (UCI ML Repository)
COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

# Categorical columns that need encoding
CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country"
]

# Numerical columns that need scaling
NUMERICAL_COLUMNS = [
    "age", "fnlwgt", "education_num", "capital_gain",
    "capital_loss", "hours_per_week"
]


class AdultDataset(Dataset):
    """PyTorch Dataset wrapper for Adult Census data."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: Preprocessed feature matrix (N x D)
            labels: Binary labels (N,)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def download_data():
    """Download Adult dataset from UCI ML Repository if not present."""
    os.makedirs(config.DATA_DIR, exist_ok=True)

    train_path = os.path.join(config.DATA_DIR, "adult.data")
    test_path = os.path.join(config.DATA_DIR, "adult.test")

    if not os.path.exists(train_path):
        print("Downloading training data...")
        urllib.request.urlretrieve(config.DATA_URL, train_path)

    if not os.path.exists(test_path):
        print("Downloading test data...")
        urllib.request.urlretrieve(config.DATA_TEST_URL, test_path)

    return train_path, test_path


def load_raw_data(filepath: str, is_test: bool = False) -> pd.DataFrame:
    """
    Load raw CSV data into pandas DataFrame.

    Args:
        filepath: Path to the data file
        is_test: Whether this is the test file (has extra header line)

    Returns:
        DataFrame with raw data
    """
    # Test file has a header line to skip
    skiprows = 1 if is_test else 0

    df = pd.read_csv(
        filepath,
        names=COLUMN_NAMES,
        sep=r",\s*",
        engine="python",
        skiprows=skiprows,
        na_values="?"
    )
    return df


def preprocess_data(df: pd.DataFrame, fit_encoders: bool = True,
                    encoders: dict = None, scaler: StandardScaler = None):
    """
    Preprocess the Adult dataset:
    - Handle missing values
    - Encode categorical variables (one-hot)
    - Scale numerical features
    - Convert labels to binary

    Args:
        df: Raw DataFrame
        fit_encoders: Whether to fit new encoders (True for training data)
        encoders: Pre-fitted encoders (for test data)
        scaler: Pre-fitted scaler (for test data)

    Returns:
        features: Preprocessed feature matrix
        labels: Binary labels
        encoders: Fitted encoders (for reuse on test data)
        scaler: Fitted scaler (for reuse on test data)
    """
    df = df.copy()

    # Drop rows with missing values (simple approach for research)
    df = df.dropna()

    # Convert income label to binary (>50K = 1, <=50K = 0)
    # Note: test file has trailing period in labels
    df["income"] = df["income"].str.replace(".", "", regex=False)
    labels = (df["income"].str.strip() == ">50K").astype(int).values

    # Initialize encoders if fitting
    if fit_encoders:
        encoders = {}
        scaler = StandardScaler()

    # One-hot encode categorical columns
    encoded_cats = []
    for col in CATEGORICAL_COLUMNS:
        if fit_encoders:
            # Get unique values and create mapping
            unique_vals = df[col].unique()
            encoders[col] = {val: i for i, val in enumerate(unique_vals)}

        # Create one-hot encoding
        n_categories = len(encoders[col])
        one_hot = np.zeros((len(df), n_categories))
        for i, val in enumerate(df[col].values):
            if val in encoders[col]:
                one_hot[i, encoders[col][val]] = 1
            # Unknown categories get all zeros
        encoded_cats.append(one_hot)

    # Scale numerical columns
    numerical_data = df[NUMERICAL_COLUMNS].values
    if fit_encoders:
        numerical_scaled = scaler.fit_transform(numerical_data)
    else:
        numerical_scaled = scaler.transform(numerical_data)

    # Combine all features
    features = np.hstack([numerical_scaled] + encoded_cats)

    return features, labels, encoders, scaler


def get_data_loaders(batch_size: int = None, val_split: float = 0.1):
    """
    Main function to get train/val/test DataLoaders.

    Args:
        batch_size: Batch size (defaults to config.BATCH_SIZE)
        val_split: Fraction of training data to use for validation

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        input_dim: Input feature dimension
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # Download data if needed
    train_path, test_path = download_data()

    # Load raw data
    train_df = load_raw_data(train_path, is_test=False)
    test_df = load_raw_data(test_path, is_test=True)

    # Preprocess training data (fit encoders)
    train_features, train_labels, encoders, scaler = preprocess_data(
        train_df, fit_encoders=True
    )

    # Preprocess test data (use fitted encoders)
    test_features, test_labels, _, _ = preprocess_data(
        test_df, fit_encoders=False, encoders=encoders, scaler=scaler
    )

    print(f"Training samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Feature dimension: {train_features.shape[1]}")
    print(f"Positive class ratio (train): {train_labels.mean():.3f}")

    # Create datasets
    full_train_dataset = AdultDataset(train_features, train_labels)
    test_dataset = AdultDataset(test_features, test_labels)

    # Split training data into train/val
    n_val = int(len(full_train_dataset) * val_split)
    n_train = len(full_train_dataset) - n_val

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [n_train, n_val], generator=generator
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    input_dim = train_features.shape[1]

    return train_loader, val_loader, test_loader, input_dim


def get_member_nonmember_split(member_ratio: float = 0.5):
    """
    Create a member/non-member split for membership inference attacks.

    For MIA evaluation:
    - Members: Data used to train the target model
    - Non-members: Data NOT used to train (held out)

    Args:
        member_ratio: Fraction of training data to use as members

    Returns:
        member_dataset: Dataset of members (used for training target)
        nonmember_dataset: Dataset of non-members (held out)
        test_dataset: Original test set
        input_dim: Input feature dimension
    """
    # Download and preprocess
    train_path, test_path = download_data()
    train_df = load_raw_data(train_path, is_test=False)
    test_df = load_raw_data(test_path, is_test=True)

    train_features, train_labels, encoders, scaler = preprocess_data(
        train_df, fit_encoders=True
    )
    test_features, test_labels, _, _ = preprocess_data(
        test_df, fit_encoders=False, encoders=encoders, scaler=scaler
    )

    # Create full training dataset
    full_dataset = AdultDataset(train_features, train_labels)
    test_dataset = AdultDataset(test_features, test_labels)

    # Split into members and non-members
    n_members = int(len(full_dataset) * member_ratio)
    n_nonmembers = len(full_dataset) - n_members

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    member_dataset, nonmember_dataset = random_split(
        full_dataset, [n_members, n_nonmembers], generator=generator
    )

    input_dim = train_features.shape[1]

    return member_dataset, nonmember_dataset, test_dataset, input_dim


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")
    train_loader, val_loader, test_loader, input_dim = get_data_loaders()

    # Check a batch
    for features, labels in train_loader:
        print(f"\nBatch shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label distribution: {labels.mean():.3f} positive")
        break

    print(f"\nInput dimension: {input_dim}")
    print("Data loading test complete!")
