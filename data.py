"""
Data loading and preprocessing for multiple datasets.
Supports: Adult Census Income, Credit Default (UCI).
Handles downloading, cleaning, encoding, and creating PyTorch DataLoaders.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

import config


# =============================================================================
# Dataset Registry
# =============================================================================
SUPPORTED_DATASETS = ["adult", "credit_default"]


# =============================================================================
# Adult Dataset Configuration
# =============================================================================
ADULT_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

ADULT_COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

ADULT_CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country"
]

ADULT_NUMERICAL_COLUMNS = [
    "age", "fnlwgt", "education_num", "capital_gain",
    "capital_loss", "hours_per_week"
]

# Sensitive attributes for fairness evaluation
ADULT_SENSITIVE_ATTRS = ["sex", "race"]


# =============================================================================
# Credit Default Dataset Configuration
# =============================================================================
CREDIT_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

CREDIT_COLUMN_NAMES = [
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default_payment_next_month"
]

CREDIT_CATEGORICAL_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE"]

CREDIT_NUMERICAL_COLUMNS = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]


# =============================================================================
# Generic Dataset Class
# =============================================================================
class TabularDataset(Dataset):
    """PyTorch Dataset wrapper for tabular data with optional sensitive attributes."""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 sensitive_attrs: dict = None):
        """
        Args:
            features: Preprocessed feature matrix (N x D)
            labels: Binary labels (N,)
            sensitive_attrs: Dictionary mapping attribute name to values (N,)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.sensitive_attrs = sensitive_attrs or {}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_sensitive_attr(self, attr_name: str) -> np.ndarray:
        """Get sensitive attribute values."""
        if attr_name in self.sensitive_attrs:
            return self.sensitive_attrs[attr_name]
        return None


# Alias for backward compatibility
AdultDataset = TabularDataset


# =============================================================================
# Adult Dataset Functions
# =============================================================================
def download_adult_data():
    """Download Adult dataset from UCI ML Repository if not present."""
    os.makedirs(config.DATA_DIR, exist_ok=True)

    train_path = os.path.join(config.DATA_DIR, "adult.data")
    test_path = os.path.join(config.DATA_DIR, "adult.test")

    if not os.path.exists(train_path):
        print("Downloading Adult training data...")
        urllib.request.urlretrieve(ADULT_DATA_URL, train_path)

    if not os.path.exists(test_path):
        print("Downloading Adult test data...")
        urllib.request.urlretrieve(ADULT_TEST_URL, test_path)

    return train_path, test_path


def load_adult_raw(filepath: str, is_test: bool = False) -> pd.DataFrame:
    """Load raw Adult CSV data into pandas DataFrame."""
    skiprows = 1 if is_test else 0
    df = pd.read_csv(
        filepath,
        names=ADULT_COLUMN_NAMES,
        sep=r",\s*",
        engine="python",
        skiprows=skiprows,
        na_values="?"
    )
    return df


def preprocess_adult(df: pd.DataFrame, fit_encoders: bool = True,
                     encoders: dict = None, scaler: StandardScaler = None,
                     return_sensitive: bool = False):
    """
    Preprocess the Adult dataset.

    Args:
        df: Raw DataFrame
        fit_encoders: Whether to fit new encoders
        encoders: Pre-fitted encoders
        scaler: Pre-fitted scaler
        return_sensitive: Whether to return sensitive attribute values

    Returns:
        features, labels, encoders, scaler, [sensitive_attrs]
    """
    df = df.copy()
    df = df.dropna()

    # Extract sensitive attributes before encoding (for fairness evaluation)
    sensitive_attrs = {}
    if return_sensitive:
        # Store original categorical values for sensitive attributes
        sensitive_attrs["sex"] = (df["sex"].str.strip() == "Male").astype(int).values
        sensitive_attrs["race"] = (df["race"].str.strip() == "White").astype(int).values

    # Convert income label to binary
    df["income"] = df["income"].str.replace(".", "", regex=False)
    labels = (df["income"].str.strip() == ">50K").astype(int).values

    if fit_encoders:
        encoders = {}
        scaler = StandardScaler()

    # One-hot encode categorical columns
    encoded_cats = []
    for col in ADULT_CATEGORICAL_COLUMNS:
        if fit_encoders:
            unique_vals = df[col].unique()
            encoders[col] = {val: i for i, val in enumerate(unique_vals)}

        n_categories = len(encoders[col])
        one_hot = np.zeros((len(df), n_categories))
        for i, val in enumerate(df[col].values):
            if val in encoders[col]:
                one_hot[i, encoders[col][val]] = 1
        encoded_cats.append(one_hot)

    # Scale numerical columns
    numerical_data = df[ADULT_NUMERICAL_COLUMNS].values
    if fit_encoders:
        numerical_scaled = scaler.fit_transform(numerical_data)
    else:
        numerical_scaled = scaler.transform(numerical_data)

    features = np.hstack([numerical_scaled] + encoded_cats)

    if return_sensitive:
        return features, labels, encoders, scaler, sensitive_attrs
    return features, labels, encoders, scaler


# =============================================================================
# Credit Default Dataset Functions
# =============================================================================
def download_credit_data():
    """Download Credit Default dataset from UCI ML Repository if not present."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    data_path = os.path.join(config.DATA_DIR, "credit_default.xls")

    if not os.path.exists(data_path):
        print("Downloading Credit Default data...")
        urllib.request.urlretrieve(CREDIT_DATA_URL, data_path)

    return data_path


def load_credit_raw(filepath: str) -> pd.DataFrame:
    """Load raw Credit Default Excel data into pandas DataFrame."""
    # Skip first row (header) and use second row as column names
    df = pd.read_excel(filepath, header=1)
    return df


def preprocess_credit(df: pd.DataFrame, fit_encoders: bool = True,
                      encoders: dict = None, scaler: StandardScaler = None):
    """
    Preprocess the Credit Default dataset.

    Args:
        df: Raw DataFrame
        fit_encoders: Whether to fit new encoders
        encoders: Pre-fitted encoders
        scaler: Pre-fitted scaler

    Returns:
        features, labels, encoders, scaler
    """
    df = df.copy()

    # Drop ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Handle missing values
    df = df.dropna()

    # Extract labels (default payment next month)
    label_col = "default payment next month"
    if label_col not in df.columns:
        label_col = "Y"  # Alternative column name in some versions
    labels = df[label_col].values.astype(int)
    df = df.drop(columns=[label_col])

    if fit_encoders:
        encoders = {}
        scaler = StandardScaler()

    # One-hot encode categorical columns
    encoded_cats = []
    for col in CREDIT_CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue

        if fit_encoders:
            unique_vals = df[col].unique()
            encoders[col] = {val: i for i, val in enumerate(unique_vals)}

        n_categories = len(encoders[col])
        one_hot = np.zeros((len(df), n_categories))
        for i, val in enumerate(df[col].values):
            if val in encoders[col]:
                one_hot[i, encoders[col][val]] = 1
        encoded_cats.append(one_hot)

    # Get numerical columns that exist in the dataframe
    numerical_cols = [c for c in CREDIT_NUMERICAL_COLUMNS if c in df.columns]
    numerical_data = df[numerical_cols].values

    if fit_encoders:
        numerical_scaled = scaler.fit_transform(numerical_data)
    else:
        numerical_scaled = scaler.transform(numerical_data)

    if encoded_cats:
        features = np.hstack([numerical_scaled] + encoded_cats)
    else:
        features = numerical_scaled

    return features, labels, encoders, scaler


# =============================================================================
# Unified Data Loading Interface
# =============================================================================
def download_data(dataset: str = "adult"):
    """Download dataset files if not present."""
    if dataset == "adult":
        return download_adult_data()
    elif dataset == "credit_default":
        return download_credit_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {SUPPORTED_DATASETS}")


def load_raw_data(filepath: str, dataset: str = "adult", is_test: bool = False) -> pd.DataFrame:
    """Load raw data into DataFrame."""
    if dataset == "adult":
        return load_adult_raw(filepath, is_test=is_test)
    elif dataset == "credit_default":
        return load_credit_raw(filepath)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def preprocess_data(df: pd.DataFrame, dataset: str = "adult",
                    fit_encoders: bool = True, encoders: dict = None,
                    scaler: StandardScaler = None, return_sensitive: bool = False):
    """Preprocess data based on dataset type."""
    if dataset == "adult":
        return preprocess_adult(df, fit_encoders, encoders, scaler, return_sensitive)
    elif dataset == "credit_default":
        result = preprocess_credit(df, fit_encoders, encoders, scaler)
        if return_sensitive:
            # Credit dataset doesn't have the same sensitive attributes
            return result + ({},)
        return result
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_data_loaders(dataset: str = "adult", batch_size: int = None,
                     val_split: float = 0.1, return_sensitive: bool = False):
    """
    Main function to get train/val/test DataLoaders.

    Args:
        dataset: Dataset name ("adult" or "credit_default")
        batch_size: Batch size (defaults to config.BATCH_SIZE)
        val_split: Fraction of training data to use for validation
        return_sensitive: Whether to include sensitive attributes

    Returns:
        train_loader, val_loader, test_loader, input_dim, [sensitive_info]
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    if dataset == "adult":
        # Adult has separate train/test files
        train_path, test_path = download_adult_data()
        train_df = load_adult_raw(train_path, is_test=False)
        test_df = load_adult_raw(test_path, is_test=True)

        if return_sensitive:
            train_features, train_labels, encoders, scaler, train_sensitive = \
                preprocess_adult(train_df, fit_encoders=True, return_sensitive=True)
            test_features, test_labels, _, _, test_sensitive = \
                preprocess_adult(test_df, fit_encoders=False, encoders=encoders,
                                scaler=scaler, return_sensitive=True)
        else:
            train_features, train_labels, encoders, scaler = \
                preprocess_adult(train_df, fit_encoders=True)
            test_features, test_labels, _, _ = \
                preprocess_adult(test_df, fit_encoders=False, encoders=encoders, scaler=scaler)
            train_sensitive, test_sensitive = {}, {}

    elif dataset == "credit_default":
        # Credit default uses single file with train/test split
        data_path = download_credit_data()
        df = load_credit_raw(data_path)
        features, labels, encoders, scaler = preprocess_credit(df, fit_encoders=True)

        # Split into train/test (80/20)
        n_samples = len(labels)
        n_test = int(n_samples * 0.2)
        n_train = n_samples - n_test

        np.random.seed(config.RANDOM_SEED)
        indices = np.random.permutation(n_samples)
        train_idx, test_idx = indices[:n_train], indices[n_train:]

        train_features = features[train_idx]
        train_labels = labels[train_idx]
        test_features = features[test_idx]
        test_labels = labels[test_idx]
        train_sensitive, test_sensitive = {}, {}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Dataset: {dataset}")
    print(f"Training samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Feature dimension: {train_features.shape[1]}")
    print(f"Positive class ratio (train): {train_labels.mean():.3f}")

    # Create datasets
    full_train_dataset = TabularDataset(train_features, train_labels, train_sensitive)
    test_dataset = TabularDataset(test_features, test_labels, test_sensitive)

    # Split training data into train/val
    n_val = int(len(full_train_dataset) * val_split)
    n_train = len(full_train_dataset) - n_val

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [n_train, n_val], generator=generator
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_features.shape[1]

    if return_sensitive:
        return train_loader, val_loader, test_loader, input_dim, {
            "train": train_sensitive,
            "test": test_sensitive,
            "full_train_dataset": full_train_dataset
        }

    return train_loader, val_loader, test_loader, input_dim


def get_member_nonmember_split(dataset: str = "adult", member_ratio: float = 0.5,
                               return_sensitive: bool = False):
    """
    Create a member/non-member split for membership inference attacks.

    Args:
        dataset: Dataset name
        member_ratio: Fraction of training data to use as members
        return_sensitive: Whether to return sensitive attributes

    Returns:
        member_dataset, nonmember_dataset, test_dataset, input_dim, [sensitive_info]
    """
    if dataset == "adult":
        train_path, test_path = download_adult_data()
        train_df = load_adult_raw(train_path, is_test=False)
        test_df = load_adult_raw(test_path, is_test=True)

        if return_sensitive:
            train_features, train_labels, encoders, scaler, train_sensitive = \
                preprocess_adult(train_df, fit_encoders=True, return_sensitive=True)
            test_features, test_labels, _, _, test_sensitive = \
                preprocess_adult(test_df, fit_encoders=False, encoders=encoders,
                                scaler=scaler, return_sensitive=True)
        else:
            train_features, train_labels, encoders, scaler = \
                preprocess_adult(train_df, fit_encoders=True)
            test_features, test_labels, _, _ = \
                preprocess_adult(test_df, fit_encoders=False, encoders=encoders, scaler=scaler)
            train_sensitive, test_sensitive = {}, {}

    elif dataset == "credit_default":
        data_path = download_credit_data()
        df = load_credit_raw(data_path)
        features, labels, encoders, scaler = preprocess_credit(df, fit_encoders=True)

        # Use 80% for member/nonmember pool, 20% for test
        n_samples = len(labels)
        n_test = int(n_samples * 0.2)

        np.random.seed(config.RANDOM_SEED)
        indices = np.random.permutation(n_samples)
        pool_idx, test_idx = indices[:-n_test], indices[-n_test:]

        train_features = features[pool_idx]
        train_labels = labels[pool_idx]
        test_features = features[test_idx]
        test_labels = labels[test_idx]
        train_sensitive, test_sensitive = {}, {}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Create full training dataset
    full_dataset = TabularDataset(train_features, train_labels, train_sensitive)
    test_dataset = TabularDataset(test_features, test_labels, test_sensitive)

    # Split into members and non-members
    n_members = int(len(full_dataset) * member_ratio)
    n_nonmembers = len(full_dataset) - n_members

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    member_dataset, nonmember_dataset = random_split(
        full_dataset, [n_members, n_nonmembers], generator=generator
    )

    input_dim = train_features.shape[1]

    if return_sensitive:
        return member_dataset, nonmember_dataset, test_dataset, input_dim, {
            "train": train_sensitive,
            "test": test_sensitive,
            "full_dataset": full_dataset
        }

    return member_dataset, nonmember_dataset, test_dataset, input_dim


def get_dataset_with_sensitive_attrs(dataset: str = "adult"):
    """
    Get full dataset with sensitive attributes for fairness evaluation.

    Returns:
        train_features, train_labels, test_features, test_labels,
        train_sensitive, test_sensitive, input_dim
    """
    if dataset != "adult":
        raise ValueError("Sensitive attribute analysis only supported for Adult dataset")

    train_path, test_path = download_adult_data()
    train_df = load_adult_raw(train_path, is_test=False)
    test_df = load_adult_raw(test_path, is_test=True)

    train_features, train_labels, encoders, scaler, train_sensitive = \
        preprocess_adult(train_df, fit_encoders=True, return_sensitive=True)
    test_features, test_labels, _, _, test_sensitive = \
        preprocess_adult(test_df, fit_encoders=False, encoders=encoders,
                        scaler=scaler, return_sensitive=True)

    input_dim = train_features.shape[1]

    return (train_features, train_labels, test_features, test_labels,
            train_sensitive, test_sensitive, input_dim)


if __name__ == "__main__":
    # Test data loading for all datasets
    for ds in SUPPORTED_DATASETS:
        print(f"\n{'='*50}")
        print(f"Testing {ds} dataset loading...")
        print('='*50)

        try:
            train_loader, val_loader, test_loader, input_dim = get_data_loaders(dataset=ds)

            # Check a batch
            for features, labels in train_loader:
                print(f"\nBatch shape: {features.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Label distribution: {labels.mean():.3f} positive")
                break

            print(f"Input dimension: {input_dim}")
            print(f"{ds} data loading test complete!")
        except Exception as e:
            print(f"Error loading {ds}: {e}")
