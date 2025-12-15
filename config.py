"""
Configuration parameters for privacy-preserving ML experiments.
All hyperparameters in one place for reproducibility.
"""

# =============================================================================
# Data Configuration
# =============================================================================
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
DATA_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
DATA_DIR = "./data"
RANDOM_SEED = 42

# =============================================================================
# Model Configuration
# =============================================================================
# Input dimension determined by preprocessing (computed dynamically)
HIDDEN_DIM = 64
OUTPUT_DIM = 1  # Binary classification

# =============================================================================
# Training Configuration (Baseline)
# =============================================================================
BATCH_SIZE = 256
LEARNING_RATE = 0.01
EPOCHS = 20

# =============================================================================
# Differential Privacy Configuration
# =============================================================================
# Privacy budget
EPSILON = 1.0  # Target epsilon
DELTA = 1e-5   # Target delta (should be < 1/N where N is training set size)

# DP-SGD parameters
MAX_GRAD_NORM = 1.0  # Gradient clipping bound
NOISE_MULTIPLIER = 1.1  # Will be computed based on epsilon if None

# =============================================================================
# Membership Inference Attack Configuration
# =============================================================================
MIA_SHADOW_MODELS = 3  # Number of shadow models for attack
MIA_ATTACK_EPOCHS = 10  # Epochs to train attack model
