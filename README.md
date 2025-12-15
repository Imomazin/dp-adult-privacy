# Privacy-Preserving Deep Learning on Adult Census Dataset

A minimal research codebase for studying differential privacy in machine learning using the Adult Census Income dataset.

## Overview

This repository implements:
- **Baseline training**: Standard neural network training (non-private)
- **DP-SGD training**: Differentially private training using Opacus
- **Membership Inference Attack (MIA)**: Evaluate privacy leakage by testing if an attacker can determine training set membership

## Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
dp-adult-privacy/
├── config.py           # Configuration parameters
├── data.py             # Data loading and preprocessing
├── model.py            # Neural network architecture
├── train_baseline.py   # Standard (non-private) training
├── train_dp.py         # Differentially private training
├── mia_attack.py       # Membership inference attack evaluation
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Usage

### 1. Data Loading (Automatic)

Data is automatically downloaded from UCI ML Repository on first run:

```bash
# Test data loading
python data.py
```

### 2. Train Baseline Model (Non-Private)

```bash
# Train with default settings
python train_baseline.py

# Train with custom parameters
python train_baseline.py --epochs 30 --lr 0.001 --save checkpoints/my_baseline.pt
```

### 3. Train with Differential Privacy (DP-SGD)

```bash
# Train with default epsilon=1.0
python train_dp.py

# Train with custom privacy budget
python train_dp.py --epsilon 0.5 --epochs 30 --save checkpoints/dp_eps05.pt
```

### 4. Evaluate Membership Inference Attack

```bash
# Run MIA on a newly trained baseline model
python mia_attack.py

# Run MIA on a pre-trained model
python mia_attack.py --model checkpoints/baseline.pt

# Compare baseline vs DP model vulnerability
python mia_attack.py --compare --epsilon 1.0
```

## Configuration

Edit `config.py` to modify default parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 256 | Training batch size |
| `EPOCHS` | 20 | Number of training epochs |
| `LEARNING_RATE` | 0.01 | Optimizer learning rate |
| `EPSILON` | 1.0 | Target privacy budget |
| `DELTA` | 1e-5 | Privacy parameter delta |
| `MAX_GRAD_NORM` | 1.0 | Gradient clipping bound |

## Key Concepts

### Differential Privacy (DP)

DP provides formal privacy guarantees by ensuring that the output of a computation is nearly the same whether or not any individual's data is included. The privacy budget (epsilon, delta) quantifies this guarantee:
- Lower epsilon = stronger privacy, but potentially lower utility
- delta should be smaller than 1/N where N is the dataset size

### DP-SGD

DP-SGD (Differentially Private Stochastic Gradient Descent) achieves DP by:
1. Clipping per-sample gradients to bound sensitivity
2. Adding calibrated Gaussian noise to gradients

### Membership Inference Attack

MIA tests if an attacker can determine whether a specific sample was used to train a model. Higher attack success indicates more privacy leakage. DP training typically reduces MIA vulnerability.

## Expected Results

| Model | Test Accuracy | MIA Attack Accuracy |
|-------|--------------|---------------------|
| Baseline | ~85% | ~55-60% |
| DP (epsilon=1) | ~80-82% | ~50-52% |

Note: DP models trade some accuracy for privacy protection.

## References

- [Opacus Documentation](https://opacus.ai/)
- [Adult Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/adult)
- Shokri et al., "Membership Inference Attacks Against Machine Learning Models" (2017)
- Abadi et al., "Deep Learning with Differential Privacy" (2016)

## License

Research code for educational purposes.
