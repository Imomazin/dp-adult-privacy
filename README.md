# Privacy-Preserving Deep Learning Research

A minimal research codebase for studying differential privacy in machine learning using tabular datasets.

## Overview

This repository implements:
- **Baseline training**: Standard neural network training (non-private)
- **DP-SGD training**: Differentially private training using Opacus
- **Membership Inference Attack (MIA)**: Threshold-based and shadow-model attacks
- **Fairness evaluation**: Subgroup metrics and TPR gap analysis
- **Privacy frontier plotting**: Visualize privacy-utility tradeoffs

## Supported Datasets

| Dataset | Description | Samples | Task |
|---------|-------------|---------|------|
| `adult` | Adult Census Income (UCI) | ~48K | Income >$50K prediction |
| `credit_default` | Credit Card Default (UCI) | ~30K | Default payment prediction |

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
├── data.py             # Multi-dataset loading and preprocessing
├── model.py            # Neural network architecture
├── train_baseline.py   # Standard (non-private) training
├── train_dp.py         # Differentially private training (DP-SGD)
├── mia_attack.py       # Membership inference attack evaluation
├── fairness_eval.py    # Fairness metrics by subgroup
├── make_plots.py       # Privacy frontier visualization
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Usage

### 1. Data Loading

Data is automatically downloaded from UCI ML Repository on first run:

```bash
# Test data loading for all datasets
python data.py
```

### 2. Train Baseline Model

```bash
# Train on Adult dataset (default)
python train_baseline.py

# Train on Credit Default dataset
python train_baseline.py --dataset credit_default

# Custom parameters
python train_baseline.py --dataset adult --epochs 30 --lr 0.001 --save checkpoints/baseline.pt
```

### 3. Train with Differential Privacy

```bash
# Train with default epsilon=1.0
python train_dp.py --dataset adult

# Train with custom privacy budget
python train_dp.py --dataset adult --epsilon 0.5 --epochs 30

# Train on Credit Default
python train_dp.py --dataset credit_default --epsilon 1.0
```

### 4. Membership Inference Attack

```bash
# Threshold-based MIA on baseline model
python mia_attack.py --dataset adult

# Compare baseline vs DP model
python mia_attack.py --dataset adult --compare --epsilon 1.0

# Include shadow-model attack (default k=5)
python mia_attack.py --dataset adult --compare --shadow --n-shadows 5

# Run on Credit Default dataset
python mia_attack.py --dataset credit_default --compare
```

### 5. Fairness Evaluation

```bash
# Evaluate fairness on Adult dataset (sex and race subgroups)
python fairness_eval.py

# Custom epsilon values
python fairness_eval.py --epsilons 0.5 1.0 2.0 5.0 --epochs 20

# Results saved to results/fairness.csv
```

### 6. Generate Plots

```bash
# Create sample results for testing
python make_plots.py --create-sample

# Generate plots from results
python make_plots.py --results results/results.csv --output results/figures/
```

Generates:
- `privacy_utility_frontier.png/pdf`: Epsilon vs test AUC/accuracy
- `privacy_leakage_frontier.png/pdf`: Epsilon vs MIA AUC
- `combined_frontier.png/pdf`: Both plots side-by-side

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

## Metrics

### Model Performance
- **Accuracy**: Classification accuracy
- **AUC-ROC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve (for imbalanced data)

### Privacy (MIA)
- **Threshold attack**: Uses prediction confidence/loss as membership signal
- **Shadow-model attack**: Trains attack classifier on shadow model outputs
- **Attack AUC**: How well attacker distinguishes members from non-members

### Fairness
- **Subgroup AUC/Accuracy**: Performance per demographic group
- **TPR Gap**: Equal opportunity difference (|TPR_group1 - TPR_group0|)

## Key Concepts

### Differential Privacy (DP)

DP provides formal privacy guarantees by ensuring model outputs are similar whether or not any individual's data is included. Privacy budget (epsilon, delta) quantifies this:
- Lower epsilon = stronger privacy, potentially lower utility
- Delta should be smaller than 1/N where N is dataset size

### DP-SGD

Differentially Private Stochastic Gradient Descent achieves DP by:
1. Clipping per-sample gradients to bound sensitivity
2. Adding calibrated Gaussian noise to aggregated gradients

### Membership Inference Attack

MIA tests if an attacker can determine whether a sample was in the training set. Higher attack success indicates more privacy leakage. DP training typically reduces MIA vulnerability.

### Fairness and Privacy

DP can have disparate impact on model fairness across demographic groups. This codebase measures TPR gaps to quantify equal opportunity differences between groups at various privacy levels.

## Expected Results

| Model | Test Accuracy | Test AUC | MIA AUC |
|-------|--------------|----------|---------|
| Baseline | ~85% | ~0.90 | ~0.55-0.60 |
| DP (eps=1) | ~80-82% | ~0.85 | ~0.50-0.52 |
| DP (eps=0.5) | ~75-78% | ~0.80 | ~0.50-0.51 |

Note: DP models trade accuracy for privacy protection.

## References

- [Opacus Documentation](https://opacus.ai/)
- [Adult Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/adult)
- [Credit Default Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- Shokri et al., "Membership Inference Attacks Against Machine Learning Models" (2017)
- Abadi et al., "Deep Learning with Differential Privacy" (2016)
- Bagdasaryan et al., "Differential Privacy Has Disparate Impact on Model Accuracy" (2019)

## License

Research code for educational purposes.
