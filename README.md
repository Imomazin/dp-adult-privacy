# Privacy-Preserving Deep Learning on Adult Census Dataset

A minimal research codebase for experimenting with differential privacy in deep learning using the Adult Census Income dataset.

## Overview

This repository provides:
- **Baseline training**: Standard neural network training without privacy
- **DP-SGD training**: Differentially private training using Opacus
- **MIA evaluation**: Membership inference attack to measure privacy leakage

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Opacus 1.4+

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

The [Adult Census Income dataset](https://archive.ics.uci.edu/ml/datasets/adult) is automatically downloaded on first run. It contains demographic information to predict whether income exceeds $50K/year.

- ~48,842 samples
- 14 features (mix of categorical and numerical)
- Binary classification task

## Usage

### Step 1: Data Loading (Optional Test)

Test that data loading works:

```bash
cd src
python data_loader.py
```

### Step 2: Train Baseline Model

Train a standard (non-private) neural network:

```bash
cd src
python train_baseline.py --epochs 20 --batch-size 256 --lr 0.001
```

Options:
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--hidden-dim`: Hidden layer dimension (default: 128)
- `--seed`: Random seed (default: 42)
- `--save-dir`: Directory for checkpoints (default: checkpoints)

### Step 3: Train DP Model

Train with differential privacy (DP-SGD):

```bash
cd src
python train_dp.py --epochs 20 --epsilon 8.0 --delta 1e-5
```

Options:
- `--epsilon`: Target epsilon for (ε, δ)-DP (default: 8.0)
- `--delta`: Target delta for (ε, δ)-DP (default: 1e-5)
- `--max-grad-norm`: Gradient clipping norm (default: 1.0)
- Plus all baseline options

### Step 4: Evaluate Privacy with MIA

Run membership inference attacks on both models:

```bash
cd src
python mia_evaluation.py --baseline checkpoints/baseline_final.pt --dp checkpoints/dp_final.pt
```

Options:
- `--baseline`: Path to baseline model checkpoint
- `--dp`: Path to DP model checkpoint
- `--batch-size`: Batch size for evaluation (default: 256)

## Expected Results

| Model    | Test Accuracy | MIA AUC |
|----------|---------------|---------|
| Baseline | ~85%          | ~0.55-0.60 |
| DP (ε=8) | ~80-83%       | ~0.50-0.52 |

Lower MIA AUC (closer to 0.5) indicates better privacy protection.

## Project Structure

```
├── README.md
├── requirements.txt
├── data/                  # Downloaded dataset (auto-created)
├── checkpoints/           # Saved models (auto-created)
└── src/
    ├── data_loader.py     # Data loading and preprocessing
    ├── model.py           # Neural network architecture
    ├── train_baseline.py  # Baseline training script
    ├── train_dp.py        # DP-SGD training script
    └── mia_evaluation.py  # Membership inference attack
```

## References

- [Opacus](https://opacus.ai/) - PyTorch library for differential privacy
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) - Abadi et al.
- [Membership Inference Attacks](https://arxiv.org/abs/1610.05820) - Shokri et al.

## License

Research use only.