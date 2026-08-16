# Intro to the Realm of Neural Networks

A three-stage deep learning project following the natural progression:

```
Classical ML (from scratch)  →  Neural Networks (PyTorch)  →  CNNs
Heart disease                    MNIST                         CIFAR-10
NumPy gradient descent           Backpropagation               Convolution
```

The point of the ordering: implement optimization and binary classification
by hand first, so that when PyTorch appears, `loss.backward()` and
`optimizer.step()` are recognizable as automation of code you already wrote
yourself.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Part A — Linear & Logistic Regression from scratch

**Data:** UCI Heart Disease (Cleveland), 297 patients after dropping 6 rows
with missing values, 13 clinical features, binary target.

**No ML library does the training** — gradient descent, sigmoid, MSE, and
binary cross-entropy are implemented directly in NumPy
([logistic_regression.py](01_logistic_regression/logistic_regression.py)).
sklearn is used only for the train/test split and ROC-AUC computation.

```bash
python3 01_logistic_regression/logistic_regression.py
```

- Linear regression (foundational exercise): predicts max heart rate from the
  other features — test RMSE **17.3 bpm**, R² 0.34
- Logistic regression (the real task): heart disease classification —
  test accuracy **83.3%**, precision 0.846, recall 0.786, F1 0.815,
  **ROC-AUC 0.949**

> Educational ML exercise, **not** a clinical diagnostic system.

## Part B — PyTorch MLP on MNIST

Architecture `784 → 128 → 64 → 10` (ReLU), Adam, 10 epochs, batch 64.

```bash
cd 02_mnist && python3 train.py
```

See `02_mnist/results/` for loss/accuracy curves, confusion matrix, and
sample correct/incorrect predictions. Final test accuracy is recorded in
`results/metrics.json` — see [RESULTS.md](RESULTS.md) for the actual number
from the last run.

## Part C — CNN on CIFAR-10

6 convolutional layers (3 blocks of Conv-BN-ReLU ×2 + MaxPool), dropout,
weight decay, random-crop + horizontal-flip augmentation, Adam with cosine
LR schedule, 15 epochs.

```bash
cd 03_cifar10_cnn && python3 train.py       # EPOCHS=n to override
```

See `03_cifar10_cnn/results/` for curves, per-class accuracy, confusion
matrix, and sample predictions.

## Results

All final numbers with figures: [RESULTS.md](RESULTS.md).

## Repo layout

```
neural-networks-project/
├── 01_logistic_regression/
│   ├── data/processed.cleveland.data     # UCI Cleveland heart disease
│   ├── logistic_regression.py            # NumPy-only training
│   └── results/
├── 02_mnist/
│   ├── model.py                          # 784-128-64-10 MLP
│   ├── train.py
│   └── results/
├── 03_cifar10_cnn/
│   ├── model.py                          # 6-conv CNN + BN + dropout
│   ├── train.py
│   └── results/
├── requirements.txt
├── README.md
└── RESULTS.md
```
