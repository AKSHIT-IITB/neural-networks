"""
Part A -- Linear & Logistic Regression from scratch (NumPy only).

Dataset: UCI Heart Disease (Cleveland processed), 303 patients, 13 features.
Target: binarized -- 0 = no heart disease, 1 = heart disease (original 1-4).

Two models, both trained with hand-written batch gradient descent:
  1. Linear regression  -- foundational exercise: predicts max heart rate
     (thalach) from the other features, MSE loss.
  2. Logistic regression -- the actual clinical task: binary classification
     of heart disease, sigmoid + binary cross-entropy loss.

No sklearn is used for training -- only for the train/test split and the
ROC-AUC computation in evaluation.

NOTE: educational ML exercise, not a clinical diagnostic system.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "processed.cleveland.data")
RESULTS = os.path.join(HERE, "results")

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]


# ----------------------------------------------------------------------------
# Data loading & preprocessing
# ----------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA, header=None, names=COLUMNS, na_values="?")
    n_missing = df.isna().sum().sum()
    df = df.dropna().reset_index(drop=True)          # 6 rows have missing ca/thal
    df["target"] = (df["target"] > 0).astype(int)    # binarize 1-4 -> 1
    print(f"Loaded {len(df)} patients ({n_missing} missing values dropped)")
    print(f"Class balance: {df['target'].value_counts().to_dict()} "
          f"(1 = heart disease)")
    return df


def standardize(X_train, X_test):
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0) + 1e-9
    return (X_train - mu) / sigma, (X_test - mu) / sigma


# ----------------------------------------------------------------------------
# Linear regression from scratch (foundational exercise)
# ----------------------------------------------------------------------------
def train_linear_regression(X, y, lr=0.01, epochs=2000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    history = []
    for _ in range(epochs):
        y_hat = X @ w + b
        err = y_hat - y
        loss = np.mean(err ** 2)
        history.append(loss)
        grad_w = (2 / n) * X.T @ err
        grad_b = (2 / n) * err.sum()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, history


# ----------------------------------------------------------------------------
# Logistic regression from scratch (the actual task)
# ----------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_logistic_regression(X, y, lr=0.1, epochs=3000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    history = []
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        history.append(loss)
        err = p - y                      # dL/dz for BCE + sigmoid
        grad_w = X.T @ err / n
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, history


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn,
                roc_auc=float(roc_auc_score(y_true, y_prob)))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    df = load_data()

    # ---- Linear regression exercise: predict thalach (max heart rate) ----
    lin_features = [c for c in COLUMNS if c not in ("thalach", "target")]
    Xl = df[lin_features].values.astype(float)
    yl = df["thalach"].values.astype(float)
    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(Xl, yl, test_size=0.2, random_state=42)
    Xl_tr, Xl_te = standardize(Xl_tr, Xl_te)
    yl_mu, yl_sd = yl_tr.mean(), yl_tr.std()
    w, b, lin_hist = train_linear_regression(Xl_tr, (yl_tr - yl_mu) / yl_sd)
    pred_te = (Xl_te @ w + b) * yl_sd + yl_mu
    lin_rmse = float(np.sqrt(np.mean((pred_te - yl_te) ** 2)))
    ss_res = np.sum((yl_te - pred_te) ** 2)
    ss_tot = np.sum((yl_te - yl_te.mean()) ** 2)
    lin_r2 = float(1 - ss_res / ss_tot)
    print(f"\n[Linear regression]  predict thalach | "
          f"test RMSE = {lin_rmse:.2f} bpm | R^2 = {lin_r2:.3f}")

    # ---- Logistic regression: heart disease classification ----
    features = [c for c in COLUMNS if c != "target"]
    X = df[features].values.astype(float)
    y = df["target"].values.astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_tr, X_te = standardize(X_tr, X_te)

    w, b, log_hist = train_logistic_regression(X_tr, y_tr)
    prob_te = sigmoid(X_te @ w + b)
    prob_tr = sigmoid(X_tr @ w + b)
    m_te = classification_metrics(y_te, prob_te)
    m_tr = classification_metrics(y_tr, prob_tr)

    print(f"\n[Logistic regression]  heart disease classification "
          f"({len(y_tr)} train / {len(y_te)} test)")
    print(f"  train accuracy : {m_tr['accuracy']*100:.1f}%")
    print(f"  test  accuracy : {m_te['accuracy']*100:.1f}%")
    print(f"  precision      : {m_te['precision']:.3f}")
    print(f"  recall         : {m_te['recall']:.3f}   <- key metric for medical FN")
    print(f"  F1 score       : {m_te['f1']:.3f}")
    print(f"  ROC-AUC        : {m_te['roc_auc']:.3f}")
    print(f"  confusion      : TP={m_te['tp']} TN={m_te['tn']} "
          f"FP={m_te['fp']} FN={m_te['fn']}")

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(lin_hist, color="#4C72B0")
    axes[0].set_title("Linear regression (MSE loss)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].plot(log_hist, color="#C44E52")
    axes[1].set_title("Logistic regression (BCE loss)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    fig.suptitle("Gradient descent convergence (implemented from scratch)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "loss_curves.png"), dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_te, prob_te)
    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, color="#C44E52", label=f"ROC (AUC = {m_te['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Heart disease classifier -- ROC curve (test set)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "roc_curve.png"), dpi=150)
    plt.close()

    metrics = {
        "n_patients": int(len(df)),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "linear_regression": {"target": "thalach", "test_rmse_bpm": lin_rmse,
                              "test_r2": lin_r2},
        "logistic_regression": {"train": m_tr, "test": m_te},
    }
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved results -> {RESULTS}")


if __name__ == "__main__":
    main()
