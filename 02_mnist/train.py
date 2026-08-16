"""
Part B -- PyTorch fully connected network on MNIST.

Architecture: 784 -> 128 -> 64 -> 10 (ReLU), CrossEntropyLoss, Adam.
Resume claim to back: >= 94% test accuracy.

Saves: loss/accuracy curves, confusion matrix, sample predictions
(correct + incorrect), metrics.json, trained weights.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MnistMLP

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")

EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
SEED = 42


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    return (correct / total, loss_sum / total,
            torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy())


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean/std
    ])
    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    train_set, val_set = random_split(
        full_train, [54000, 6000], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)
    test_loader = DataLoader(test_set, batch_size=256)

    model = MnistMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"Device: {device} | params: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            run_total += y.size(0)

        val_acc, val_loss, _, _ = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": run_loss / run_total,
            "train_acc": run_correct / run_total,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(f"epoch {epoch:2d}/{EPOCHS} | "
              f"train loss {history[-1]['train_loss']:.4f} "
              f"acc {history[-1]['train_acc']*100:.2f}% | "
              f"val loss {val_loss:.4f} acc {val_acc*100:.2f}%")

    test_acc, test_loss, preds, labels = evaluate(model, test_loader, device)
    print(f"\nTEST accuracy: {test_acc*100:.2f}%  (resume claim: >=94%)")

    # ---- curves ----
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(ep, [h["train_acc"] for h in history], label="train")
    axes[1].plot(ep, [h["val_acc"] for h in history], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.suptitle("MNIST MLP (784-128-64-10, Adam)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "curves.png"), dpi=150)
    plt.close()

    # ---- confusion matrix ----
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"MNIST test confusion matrix ({test_acc*100:.2f}%)")
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ---- sample predictions: 8 correct + 8 wrong ----
    raw_test = datasets.MNIST(DATA_DIR, train=False, download=False,
                              transform=transforms.ToTensor())
    wrong_idx = np.where(preds != labels)[0][:8]
    right_idx = np.where(preds == labels)[0][:8]
    fig, axes = plt.subplots(2, 8, figsize=(13, 4))
    for ax, i in zip(axes[0], right_idx):
        ax.imshow(raw_test[i][0].squeeze(), cmap="gray")
        ax.set_title(f"{preds[i]} ✓", fontsize=9, color="green")
        ax.axis("off")
    for ax, i in zip(axes[1], wrong_idx):
        ax.imshow(raw_test[i][0].squeeze(), cmap="gray")
        ax.set_title(f"pred {preds[i]} / true {labels[i]}", fontsize=8, color="red")
        ax.axis("off")
    fig.suptitle("Top: correct predictions -- Bottom: mistakes")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "sample_predictions.png"), dpi=150)
    plt.close()

    # top confusions
    cm_off = cm.copy(); np.fill_diagonal(cm_off, 0)
    top_conf = []
    flat = np.argsort(cm_off.ravel())[::-1][:5]
    for f in flat:
        i, j = divmod(int(f), 10)
        if cm_off[i, j] > 0:
            top_conf.append({"true": i, "pred": j, "count": int(cm_off[i, j])})

    torch.save(model.state_dict(), os.path.join(RESULTS, "mnist_mlp.pt"))
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": test_acc, "test_loss": test_loss,
                   "epochs": EPOCHS, "history": history,
                   "top_confusions": top_conf}, f, indent=2)
    print(f"Saved results -> {RESULTS}")
    print("Top confusions:", top_conf)


if __name__ == "__main__":
    main()
