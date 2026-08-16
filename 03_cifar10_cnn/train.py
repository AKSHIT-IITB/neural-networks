"""
Part C -- CNN on CIFAR-10 (PyTorch).

6-conv-layer CNN with BatchNorm + Dropout, trained with Adam, weight decay,
random-crop + horizontal-flip augmentation, and cosine LR schedule.

Saves: curves, per-class accuracy, confusion matrix, sample predictions,
metrics.json, trained weights.
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import Cifar10CNN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")

EPOCHS = int(os.environ.get("EPOCHS", 15))
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 5e-4
SEED = 42

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


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
            preds = logits.argmax(1)
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

    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_set = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tfm)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tfm)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, num_workers=2)

    model = Cifar10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    print(f"Device: {device} | params: "
          f"{sum(p.numel() for p in model.parameters()):,} | epochs: {EPOCHS}")

    history = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
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
        scheduler.step()

        test_acc, test_loss, _, _ = evaluate(model, test_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": run_loss / run_total,
            "train_acc": run_correct / run_total,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })
        print(f"epoch {epoch:2d}/{EPOCHS} | "
              f"train loss {history[-1]['train_loss']:.4f} "
              f"acc {history[-1]['train_acc']*100:.2f}% | "
              f"test acc {test_acc*100:.2f}% | {time.time()-t0:.0f}s",
              flush=True)

    test_acc, test_loss, preds, labels = evaluate(model, test_loader, device)
    print(f"\nFINAL TEST accuracy: {test_acc*100:.2f}%")

    # ---- curves ----
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["test_loss"] for h in history], label="test")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(ep, [h["train_acc"] for h in history], label="train")
    axes[1].plot(ep, [h["test_acc"] for h in history], label="test")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.suptitle("CIFAR-10 CNN (6 conv + BN + dropout + augmentation, Adam)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "curves.png"), dpi=150)
    plt.close()

    # ---- confusion matrix + per-class accuracy ----
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    per_class = {CLASSES[i]: float(cm[i, i] / cm[i].sum()) for i in range(10)}

    fig, ax = plt.subplots(figsize=(7.5, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"CIFAR-10 test confusion matrix ({test_acc*100:.2f}%)")
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ---- sample predictions ----
    raw_test = datasets.CIFAR10(DATA_DIR, train=False, download=False,
                                transform=transforms.ToTensor())
    wrong_idx = np.where(preds != labels)[0][:8]
    right_idx = np.where(preds == labels)[0][:8]
    fig, axes = plt.subplots(2, 8, figsize=(14, 4.5))
    for ax, i in zip(axes[0], right_idx):
        ax.imshow(raw_test[i][0].permute(1, 2, 0))
        ax.set_title(f"{CLASSES[preds[i]]} ✓", fontsize=8, color="green")
        ax.axis("off")
    for ax, i in zip(axes[1], wrong_idx):
        ax.imshow(raw_test[i][0].permute(1, 2, 0))
        ax.set_title(f"{CLASSES[preds[i]]}\n(true {CLASSES[labels[i]]})",
                     fontsize=7, color="red")
        ax.axis("off")
    fig.suptitle("Top: correct -- Bottom: mistakes")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "sample_predictions.png"), dpi=150)
    plt.close()

    torch.save(model.state_dict(), os.path.join(RESULTS, "cifar10_cnn.pt"))
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": test_acc, "test_loss": test_loss,
                   "epochs": EPOCHS, "per_class_accuracy": per_class,
                   "history": history}, f, indent=2)
    print(f"Saved results -> {RESULTS}")
    print("Per-class accuracy:", json.dumps(per_class, indent=2))


if __name__ == "__main__":
    main()
