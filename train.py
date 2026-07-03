"""
train.py — Vision Transformer Fine-Tuning for Rice Disease Classification
==========================================================================
Handles severe class imbalance (Leaf Smut: 130 vs Healthy: 1214) via:
  1. Inverse-frequency weighted CrossEntropyLoss with label smoothing
  2. WeightedRandomSampler for balanced batches
  3. Aggressive + environment-aware data augmentation
  4. Mixed precision training (AMP)
  5. Cosine annealing with warmup
  6. Early stopping on validation macro-F1
  7. Focal Loss option for hard-sample mining
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from PIL import Image
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/training")
CLASSES = ["Bacterial Blight", "Brown Spot", "Healthy", "Leaf Blast", "Leaf Smut", "Tungro"]
NUM_CLASSES = len(CLASSES)
ENVS = ["Lab", "Field"]
IMG_SIZE = 224
BATCH_SIZE = 32
ACCUM_STEPS = 2          # gradient accumulation → effective batch = 64
NUM_EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 3
PATIENCE = 12
SEED = 42
MODEL_NAME = "vit_base_patch16_224"
LABEL_SMOOTHING = 0.1
USE_FOCAL_LOSS = True     # focal loss for hard-sample mining (Bacterial Blight field etc.)
FOCAL_GAMMA = 2.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Focal Loss ──────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance — down-weights easy examples."""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ── Dataset ─────────────────────────────────────────────────────────────────
class RiceDiseaseDataset(Dataset):
    """Dataset that loads from data/{split}/{class}/{Lab|Field}/*.jpg"""

    def __init__(self, split, transform=None):
        self.transform = transform
        self.samples = []   # (path, class_idx, env)
        self.targets = []   # class indices for sampler
        self.envs = []      # environment labels

        for cls_idx, cls_name in enumerate(CLASSES):
            for env in ENVS:
                d = DATA_DIR / split / cls_name / env
                if not d.exists():
                    continue
                for fname in sorted(os.listdir(d)):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((d / fname, cls_idx, env))
                        self.targets.append(cls_idx)
                        self.envs.append(env)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, env = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, env


# ── Transforms ──────────────────────────────────────────────────────────────
def get_transforms(split):
    if split == "Train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ── Model ───────────────────────────────────────────────────────────────────
def create_model(num_classes, pretrained=True):
    model = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)
    return model


# ── Training Utilities ──────────────────────────────────────────────────────
def compute_class_weights(dataset):
    """Inverse frequency class weights — critical for Leaf Smut (130 vs 1214)."""
    counts = np.bincount(dataset.targets, minlength=NUM_CLASSES).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES  # normalize so mean = 1
    return torch.FloatTensor(weights)


def create_weighted_sampler(dataset):
    """WeightedRandomSampler for balanced batches — ensures Leaf Smut is sampled fairly."""
    counts = np.bincount(dataset.targets, minlength=NUM_CLASSES).astype(float)
    class_weight = 1.0 / (counts + 1e-6)
    sample_weights = [class_weight[t] for t in dataset.targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    return sampler


def get_lr_lambda(epoch, warmup_epochs, total_epochs):
    """Warmup + cosine annealing schedule."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return 0.5 * (1 + np.cos(np.pi * progress))


# ── Training Loop ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps=1):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    all_envs = []
    optimizer.zero_grad()

    for batch_idx, (images, labels, envs_batch) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_envs.extend(envs_batch)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1, per_class_f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    all_envs = []

    for images, labels, envs_batch in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_envs.extend(envs_batch)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    # Environment-level metrics
    all_envs = np.array(all_envs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    env_metrics = {}
    for env in ENVS:
        mask = all_envs == env
        if mask.sum() > 0:
            env_metrics[env] = {
                "acc": accuracy_score(all_labels[mask], all_preds[mask]),
                "f1": f1_score(all_labels[mask], all_preds[mask], average="macro", zero_division=0),
                "n": int(mask.sum()),
            }

    return epoch_loss, epoch_acc, epoch_f1, per_class_f1, env_metrics


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_training_curves(history):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "o-", color="#3498db", label="Train", markersize=3)
    ax.plot(epochs, history["val_loss"], "o-", color="#e74c3c", label="Validation", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], "o-", color="#3498db", label="Train", markersize=3)
    ax.plot(epochs, history["val_acc"], "o-", color="#e74c3c", label="Validation", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1
    ax = axes[1, 0]
    ax.plot(epochs, history["train_f1"], "o-", color="#3498db", label="Train", markersize=3)
    ax.plot(epochs, history["val_f1"], "o-", color="#e74c3c", label="Validation", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1-Score")
    ax.set_title("F1-Score Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    best_epoch = np.argmax(history["val_f1"]) + 1
    best_f1 = max(history["val_f1"])
    axes[1, 0].axvline(best_epoch, color="#2ecc71", linestyle="--", alpha=0.7)
    axes[1, 0].annotate(f"Best: {best_f1:.4f}\n(epoch {best_epoch})",
                         xy=(best_epoch, best_f1), fontsize=9,
                         bbox=dict(boxstyle="round", facecolor="#d5f5e3"))

    # Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, history["lr"], "o-", color="#9b59b6", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves — Rice Disease ViT", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1_history(history):
    """Plot per-class F1 across epochs — crucial to monitor Leaf Smut convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    epochs = range(1, len(history["val_per_class_f1"]) + 1)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for ax_idx, (title, key) in enumerate([
        ("Train Per-Class F1", "train_per_class_f1"),
        ("Validation Per-Class F1", "val_per_class_f1")
    ]):
        ax = axes[ax_idx]
        for i, cls in enumerate(CLASSES):
            vals = [ep_f1[i] for ep_f1 in history[key]]
            ax.plot(epochs, vals, "o-", color=colors[i], label=cls, markersize=3, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1-Score")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle("Per-Class F1 Progression (Monitoring Minority Classes)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_class_f1_history.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_class_weight_chart(class_weights, train_counts):
    """Visualize class weights vs sample counts."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(CLASSES))
    color1 = "#3498db"
    color2 = "#e74c3c"

    bars = ax1.bar(x - 0.2, train_counts, 0.4, label="Sample Count", color=color1,
                   edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Disease Class")
    ax1.set_ylabel("Sample Count", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, class_weights, 0.4, label="Class Weight", color=color2,
            edgecolor="white", alpha=0.8)
    ax2.set_ylabel("Class Weight", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax1.set_title("Sample Counts vs Class Weights\n(Inverse frequency weighting for imbalance correction)",
                  fontweight="bold")

    # Add text annotations
    for i, (cnt, wt) in enumerate(zip(train_counts, class_weights)):
        ax1.text(i - 0.2, cnt + 20, str(cnt), ha="center", fontsize=8, fontweight="bold", color=color1)
        ax2.text(i + 0.2, wt + 0.1, f"{wt:.2f}", ha="center", fontsize=8, fontweight="bold", color=color2)

    fig.legend(loc="upper right", bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_weights_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(" Rice Disease ViT Training Pipeline")
    print("=" * 70)
    print(f"  Device:     {device}")
    if device.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:       {mem:.1f} GB")
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Image size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUM_STEPS} w/ grad accum)")
    print(f"  Epochs:     {NUM_EPOCHS} (patience={PATIENCE})")
    print(f"  LR:         {LR}")
    print(f"  Label smooth: {LABEL_SMOOTHING}")
    print(f"  Loss fn:    {'FocalLoss' if USE_FOCAL_LOSS else 'WeightedCE'}")
    print()

    # ── Data ────────────────────────────────────────────────────────────────
    print("[1/5] Loading datasets...")
    train_dataset = RiceDiseaseDataset("Train", transform=get_transforms("Train"))
    val_dataset = RiceDiseaseDataset("Valid", transform=get_transforms("Valid"))

    print(f"  Train: {len(train_dataset)} images")
    print(f"  Valid: {len(val_dataset)} images")

    # Class distribution
    train_counts = np.bincount(train_dataset.targets, minlength=NUM_CLASSES)
    val_counts = np.bincount(val_dataset.targets, minlength=NUM_CLASSES)

    print("\n  Class distribution:")
    print(f"  {'Class':<20} {'Train':>7} {'Valid':>7} {'Lab%':>7} {'Field%':>7}")
    print("  " + "-" * 55)
    for i, cls in enumerate(CLASSES):
        total_train = train_counts[i]
        total_val = val_counts[i]
        lab_count = sum(1 for s in train_dataset.samples if s[1] == i and s[2] == "Lab")
        field_count = sum(1 for s in train_dataset.samples if s[1] == i and s[2] == "Field")
        lab_pct = lab_count / total_train * 100 if total_train > 0 else 0
        field_pct = field_count / total_train * 100 if total_train > 0 else 0
        marker = " ⚠️ MINORITY" if total_train < 200 else ""
        print(f"  {cls:<20} {total_train:>7} {total_val:>7} {lab_pct:>6.1f}% {field_pct:>6.1f}%{marker}")

    imbalance_ratio = train_counts.max() / max(train_counts.min(), 1)
    print(f"\n  ⚠️  Imbalance ratio: {imbalance_ratio:.1f}:1 "
          f"(max={CLASSES[train_counts.argmax()]}={train_counts.max()}, "
          f"min={CLASSES[train_counts.argmin()]}={train_counts.min()})")

    # Class weights
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"\n  Class weights: {dict(zip(CLASSES, [f'{w:.3f}' for w in class_weights.cpu().numpy()]))}")

    # Plot class weight chart
    plot_class_weight_chart(class_weights.cpu().numpy(), train_counts)
    print("  ✓ class_weights_chart.png")

    # Weighted sampler
    sampler = create_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ── Model ───────────────────────────────────────────────────────────────
    print("\n[2/5] Building model...")
    model = create_model(NUM_CLASSES, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # ── Training Setup ──────────────────────────────────────────────────────
    print("\n[3/5] Setting up training...")

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA,
                              label_smoothing=LABEL_SMOOTHING)
        print(f"  Loss: FocalLoss(gamma={FOCAL_GAMMA}, smoothing={LABEL_SMOOTHING})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
        print(f"  Loss: WeightedCE(smoothing={LABEL_SMOOTHING})")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: get_lr_lambda(epoch, WARMUP_EPOCHS, NUM_EPOCHS)
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Training Loop ───────────────────────────────────────────────────────
    print("\n[4/5] Training...")
    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [], "lr": [],
        "train_per_class_f1": [], "val_per_class_f1": [],
    }
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        train_loss, train_acc, train_f1, train_pcf1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, ACCUM_STEPS
        )
        val_loss, val_acc, val_f1, val_pcf1, val_env = validate(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)
        history["train_per_class_f1"].append(train_pcf1.tolist())
        history["val_per_class_f1"].append(val_pcf1.tolist())

        # Print epoch summary
        print(f"  Epoch {epoch+1:>3}/{NUM_EPOCHS} │ "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} │ "
              f"Acc: {train_acc:.4f}/{val_acc:.4f} │ "
              f"F1: {train_f1:.4f}/{val_f1:.4f} │ "
              f"LR: {current_lr:.2e} │ {elapsed:.1f}s", end="")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "class_names": CLASSES,
                "val_per_class_f1": val_pcf1.tolist(),
                "val_env_metrics": val_env,
                "imbalance_ratio": float(imbalance_ratio),
            }, OUTPUT_DIR / "best_model.pth")
            print(" ★ best", end="")
        else:
            patience_counter += 1

        print()

        # Print per-class F1 every 5 epochs for monitoring minority classes
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"         Per-class Val F1: ", end="")
            for i, cls in enumerate(CLASSES):
                marker = "⚠️" if val_pcf1[i] < 0.5 else "✓"
                print(f"{cls[:8]}={val_pcf1[i]:.3f}{marker} ", end="")
            print()
            if val_env:
                for env, m in val_env.items():
                    print(f"         {env}: acc={m['acc']:.4f} f1={m['f1']:.4f} (n={m['n']})")

        if patience_counter >= PATIENCE:
            print(f"\n  ⏹ Early stopping at epoch {epoch+1} (patience={PATIENCE})")
            break

    # ── Save Results ────────────────────────────────────────────────────────
    print("\n[5/5] Saving results...")
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("  ✓ training_history.json")

    plot_training_curves(history)
    print("  ✓ training_curves.png")

    plot_per_class_f1_history(history)
    print("  ✓ per_class_f1_history.png")

    # Save training config
    config = {
        "model": MODEL_NAME, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "accum_steps": ACCUM_STEPS, "effective_batch": BATCH_SIZE * ACCUM_STEPS,
        "epochs_trained": len(history["train_loss"]), "max_epochs": NUM_EPOCHS,
        "lr": LR, "weight_decay": WEIGHT_DECAY, "warmup_epochs": WARMUP_EPOCHS,
        "patience": PATIENCE, "seed": SEED, "label_smoothing": LABEL_SMOOTHING,
        "use_focal_loss": USE_FOCAL_LOSS, "focal_gamma": FOCAL_GAMMA,
        "best_val_f1": best_f1, "imbalance_ratio": float(imbalance_ratio),
        "class_weights": class_weights.cpu().numpy().tolist(),
        "train_counts": train_counts.tolist(),
        "classes": CLASSES,
    }
    with open(OUTPUT_DIR / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("  ✓ training_config.json")

    print(f"\n{'='*70}")
    print(f" ✅ Training complete! Best Val F1: {best_f1:.4f}")
    print(f"    Model saved to: {OUTPUT_DIR / 'best_model.pth'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
