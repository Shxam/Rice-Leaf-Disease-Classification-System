"""
evaluate.py — Comprehensive Model Evaluation
==============================================
Confusion matrices, per-class metrics, lab/field analysis,
ROC curves, PR curves, classification report, environment
performance breakdown, and imbalance-aware reporting.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from PIL import Image
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, roc_curve, auc, precision_recall_curve,
    average_precision_score, cohen_kappa_score, matthews_corrcoef,
    balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DATA_DIR = Path("data")
MODEL_PATH = Path("outputs/training/best_model.pth")
OUTPUT_DIR = Path("outputs/evaluation")
CLASSES = ["Bacterial Blight", "Brown Spot", "Healthy", "Leaf Blast", "Leaf Smut", "Tungro"]
NUM_CLASSES = len(CLASSES)
ENVS = ["Lab", "Field"]
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_NAME = "vit_base_patch16_224"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200,
    "font.family": "sans-serif", "font.size": 11,
    "axes.facecolor": "#fafafa", "figure.facecolor": "white"
})


class RiceDiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        self.samples, self.targets = [], []
        for ci, cn in enumerate(CLASSES):
            for env in ENVS:
                d = DATA_DIR / split / cn / env
                if not d.exists():
                    continue
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((d / f, ci, env))
                        self.targets.append(ci)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, l, e = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, l, e


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    preds, labels, envs, probs = [], [], [], []
    for imgs, labs, es in loader:
        imgs = imgs.to(device)
        with torch.amp.autocast("cuda"):
            out = model(imgs)
        probs.append(F.softmax(out, dim=1).cpu().numpy())
        preds.extend(out.argmax(1).cpu().numpy())
        labels.extend(labs.numpy())
        envs.extend(es)
    return np.array(labels), np.array(preds), envs, np.concatenate(probs)


def plot_confusion_matrices(yt, yp):
    """Raw + Normalized confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    short_classes = [c.replace("Bacterial ", "Bact. ") for c in CLASSES]

    cm = confusion_matrix(yt, yp)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=short_classes, yticklabels=CLASSES,
                linewidths=.5, linecolor="white")
    axes[0].set_title("Confusion Matrix (Raw Counts)", fontweight="bold", fontsize=13)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1],
                xticklabels=short_classes, yticklabels=CLASSES,
                linewidths=.5, linecolor="white", vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Recall / Row-Normalized)", fontweight="bold", fontsize=13)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ confusion_matrix.png")


def plot_per_class_metrics(yt, yp):
    """Per-class Precision, Recall, F1 bar chart with support annotations."""
    rep = classification_report(yt, yp, target_names=CLASSES, output_dict=True, zero_division=0)
    p = [rep[c]["precision"] for c in CLASSES]
    r = [rep[c]["recall"] for c in CLASSES]
    f = [rep[c]["f1-score"] for c in CLASSES]
    sup = [rep[c]["support"] for c in CLASSES]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(CLASSES))
    w = 0.22

    b1 = ax.bar(x - w, p, w, label="Precision", color="#3498db", edgecolor="white")
    b2 = ax.bar(x, r, w, label="Recall", color="#2ecc71", edgecolor="white")
    b3 = ax.bar(x + w, f, w, label="F1-Score", color="#e74c3c", edgecolor="white")

    for bars in [b1, b2, b3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    for i, s in enumerate(sup):
        ax.text(i, -0.08, f"n={s}", ha="center", fontsize=9, color="gray")
        # Highlight minority class
        if s < 50:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.1, color='red')

    ax.set_xlabel("Disease Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics (Test Set)\nHighlighted: minority classes with <50 test samples",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.set_ylim(-0.12, 1.2)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_class_metrics.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ per_class_metrics.png")
    return rep


def plot_lab_vs_field(yt, yp, envs):
    """Lab vs Field performance comparison — overall + per-class heatmap."""
    envs = np.array(envs)
    lm, fm = envs == "Lab", envs == "Field"

    la = accuracy_score(yt[lm], yp[lm]) if lm.sum() > 0 else 0
    fa = accuracy_score(yt[fm], yp[fm]) if fm.sum() > 0 else 0
    lf = f1_score(yt[lm], yp[lm], average="macro", zero_division=0) if lm.sum() > 0 else 0
    ff = f1_score(yt[fm], yp[fm], average="macro", zero_division=0) if fm.sum() > 0 else 0

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Overall bar chart
    ax = axes[0]
    x = np.arange(2)
    w = 0.3
    b1 = ax.bar(x - w/2, [la, lf], w, label=f"Lab (n={lm.sum()})", color="#3498db", edgecolor="white")
    b2 = ax.bar(x + w/2, [fa, ff], w, label=f"Field (n={fm.sum()})", color="#e67e22", edgecolor="white")
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Lab vs Field — Overall Performance", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Macro F1"])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=.3)

    # Per-class accuracy heatmap
    ax = axes[1]
    data_acc = np.zeros((NUM_CLASSES, 2))
    data_counts = np.zeros((NUM_CLASSES, 2), dtype=int)
    for ci in range(NUM_CLASSES):
        for ei, env in enumerate(ENVS):
            mask = (yt == ci) & (envs == env)
            data_counts[ci, ei] = mask.sum()
            if mask.sum() > 0:
                data_acc[ci, ei] = accuracy_score(yt[mask], yp[mask])
    df_acc = pd.DataFrame(data_acc, index=CLASSES, columns=ENVS)
    annot = np.array([[f"{data_acc[i,j]:.2f}\n(n={data_counts[i,j]})"
                       for j in range(2)] for i in range(NUM_CLASSES)])
    sns.heatmap(df_acc, annot=annot, fmt="", cmap="YlGnBu", ax=ax,
                linewidths=.5, linecolor="white", vmin=0, vmax=1)
    ax.set_title("Per-Class Accuracy by Environment", fontweight="bold")

    # Per-class F1 heatmap
    ax = axes[2]
    data_f1 = np.zeros((NUM_CLASSES, 2))
    for ci in range(NUM_CLASSES):
        for ei, env in enumerate(ENVS):
            mask = (yt == ci) & (envs == env)
            if mask.sum() > 0:
                # Binary F1 for this class in this environment
                binary_true = (yt[envs == env] == ci).astype(int)
                binary_pred = (yp[envs == env] == ci).astype(int)
                data_f1[ci, ei] = f1_score(binary_true, binary_pred, zero_division=0)
    df_f1 = pd.DataFrame(data_f1, index=CLASSES, columns=ENVS)
    sns.heatmap(df_f1, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
                linewidths=.5, linecolor="white", vmin=0, vmax=1)
    ax.set_title("Per-Class F1 by Environment", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "lab_vs_field_accuracy.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ lab_vs_field_accuracy.png")

    return {
        "overall": {"lab_acc": la, "field_acc": fa, "lab_f1": lf, "field_f1": ff,
                     "lab_n": int(lm.sum()), "field_n": int(fm.sum())},
        "per_class_acc": df_acc.to_dict(),
        "per_class_f1": df_f1.to_dict(),
    }


def plot_roc_curves(yt, yp):
    """One-vs-Rest ROC curves for each disease class."""
    yb = label_binarize(yt, classes=list(range(NUM_CLASSES)))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(10, 8))
    aucs = {}
    for i, (c, col) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(yb[:, i], yp[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[c] = roc_auc
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{c} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ roc_curves.png")
    return aucs


def plot_pr_curves(yt, yp):
    """Precision-Recall curves — especially important for Leaf Smut."""
    yb = label_binarize(yt, classes=list(range(NUM_CLASSES)))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(10, 8))
    aps = {}
    for i, (c, col) in enumerate(zip(CLASSES, colors)):
        pr, rc, _ = precision_recall_curve(yb[:, i], yp[:, i])
        ap = average_precision_score(yb[:, i], yp[:, i])
        aps[c] = ap
        ax.plot(rc, pr, color=col, lw=2, label=f"{c} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves\n(Critical for imbalanced classes like Leaf Smut)",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=.3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "precision_recall_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ precision_recall_curves.png")
    return aps


def plot_confidence_distribution(yt, yp_labels, probs):
    """Distribution of prediction confidence for correct vs incorrect predictions."""
    max_probs = probs.max(axis=1)
    correct = yt == yp_labels
    incorrect = ~correct

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Overall
    ax = axes[0]
    ax.hist(max_probs[correct], bins=30, alpha=0.7, label=f"Correct (n={correct.sum()})",
            color="#2ecc71", edgecolor="white")
    ax.hist(max_probs[incorrect], bins=30, alpha=0.7, label=f"Incorrect (n={incorrect.sum()})",
            color="#e74c3c", edgecolor="white")
    ax.set_xlabel("Prediction Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-class
    ax = axes[1]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    class_confidences = []
    for i, cls in enumerate(CLASSES):
        mask = yt == i
        class_confidences.append(max_probs[mask])

    bp = ax.boxplot(class_confidences, labels=CLASSES, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("Disease Class")
    ax.set_ylabel("Prediction Confidence")
    ax.set_title("Confidence by Class\n(Low confidence = model uncertainty)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "confidence_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ confidence_distribution.png")


def plot_env_confusion_matrices(yt, yp, envs):
    """Separate confusion matrices for Lab and Field environments."""
    envs = np.array(envs)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    short_classes = [c.replace("Bacterial ", "B. ") for c in CLASSES]

    for ax_idx, env in enumerate(ENVS):
        mask = envs == env
        if mask.sum() == 0:
            continue
        cm = confusion_matrix(yt[mask], yp[mask], labels=range(NUM_CLASSES))
        cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="YlOrRd" if env == "Field" else "Blues",
                    ax=axes[ax_idx], xticklabels=short_classes, yticklabels=CLASSES,
                    linewidths=.5, linecolor="white", vmin=0, vmax=1)
        axes[ax_idx].set_title(f"{env} Environment (n={mask.sum()})", fontweight="bold", fontsize=13)
        axes[ax_idx].set_xlabel("Predicted")
        axes[ax_idx].set_ylabel("True")
        axes[ax_idx].set_xticklabels(axes[ax_idx].get_xticklabels(), rotation=30, ha="right", fontsize=9)

    plt.suptitle("Confusion Matrices by Environment", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "env_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ env_confusion_matrices.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(" Rice Disease — Comprehensive Evaluation")
    print("=" * 70)

    # ── Load Model ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading model...")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"  Loaded: epoch {ckpt.get('epoch', '?')}, val F1={ckpt.get('best_f1', '?')}")
    if 'imbalance_ratio' in ckpt:
        print(f"  Training imbalance ratio: {ckpt['imbalance_ratio']:.1f}:1")

    # ── Load Test Data ──────────────────────────────────────────────────────
    print("\n[2/7] Loading test data...")
    ds = RiceDiseaseDataset("Test", transform=get_eval_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"  Test: {len(ds)} samples")

    test_counts = np.bincount(ds.targets, minlength=NUM_CLASSES)
    print("  Per-class counts:")
    for i, cls in enumerate(CLASSES):
        lab_n = sum(1 for s in ds.samples if s[1] == i and s[2] == "Lab")
        field_n = sum(1 for s in ds.samples if s[1] == i and s[2] == "Field")
        print(f"    {cls:<20}: {test_counts[i]:>4} (Lab={lab_n}, Field={field_n})")

    # ── Inference ───────────────────────────────────────────────────────────
    print("\n[3/7] Running inference...")
    yt, yp, envs, probs = run_inference(model, loader, device)

    # Overall metrics
    acc = accuracy_score(yt, yp)
    f1m = f1_score(yt, yp, average="macro", zero_division=0)
    f1w = f1_score(yt, yp, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(yt, yp)
    kappa = cohen_kappa_score(yt, yp)
    mcc = matthews_corrcoef(yt, yp)

    print(f"\n  ┌─── Overall Metrics ───────────────────────┐")
    print(f"  │ Accuracy:           {acc:.4f}              │")
    print(f"  │ Balanced Accuracy:  {bal_acc:.4f}              │")
    print(f"  │ F1 (macro):         {f1m:.4f}              │")
    print(f"  │ F1 (weighted):      {f1w:.4f}              │")
    print(f"  │ Cohen's Kappa:      {kappa:.4f}              │")
    print(f"  │ MCC:                {mcc:.4f}              │")
    print(f"  └─────────────────────────────────────────────┘")

    print("\n" + classification_report(yt, yp, target_names=CLASSES, zero_division=0))
    rep = classification_report(yt, yp, target_names=CLASSES, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(OUTPUT_DIR / "classification_report.csv")
    print("  ✓ classification_report.csv")

    # ── Plots ───────────────────────────────────────────────────────────────
    print("\n[4/7] Generating confusion matrices...")
    plot_confusion_matrices(yt, yp)
    plot_env_confusion_matrices(yt, yp, envs)

    print("\n[5/7] Generating per-class metrics & curves...")
    plot_per_class_metrics(yt, yp)
    aucs = plot_roc_curves(yt, probs)
    aps = plot_pr_curves(yt, probs)

    print("\n[6/7] Analyzing environments & confidence...")
    env_res = plot_lab_vs_field(yt, yp, envs)
    plot_confidence_distribution(yt, yp, probs)

    # ── Save Results JSON ───────────────────────────────────────────────────
    print("\n[7/7] Saving results JSON...")
    results = {
        "overall": {
            "accuracy": acc, "balanced_accuracy": bal_acc,
            "f1_macro": f1m, "f1_weighted": f1w,
            "cohen_kappa": kappa, "mcc": mcc,
            "n_test": len(ds),
        },
        "per_class": {
            c: {
                "precision": rep[c]["precision"],
                "recall": rep[c]["recall"],
                "f1": rep[c]["f1-score"],
                "support": int(rep[c]["support"]),
                "roc_auc": aucs.get(c, 0),
                "avg_precision": aps.get(c, 0),
            } for c in CLASSES
        },
        "environment": env_res,
    }
    with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  ✓ evaluation_results.json")

    print(f"\n{'='*70}")
    print(f" ✅ Evaluation complete! → {OUTPUT_DIR}/")
    print(f"    Outputs: confusion_matrix.png, env_confusion_matrices.png,")
    print(f"             per_class_metrics.png, roc_curves.png,")
    print(f"             precision_recall_curves.png, lab_vs_field_accuracy.png,")
    print(f"             confidence_distribution.png, classification_report.csv,")
    print(f"             evaluation_results.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
