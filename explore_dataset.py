"""
explore_dataset.py — Comprehensive Dataset Analysis & Visualization
=====================================================================
Generates publication-ready plots analyzing class balance, lab/field
distribution, sample images, resolution statistics, and imbalance metrics.
"""

import os
import sys
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import defaultdict

# ── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/exploration")
SPLITS = ["Train", "Valid", "Test"]
CLASSES = ["Bacterial Blight", "Brown Spot", "Healthy", "Leaf Blast", "Leaf Smut", "Tungro"]
ENVS = ["Lab", "Field"]
SEED = 42

# ── Style Setup ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
})
PALETTE = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]

random.seed(SEED)
np.random.seed(SEED)


def collect_stats():
    """Collect image counts and file paths for all splits/classes/envs."""
    stats = {}  # stats[split][class][env] = [file_paths]
    for split in SPLITS:
        stats[split] = {}
        for cls in CLASSES:
            stats[split][cls] = {}
            for env in ENVS:
                d = DATA_DIR / split / cls / env
                if d.exists():
                    files = sorted([
                        d / f for f in os.listdir(d)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    ])
                else:
                    files = []
                stats[split][cls][env] = files
    return stats


def plot_class_distribution(stats):
    """Bar chart of class distribution across Train/Valid/Test."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(CLASSES))
    width = 0.25
    split_colors = {"Train": "#3498db", "Valid": "#2ecc71", "Test": "#e74c3c"}

    for i, split in enumerate(SPLITS):
        counts = []
        for cls in CLASSES:
            total = sum(len(stats[split][cls][env]) for env in ENVS)
            counts.append(total)
        bars = ax.bar(x + i * width, counts, width, label=split,
                      color=split_colors[split], edgecolor="white", linewidth=0.5)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                    str(count), ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Disease Class")
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution Across Splits", fontweight="bold", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.legend(frameon=True, fancybox=True, shadow=True)

    max_count = max(
        sum(len(stats["Train"][cls][env]) for env in ENVS) for cls in CLASSES
    )
    ax.set_ylim(0, max_count * 1.2)

    # Imbalance ratio
    train_counts = [sum(len(stats["Train"][cls][env]) for env in ENVS) for cls in CLASSES]
    ratio = max(train_counts) / max(min(train_counts), 1)
    ax.annotate(f"⚠️ Imbalance Ratio: {ratio:.1f}:1\n"
                f"Max: {CLASSES[np.argmax(train_counts)]} ({max(train_counts)})\n"
                f"Min: {CLASSES[np.argmin(train_counts)]} ({min(train_counts)})",
                xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffcccc", edgecolor="#e74c3c"))

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ class_distribution.png")


def plot_lab_vs_field(stats):
    """Stacked bar chart showing Lab vs Field per class."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Absolute counts
    ax = axes[0]
    lab_counts = [len(stats["Train"][cls]["Lab"]) for cls in CLASSES]
    field_counts = [len(stats["Train"][cls]["Field"]) for cls in CLASSES]
    x = np.arange(len(CLASSES))

    ax.bar(x, lab_counts, 0.6, label="Lab", color="#3498db", edgecolor="white")
    ax.bar(x, field_counts, 0.6, bottom=lab_counts, label="Field",
           color="#e67e22", edgecolor="white")

    for i, (l, f) in enumerate(zip(lab_counts, field_counts)):
        if l > 0:
            ax.text(i, l / 2, str(l), ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white")
        if f > 0:
            ax.text(i, l + f / 2, str(f), ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white")

    ax.set_xlabel("Disease Class")
    ax.set_ylabel("Number of Images")
    ax.set_title("Lab vs Field Distribution (Training Set — Absolute)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.legend(frameon=True, fancybox=True)

    # Percentage
    ax = axes[1]
    totals = [l + f for l, f in zip(lab_counts, field_counts)]
    lab_pct = [l / t * 100 if t > 0 else 0 for l, t in zip(lab_counts, totals)]
    field_pct = [f / t * 100 if t > 0 else 0 for f, t in zip(field_counts, totals)]

    ax.barh(x, lab_pct, 0.6, label="Lab", color="#3498db", edgecolor="white")
    ax.barh(x, field_pct, 0.6, left=lab_pct, label="Field",
            color="#e67e22", edgecolor="white")

    for i, (lp, fp) in enumerate(zip(lab_pct, field_pct)):
        if lp > 10:
            ax.text(lp / 2, i, f"{lp:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        if fp > 10:
            ax.text(lp + fp / 2, i, f"{fp:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

    ax.set_xlabel("Percentage (%)")
    ax.set_title("Lab vs Field Distribution (Training Set — Percentage)", fontweight="bold")
    ax.set_yticks(x)
    ax.set_yticklabels(CLASSES)
    ax.set_xlim(0, 100)
    ax.legend(frameon=True, fancybox=True)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "lab_vs_field_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ lab_vs_field_distribution.png")


def plot_imbalance_analysis(stats):
    """Detailed imbalance analysis for the project report."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    train_counts = [sum(len(stats["Train"][cls][env]) for env in ENVS) for cls in CLASSES]
    total = sum(train_counts)

    # 1. Pie chart of class proportions
    ax = axes[0, 0]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    explode = [0.1 if c == min(train_counts) else 0 for c in train_counts]
    wedges, texts, autotexts = ax.pie(train_counts, labels=CLASSES, colors=colors,
                                       explode=explode, autopct='%1.1f%%',
                                       pctdistance=0.85, startangle=90)
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax.set_title("Training Set Class Proportions", fontweight="bold", fontsize=13)

    # 2. Imbalance ratios (each class relative to max)
    ax = axes[0, 1]
    max_count = max(train_counts)
    ratios = [max_count / c if c > 0 else 0 for c in train_counts]
    bars = ax.barh(range(len(CLASSES)), ratios, color=colors, edgecolor="white")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Undersampling Ratio (relative to max class)")
    ax.set_title("Class Imbalance Ratios\n(Higher = more underrepresented)", fontweight="bold")
    for i, (bar, r) in enumerate(zip(bars, ratios)):
        ax.text(bar.get_width() + 0.1, i, f"{r:.1f}×", va="center", fontsize=10, fontweight="bold")
    ax.axvline(x=1, color="green", linestyle="--", alpha=0.5, label="Balanced (1.0×)")
    ax.legend()

    # 3. Lab/Field balance per class — heatmap with counts
    ax = axes[1, 0]
    lab_counts_all = {}
    field_counts_all = {}
    for split in SPLITS:
        lab_counts_all[split] = [len(stats[split][cls]["Lab"]) for cls in CLASSES]
        field_counts_all[split] = [len(stats[split][cls]["Field"]) for cls in CLASSES]

    # Show lab percentage for all splits
    data = np.zeros((len(CLASSES), len(SPLITS)))
    for j, split in enumerate(SPLITS):
        for i, cls in enumerate(CLASSES):
            total_cls = lab_counts_all[split][i] + field_counts_all[split][i]
            data[i, j] = lab_counts_all[split][i] / total_cls * 100 if total_cls > 0 else 50

    annot_text = np.array([[f"{data[i,j]:.0f}%\n({lab_counts_all[SPLITS[j]][i]}L/{field_counts_all[SPLITS[j]][i]}F)"
                            for j in range(len(SPLITS))] for i in range(len(CLASSES))])
    sns.heatmap(data, annot=annot_text, fmt="", cmap="RdYlBu", ax=ax,
                xticklabels=SPLITS, yticklabels=CLASSES,
                center=50, vmin=0, vmax=100,
                linewidths=.5, linecolor="white")
    ax.set_title("Lab Image Percentage per Class per Split\n(50% = balanced Lab/Field)", fontweight="bold")

    # 4. Total counts comparison across splits
    ax = axes[1, 1]
    split_totals = {split: [sum(len(stats[split][cls][env]) for env in ENVS) for cls in CLASSES]
                    for split in SPLITS}
    x = np.arange(len(CLASSES))
    w = 0.25
    split_colors = {"Train": "#3498db", "Valid": "#2ecc71", "Test": "#e74c3c"}
    for i, split in enumerate(SPLITS):
        ax.bar(x + i * w, split_totals[split], w, label=split,
               color=split_colors[split], edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.set_ylabel("Number of Images")
    ax.set_title("Sample Counts Across All Splits", fontweight="bold")
    ax.legend()

    plt.suptitle("Dataset Imbalance Analysis — Rice Plant Disease",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "imbalance_analysis.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ imbalance_analysis.png")


def plot_sample_grid(stats):
    """3-row × 6-col grid: for each class shows 1 Lab + 1 Field + 1 random sample."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 6, hspace=0.4, wspace=0.15)

    for col, cls in enumerate(CLASSES):
        for row, (label, env_pick) in enumerate(
            [("Lab", "Lab"), ("Field", "Field"), ("Random", None)]
        ):
            ax = fig.add_subplot(gs[row, col])

            if env_pick:
                files = stats["Train"][cls][env_pick]
            else:
                files = stats["Train"][cls]["Lab"] + stats["Train"][cls]["Field"]

            if files:
                img_path = random.choice(files)
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(f"{cls}\n({label})", fontsize=9, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{cls}\n({label})", fontsize=9)

            ax.axis("off")

    fig.suptitle("Sample Images per Class & Environment (Training Set)",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.savefig(OUTPUT_DIR / "sample_images_grid.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ sample_images_grid.png")


def plot_resolution_distribution(stats):
    """Histogram of image widths and heights."""
    widths, heights = [], []
    env_labels = []

    for cls in CLASSES:
        for env in ENVS:
            for fpath in stats["Train"][cls][env]:
                try:
                    img = Image.open(fpath)
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    env_labels.append(env)
                except Exception:
                    pass

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Width distribution
    ax = axes[0]
    ax.hist(widths, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Image Width Distribution", fontweight="bold")
    ax.axvline(np.median(widths), color="#e74c3c", linestyle="--",
               label=f"Median: {int(np.median(widths))}px")
    ax.legend()

    # Height distribution
    ax = axes[1]
    ax.hist(heights, bins=50, color="#2ecc71", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Height (pixels)")
    ax.set_ylabel("Count")
    ax.set_title("Image Height Distribution", fontweight="bold")
    ax.axvline(np.median(heights), color="#e74c3c", linestyle="--",
               label=f"Median: {int(np.median(heights))}px")
    ax.legend()

    # Scatter: width vs height
    ax = axes[2]
    colors = ["#3498db" if e == "Lab" else "#e67e22" for e in env_labels]
    ax.scatter(widths, heights, c=colors, alpha=0.3, s=8)
    ax.set_xlabel("Width (pixels)")
    ax.set_ylabel("Height (pixels)")
    ax.set_title("Resolution Scatter (Lab=Blue, Field=Orange)", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "resolution_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ resolution_distribution.png")


def plot_channel_statistics(stats):
    """Mean pixel intensity distribution per class (RGB channels)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    channel_names = ["Red", "Green", "Blue"]
    channel_colors = ["#e74c3c", "#2ecc71", "#3498db"]

    class_means = {cls: {ch: [] for ch in range(3)} for cls in CLASSES}

    for cls in CLASSES:
        files = stats["Train"][cls]["Lab"] + stats["Train"][cls]["Field"]
        sample_files = random.sample(files, min(50, len(files)))
        for fpath in sample_files:
            try:
                img = np.array(Image.open(fpath).convert("RGB").resize((64, 64)))
                for ch in range(3):
                    class_means[cls][ch].append(img[:, :, ch].mean())
            except Exception:
                pass

    for ch_idx, (ch_name, ch_color) in enumerate(zip(channel_names, channel_colors)):
        ax = axes[ch_idx]
        data = [class_means[cls][ch_idx] for cls in CLASSES]
        bp = ax.boxplot(data, labels=CLASSES, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(ch_color)
            patch.set_alpha(0.5)
        ax.set_title(f"{ch_name} Channel Mean Intensity", fontweight="bold")
        ax.set_ylabel("Pixel Value (0-255)")
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle("RGB Channel Statistics by Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "channel_statistics.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ channel_statistics.png")


def save_summary_json(stats):
    """Save a JSON summary of all counts."""
    summary = {}
    for split in SPLITS:
        summary[split] = {}
        for cls in CLASSES:
            summary[split][cls] = {
                env: len(stats[split][cls][env]) for env in ENVS
            }
            summary[split][cls]["total"] = sum(
                len(stats[split][cls][env]) for env in ENVS
            )

    # Add analysis metadata
    train_counts = [summary["Train"][cls]["total"] for cls in CLASSES]
    summary["_analysis"] = {
        "imbalance_ratio": max(train_counts) / max(min(train_counts), 1),
        "max_class": CLASSES[np.argmax(train_counts)],
        "min_class": CLASSES[np.argmin(train_counts)],
        "total_images": {split: sum(summary[split][cls]["total"] for cls in CLASSES) for split in SPLITS},
    }

    with open(OUTPUT_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  ✓ dataset_summary.json")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" Rice Plant Disease Dataset — Comprehensive Exploration")
    print("=" * 70)

    print("\n[1/8] Collecting statistics...")
    stats = collect_stats()

    # Print summary table
    print("\n  ┌─── Dataset Summary ──────────────────────────────────────────────┐")
    print(f"  │ {'Class':<20} {'Lab':>6} {'Field':>6} {'Total':>6} {'%':>7} {'Status':>10} │")
    print(f"  │{'-'*62}│")

    grand_total = sum(
        sum(len(stats["Train"][cls][env]) for env in ENVS) for cls in CLASSES
    )
    train_counts = []
    for cls in CLASSES:
        lab = len(stats["Train"][cls]["Lab"])
        field = len(stats["Train"][cls]["Field"])
        total = lab + field
        train_counts.append(total)
        pct = total / grand_total * 100
        status = "⚠️ MINORITY" if total < 200 else "OK"
        print(f"  │ {cls:<20} {lab:>6} {field:>6} {total:>6} {pct:>6.1f}% {status:>10} │")

    print(f"  │{'-'*62}│")
    print(f"  │ {'TOTAL':<20} {'':>6} {'':>6} {grand_total:>6} {100.0:>6.1f}%            │")
    print(f"  └────────────────────────────────────────────────────────────────┘")

    ratio = max(train_counts) / max(min(train_counts), 1)
    print(f"\n  ⚠️  Imbalance Ratio: {ratio:.1f}:1")
    print(f"      Majority: {CLASSES[np.argmax(train_counts)]} ({max(train_counts)})")
    print(f"      Minority: {CLASSES[np.argmin(train_counts)]} ({min(train_counts)})")

    print("\n[2/8] Plotting class distribution...")
    plot_class_distribution(stats)

    print("[3/8] Plotting lab vs field distribution...")
    plot_lab_vs_field(stats)

    print("[4/8] Creating imbalance analysis...")
    plot_imbalance_analysis(stats)

    print("[5/8] Creating sample image grid...")
    plot_sample_grid(stats)

    print("[6/8] Plotting resolution distribution...")
    plot_resolution_distribution(stats)

    print("[7/8] Computing channel statistics...")
    plot_channel_statistics(stats)

    print("[8/8] Saving summary JSON...")
    save_summary_json(stats)

    print("\n" + "=" * 70)
    print(f" ✅ All exploration outputs saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
