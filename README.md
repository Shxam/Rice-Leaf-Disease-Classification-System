# 🌾 ViT-XAI Rice Leaf Disease Classification System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-red) ![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive **Vision Transformer (ViT)** based system for automated rice leaf disease classification with **explainable AI (XAI)** methods. Handles severe class imbalance with advanced training techniques and provides multi-method interpretability through Grad-CAM, Attention Rollout, SHAP, and LIME.

---

## 📋 Overview

| Aspect | Details |
|--------|---------|
| **Model Architecture** | ViT-Base/16 (Vision Transformer) |
| **Disease Classes** | 6 (Bacterial Blight, Brown Spot, Healthy, Leaf Blast, Leaf Smut, Tungro) |
| **Dataset Size** | ~5,300 training, ~1,100 validation, ~900 test samples |
| **Key Challenge** | Severe class imbalance (Leaf Smut: 130 vs Healthy: 1214 = **9.3:1 ratio**) |
| **XAI Methods** | 4-method ensemble (Grad-CAM + Attention Rollout + SHAP + LIME) |
| **Environment Tracking** | Lab vs Field images analyzed separately |

### Dataset Structure
```
data/
├── Train/     (5 classes: Bacterial Blight, Brown Spot, Healthy, Leaf Blast, Leaf Smut, Tungro)
│   ├── Bacterial Blight/ → Lab/, Field/
│   ├── Brown Spot/ → Lab/, Field/
│   ├── Healthy/ → Lab/, Field/
│   ├── Leaf Blast/ → Lab/, Field/
│   ├── Leaf Smut/ → Lab/, Field/  ⚠️ MINORITY CLASS
│   └── Tungro/ → Lab/, Field/
├── Valid/     (similar structure)
└── Test/      (similar structure)
```

---

## 🔧 Key Features

### 1. **Imbalance Mitigation Pipeline**
- **Focal Loss** (γ=2.0) for hard-sample mining
- **Inverse-frequency weighted CrossEntropyLoss** (Leaf Smut gets ~9× weight)
- **WeightedRandomSampler** for balanced mini-batches
- **Label smoothing** (ε=0.1) to prevent overconfidence
- **Aggressive augmentation**: crop, flip, rotation, color jitter, random erasing
- **Cosine annealing with warmup** (3-epoch linear warmup, then cosine decay)

### 2. **Explainability Methods**
- 🎯 **Grad-CAM** — Gradient-based class activation maps
- 🔄 **Attention Rollout** — Aggregated ViT self-attention across 12 layers
- 📊 **SHAP** (GradientExplainer) — Shapley value pixel attribution
- 🎨 **LIME** (LimeImageExplainer) — Perturbation-based superpixel importance

### 3. **Comprehensive Evaluation**
- Confusion matrices (raw + row-normalized)
- Per-class metrics: Precision, Recall, F1
- ROC curves (One-vs-Rest) with AUC scores
- Precision-Recall curves (imbalance-aware)
- Lab vs Field performance breakdown
- Confidence distribution analysis
- Cohen's Kappa & Matthews Correlation Coefficient

### 4. **Advanced Training Techniques**
- Mixed precision training (AMP) with GradScaler
- Gradient accumulation (effective batch size: 64)
- Early stopping on validation macro-F1
- Per-class F1 monitoring to track minority class convergence
- Environment-aware metrics

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA 12.1+ (recommended, CPU fallback supported)
- 8 GB VRAM (RTX 4060 tested)

### Setup

```bash
# Clone repository
git clone https://github.com/Shxam/Rice-Leaf-Disease-Classification-System.git
cd Rice-Leaf-Disease-Classification-System

# Create conda environment
conda create -n mlenv python=3.9
conda activate mlenv

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm scikit-learn opencv-python pandas matplotlib seaborn
pip install pytorch-grad-cam shap lime Pillow
```

### Verified Package Versions
```
PyTorch:        2.5.1+cu121
timm:           1.0.9
scikit-learn:   1.6.1
OpenCV:         4.13.0
pandas:         2.2.3
seaborn:        0.13.2
shap:           0.49.1
lime:           0.2.0.1
matplotlib:     3.7+
```

---

## 🚀 Quick Start

### Option 1: Full Pipeline (Start to Finish)
```bash
cd Rice-Leaf-Disease-Classification-System
python explore_dataset.py    # 2 min  — Dataset analysis & visualizations
python train.py             # 30-60 min — Model training with early stopping
python evaluate.py          # 5 min  — Comprehensive evaluation
python explain.py           # 10-15 min — XAI analysis & visualizations
```

### Option 2: Train + Evaluate Only
```bash
python train.py && python evaluate.py
```

### Option 3: Run All Sequentially
```bash
python run_all.py
```

---

## 📊 Usage Details

### 1. **Dataset Exploration** (`explore_dataset.py`)
Generates statistical and visual insights:

```bash
python explore_dataset.py
```

**Outputs:**
- `outputs/exploration/class_distribution.png` — Class imbalance visualization
- `outputs/exploration/lab_vs_field_distribution.png` — Environment split per class
- `outputs/exploration/imbalance_analysis.png` — Severity metrics
- `outputs/exploration/sample_images_grid.png` — Representative images
- `outputs/exploration/resolution_distribution.png` — Image size analysis
- `outputs/exploration/channel_statistics.png` — RGB statistics
- `outputs/exploration/dataset_summary.json` — Numerical summary

### 2. **Model Training** (`train.py`)
Fine-tunes ViT-Base/16 on the imbalanced dataset:

```bash
python train.py
```

**Configuration:**
```python
LR = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50 (with early stopping, patience=12)
WARMUP_EPOCHS = 3
LABEL_SMOOTHING = 0.1
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
```

**Outputs:**
- `outputs/training/best_model.pth` — Best checkpoint (selected on val macro-F1)
- `outputs/training/training_history.json` — Epoch-by-epoch metrics
- `outputs/training/training_curves.png` — Loss, Accuracy, F1, LR schedules
- `outputs/training/per_class_f1_history.png` — Per-class F1 progression
- `outputs/training/class_weights_chart.png` — Inverse-frequency weights
- `outputs/training/training_config.json` — Hyperparameter record

### 3. **Model Evaluation** (`evaluate.py`)
Comprehensive test-set analysis:

```bash
python evaluate.py
```

**Outputs:**
- `outputs/evaluation/confusion_matrix.png` — Raw + normalized CMs
- `outputs/evaluation/env_confusion_matrices.png` — Lab vs Field CMs
- `outputs/evaluation/per_class_metrics.png` — P/R/F1 bars with sample counts
- `outputs/evaluation/roc_curves.png` — One-vs-Rest ROC (6 classes)
- `outputs/evaluation/precision_recall_curves.png` — PR curves (imbalance-aware)
- `outputs/evaluation/lab_vs_field_accuracy.png` — Environment performance heatmaps
- `outputs/evaluation/confidence_distribution.png` — Correct vs incorrect confidence
- `outputs/evaluation/classification_report.csv` — Full sklearn report
- `outputs/evaluation/evaluation_results.json` — All metrics in JSON

### 4. **Explainability Analysis** (`explain.py`)
Generates 4-method XAI visualizations:

```bash
python explain.py
```

**Outputs:**
- `outputs/explainability/gradcam_grid_all_classes.png` — Grad-CAM heatmaps
- `outputs/explainability/attention_rollout_grid.png` — ViT attention maps
- `outputs/explainability/shap_grid_all_classes.png` — SHAP pixel importance
- `outputs/explainability/lime_grid_all_classes.png` — LIME superpixel importance
- `outputs/explainability/xai_4method_comparison.png` — 4-method side-by-side
- `outputs/explainability/xai_comparison_grid.png` — Grad-CAM vs Attention
- `outputs/explainability/xai_method_summary.png` — Method agreement analysis
- `outputs/explainability/misclassified_explanations.png` — Dual explanations on errors
- `outputs/explainability/attention_statistics.png` — Activation + SHAP statistics
- `outputs/explainability/individual/` — Per-class detailed panels
- `outputs/explainability/gradcam_individual/` — Individual heatmaps
- `outputs/explainability/xai_summary.json` — XAI run metadata

---

## 📈 Model Performance

### Expected Results (on Test Set)
With the current pipeline, typical performance:

| Metric | Value |
|--------|-------|
| **Accuracy** | ~82-86% |
| **Balanced Accuracy** | ~75-80% |
| **Macro F1** | ~78-82% |
| **Weighted F1** | ~82-86% |

### Per-Class Performance (typical)
| Disease | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Bacterial Blight | 0.85 | 0.80 | 0.82 | ~150 |
| Brown Spot | 0.82 | 0.78 | 0.80 | ~120 |
| Healthy | 0.90 | 0.92 | 0.91 | ~150 |
| Leaf Blast | 0.78 | 0.75 | 0.76 | ~110 |
| **Leaf Smut** | 0.65 | 0.58 | 0.61 | ~15 | ⚠️ Minority
| Tungro | 0.83 | 0.81 | 0.82 | ~110 |

### Environment Performance (typical)
- **Lab** — Higher accuracy (~85%), controlled conditions
- **Field** — Lower accuracy (~75-80%), real-world variation

---

## 📁 Output Directory Structure

```
outputs/
├── exploration/
│   ├── class_distribution.png
│   ├── lab_vs_field_distribution.png
│   ├── imbalance_analysis.png
│   ├── sample_images_grid.png
│   ├── resolution_distribution.png
│   ├── channel_statistics.png
│   └── dataset_summary.json
│
├── training/
│   ├── best_model.pth                    ← Load this for inference
│   ├── training_history.json
│   ├── training_curves.png
│   ├── per_class_f1_history.png
│   ├── class_weights_chart.png
│   └── training_config.json
│
├── evaluation/
│   ├── confusion_matrix.png
│   ├── env_confusion_matrices.png
│   ├── per_class_metrics.png
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   ├── lab_vs_field_accuracy.png
│   ├── confidence_distribution.png
│   ├── classification_report.csv
│   └── evaluation_results.json
│
└── explainability/
    ├── gradcam_grid_all_classes.png      ← Gradient-based
    ├── attention_rollout_grid.png        ← Attention-based
    ├── shap_grid_all_classes.png         ← Shapley values
    ├── lime_grid_all_classes.png         ← Perturbation-based
    ├── xai_4method_comparison.png        ← All 4 methods
    ├── xai_comparison_grid.png
    ├── xai_method_summary.png
    ├── misclassified_explanations.png
    ├── attention_statistics.png
    ├── xai_summary.json
    ├── gradcam_individual/               ← Individual heatmaps per image
    └── individual/                       ← Detailed per-class analysis
```

---

## 🎯 Class Imbalance Handling

The system addresses the **9.3:1 imbalance** through:

1. **Loss Function**: Focal Loss down-weights easy, well-classified examples
2. **Sampling**: WeightedRandomSampler ensures balanced batches
3. **Class Weights**: Leaf Smut samples weighted ~9× more than Healthy
4. **Data Augmentation**: Aggressive transforms (rotate, color jitter, erasing) boost minority class generalization
5. **Metrics**: Macro-F1 and Balanced Accuracy prioritize minority class performance
6. **Monitoring**: Per-class F1 plots track convergence on each class separately

### Example: Leaf Smut (Minority Class)
- **Train samples**: 130 images (~1.6% of dataset)
- **Weight**: 8.8× higher than average
- **Augmentation**: Heavy random transforms to reduce overfitting
- **Monitored metric**: Macro-F1 (treats all classes equally)

---

## 🔍 Explainability Methods Comparison

| Method | Type | Strength | Limitation |
|--------|------|----------|-----------|
| **Grad-CAM** | Gradient-based | Fast, identifies decision regions | Limited to last layer |
| **Attention Rollout** | Attention-based | Native to ViT, lightweight | May highlight backgrounds |
| **SHAP** | Game-theoretic | Theoretically sound, pixel-level | Computationally expensive |
| **LIME** | Perturbation-based | Model-agnostic, intuitive | Local approximation only |

**Best practice**: Use all 4 methods together. Agreement indicates robust model behavior.

---

## 💾 Model Inference (Using Pre-trained Model)

To use the trained model for predictions on new images:

```python
import torch
from torchvision import transforms
from PIL import Image
import timm

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=6)
ckpt = torch.load("outputs/training/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

img = Image.open("leaf.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)
    pred_class = probs.argmax(1).item()
    pred_prob = probs[0, pred_class].item()

classes = ["Bacterial Blight", "Brown Spot", "Healthy", "Leaf Blast", "Leaf Smut", "Tungro"]
print(f"Prediction: {classes[pred_class]} ({pred_prob:.1%} confidence)")
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce `BATCH_SIZE` or use CPU (`device = "cpu"`) |
| **Very low Leaf Smut F1** | Dataset may be too small; consider collecting more samples or using synthetically augmented data |
| **Slow SHAP/LIME** | These methods are computationally expensive; reduce number of explanations sampled in `explain.py` |
| **Lab vs Field gap** | High gap indicates domain shift; consider Domain Adaptation techniques |
| **Model overfitting** | Increase `LABEL_SMOOTHING` or enable stronger augmentation |

---

## 📚 Dataset Citation & Attribution

If you use this system in research, please cite the original ViT-XAI Rice Leaf Disease Classification work:

```bibtex
@repository{ViT-XAI-Rice,
  title={ViT-XAI based Rice Leaf Disease Classification System},
  author={Varun N.},
  year={2024},
  url={https://github.com/varunn4/ViT-XAI-based-Rice-Leaf-Disease-Classification-system}
}
```

---

## 📖 References

- **Vision Transformers**: Dosovitskiy et al. (2021) — [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **Grad-CAM**: Selvaraju et al. (2017) — [Grad-CAM: Visual Explanations](https://arxiv.org/abs/1610.02055)
- **SHAP**: Lundberg & Lee (2017) — [A Unified Approach to Interpreting Predictions](https://arxiv.org/abs/1705.07874)
- **LIME**: Ribeiro et al. (2016) — [Why Should I Trust You?](https://arxiv.org/abs/1602.04938)
- **Focal Loss**: Lin et al. (2017) — [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

---

## 📝 License

This repository is a fork/adaptation. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m "Add improvement"`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📧 Contact & Support

For questions or issues:
- Open a GitHub Issue
- Check the **Troubleshooting** section above
- Review code comments in individual scripts

---

## ⭐ Acknowledgments

- Original dataset authors and domain experts
- PyTorch and timm community for excellent tooling
- XAI research community for interpretability methods

---

**Happy disease classification! 🌾🤖**