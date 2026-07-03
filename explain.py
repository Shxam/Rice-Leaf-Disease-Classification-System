"""
explain.py — Explainability: Grad-CAM + Attention Rollout + SHAP + LIME
=========================================================================
Generates XAI heatmaps for Vision Transformer predictions.
Outputs:
  - Grad-CAM heatmaps (per class × per environment)
  - Attention Rollout maps
  - SHAP value maps (GradientExplainer)
  - LIME superpixel explanations
  - Side-by-side 4-method XAI comparison panels
  - Misclassified sample explanations
  - Per-class attention & SHAP statistics
  - Top-K most/least confident predictions with explanations
"""

import os
import json
import random
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
import cv2
from PIL import Image
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path("data")
MODEL_PATH = Path("outputs/training/best_model.pth")
OUTPUT_DIR = Path("outputs/explainability")
CLASSES = ["Bacterial Blight", "Brown Spot", "Healthy", "Leaf Blast", "Leaf Smut", "Tungro"]
NUM_CLASSES = len(CLASSES)
ENVS = ["Lab", "Field"]
IMG_SIZE = 224
MODEL_NAME = "vit_base_patch16_224"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200,
    "font.family": "sans-serif", "font.size": 10,
    "figure.facecolor": "white"
})

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])


def denormalize(tensor):
    """Convert normalized tensor back to 0-1 range numpy array."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def reshape_transform(tensor):
    """Reshape ViT attention output for Grad-CAM (remove CLS token, reshape to spatial)."""
    result = tensor[:, 1:, :]  # remove CLS token
    h = w = int(result.shape[1] ** 0.5)
    result = result.reshape(result.shape[0], h, w, result.shape[2])
    result = result.permute(0, 3, 1, 2)  # B, C, H, W
    return result


def get_attention_rollout(model, input_tensor, device):
    """Compute attention rollout across all transformer layers.

    Uses a clean hook-based approach compatible with timm >= 1.0
    (handles fused_attn by temporarily disabling it so explicit
    attention weights are materialized and capturable via attn_drop hooks).
    """
    model.eval()

    # ── 1. Temporarily disable fused attention so weights are explicit ───
    fused_states = {}
    for name, module in model.named_modules():
        if hasattr(module, 'fused_attn'):
            fused_states[name] = module.fused_attn
            module.fused_attn = False          # forces the non-fused path

    # ── 2. Register hooks on attn_drop to capture softmaxed attention ────
    attention_maps = []
    hooks = []

    for block in model.blocks:
        def _make_hook():
            def _hook_fn(module, inp, out):
                # inp[0] = softmaxed attention matrix  (B, heads, N, N)
                attention_maps.append(inp[0].detach().cpu())
            return _hook_fn
        h = block.attn.attn_drop.register_forward_hook(_make_hook())
        hooks.append(h)

    # ── 3. Forward pass ──────────────────────────────────────────────────
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        _ = model(input_tensor)

    # ── 4. Cleanup: remove hooks, restore fused_attn flags ───────────────
    for h in hooks:
        h.remove()
    for name, module in model.named_modules():
        if name in fused_states:
            module.fused_attn = fused_states[name]

    if not attention_maps:
        return None

    # ── 5. Compute rollout ───────────────────────────────────────────────
    rollout = torch.eye(attention_maps[0].shape[-1]).unsqueeze(0)
    for attn in attention_maps:
        attn_mean = attn.mean(dim=1)                                  # avg heads
        attn_mean = attn_mean + torch.eye(attn_mean.shape[-1]).unsqueeze(0)
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
        rollout = rollout @ attn_mean

    # CLS token attention → patch grid
    mask = rollout[0, 0, 1:]           # CLS -> patches
    h = w = int(mask.shape[0] ** 0.5)
    mask = mask.reshape(h, w).numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    return mask


# ── SHAP Explainer ─────────────────────────────────────────────────────────
class _SingleClassWrapper(torch.nn.Module):
    """Wraps a classifier to output only one class logit — needed for SHAP
    GradientExplainer so the output is scalar and SHAP values have the same
    spatial shape as the input (avoids the multi-class broadcast error)."""
    def __init__(self, model, target_class):
        super().__init__()
        self.model = model
        self.target_class = target_class
    def forward(self, x):
        return self.model(x)[:, self.target_class : self.target_class + 1]


def generate_shap_explanation(model, input_tensor, background_tensors, device, pred_class):
    """Generate SHAP values using GradientExplainer for an image.

    Returns a heatmap (H, W) of mean absolute SHAP values across RGB channels.
    """
    import shap

    model.eval()
    background = background_tensors.to(device)
    input_t = input_tensor.to(device)

    # Wrap model to output only the predicted-class logit.
    # This guarantees shap_values shape is (1, C, H, W) — same spatial
    # layout as the input, avoiding the multi-class broadcast error.
    wrapped = _SingleClassWrapper(model, pred_class)
    wrapped.eval()

    explainer = shap.GradientExplainer(wrapped, background)
    shap_values = explainer.shap_values(input_t)

    # --- Robust extraction of (H, W) heatmap from any SHAP output shape ---
    # GradientExplainer can return:
    #   list[ndarray]  — one array per output neuron
    #   ndarray        — single array, shape varies
    if isinstance(shap_values, list):
        sv = np.array(shap_values[0])
    else:
        sv = np.array(shap_values)

    sv = np.squeeze(sv)  # Remove all size-1 dims: e.g. (1,3,224,224) → (3,224,224)

    if sv.ndim == 3:
        # (C, H, W) or (H, W, C) — aggregate across channels
        if sv.shape[0] == 3:
            heatmap = np.abs(sv).mean(axis=0)   # (C, H, W) → (H, W)
        elif sv.shape[2] == 3:
            heatmap = np.abs(sv).mean(axis=2)   # (H, W, C) → (H, W)
        else:
            heatmap = np.abs(sv).mean(axis=0)   # fallback
    elif sv.ndim == 2:
        heatmap = np.abs(sv)
    else:
        # Higher dims — flatten to last 2 spatial dims
        while sv.ndim > 3:
            sv = sv[0]
        sv = np.squeeze(sv)
        if sv.ndim == 3:
            heatmap = np.abs(sv).mean(axis=0)
        else:
            heatmap = np.abs(sv)

    heatmap = heatmap.astype(np.float64)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap


def generate_shap_explanations(model, device, samples, transform, background_tensors):
    """Generate SHAP heatmaps for all sample images."""
    results = []
    total = sum(len(samples[c][e]) for c in CLASSES for e in ENVS)
    count = 0

    for cls_idx, cls in enumerate(CLASSES):
        for env in ENVS:
            for img_path in samples[cls][env]:
                count += 1
                raw_img = Image.open(img_path).convert("RGB")
                raw_img_resized = raw_img.resize((IMG_SIZE, IMG_SIZE))
                raw_np = np.array(raw_img_resized).astype(np.float32) / 255.0

                input_tensor = transform(raw_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    pred = output.argmax(1).item()
                    probs = F.softmax(output, dim=1)[0]
                    conf = probs[pred].item()

                try:
                    shap_map = generate_shap_explanation(
                        model, input_tensor, background_tensors, device, pred
                    )
                    # Ensure shap_map is strictly 2D (H, W)
                    shap_map = np.squeeze(shap_map)
                    if shap_map.ndim != 2:
                        raise ValueError(f"SHAP heatmap has unexpected shape {shap_map.shape}")
                    if shap_map.shape != (IMG_SIZE, IMG_SIZE):
                        shap_map = cv2.resize(shap_map, (IMG_SIZE, IMG_SIZE))
                    # Create colored overlay
                    shap_colored = cv2.applyColorMap(
                        (shap_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
                    )
                    shap_colored = cv2.cvtColor(shap_colored, cv2.COLOR_BGR2RGB)
                    overlay = (0.4 * raw_np * 255 + 0.6 * shap_colored).astype(np.uint8)
                except Exception as e:
                    print(f"    ⚠ SHAP failed for {img_path.name}: {e}")
                    shap_map = np.zeros((IMG_SIZE, IMG_SIZE))
                    overlay = (raw_np * 255).astype(np.uint8)

                results.append({
                    "class": cls, "env": env, "path": str(img_path),
                    "pred": CLASSES[pred], "conf": conf,
                    "raw_img": raw_np, "shap_overlay": overlay,
                    "shap_map": shap_map, "cls_idx": cls_idx,
                    "correct": pred == cls_idx,
                })
                print(f"\r    SHAP: {count}/{total}", end="", flush=True)

    print()
    return results


# ── LIME Explainer ─────────────────────────────────────────────────────────
def generate_lime_explanation(model, raw_np, transform, device, pred_class, num_samples=300):
    """Generate LIME explanation for an image.

    Returns:
        explanation_img: RGB image with LIME superpixels highlighted
        mask: binary mask of top positive superpixels
    """
    from lime import lime_image

    explainer = lime_image.LimeImageExplainer()

    # LIME predict function: takes batch of numpy images (N, H, W, 3) in [0,1]
    def predict_fn(images):
        batch = []
        for img in images:
            # img is (H, W, 3) numpy in [0..1] float or [0..255] uint8
            if img.max() > 1.0:
                img = img.astype(np.float32) / 255.0
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            tensor = transform(pil_img)
            batch.append(tensor)
        batch = torch.stack(batch).to(device)
        with torch.no_grad():
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    # raw_np is (H, W, 3) float32 in [0, 1]
    explanation = explainer.explain_instance(
        (raw_np * 255).astype(np.uint8),
        predict_fn,
        top_labels=NUM_CLASSES,
        hide_color=0,
        num_samples=num_samples,
        batch_size=32,
    )

    # Get explanation for the predicted class
    temp, mask = explanation.get_image_and_mask(
        pred_class,
        positive_only=True,
        num_features=5,
        hide_rest=False,
    )

    # Create a clean colored overlay
    from skimage.segmentation import mark_boundaries
    explanation_img = mark_boundaries(
        (raw_np * 255).astype(np.uint8) / 255.0,
        mask,
        color=(0, 1, 0),   # green boundaries
        mode='thick'
    )

    # Also create a heatmap from LIME weights
    segments = explanation.segments
    heatmap = np.zeros(segments.shape, dtype=np.float64)
    if pred_class in explanation.local_exp:
        for seg_idx, weight in explanation.local_exp[pred_class]:
            heatmap[segments == seg_idx] = weight

    # Normalize heatmap to [0, 1]
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    return explanation_img, mask, heatmap


def generate_lime_explanations(model, device, samples, transform):
    """Generate LIME explanations for all sample images."""
    results = []
    total = sum(len(samples[c][e]) for c in CLASSES for e in ENVS)
    count = 0

    for cls_idx, cls in enumerate(CLASSES):
        for env in ENVS:
            for img_path in samples[cls][env]:
                count += 1
                raw_img = Image.open(img_path).convert("RGB")
                raw_img_resized = raw_img.resize((IMG_SIZE, IMG_SIZE))
                raw_np = np.array(raw_img_resized).astype(np.float32) / 255.0

                input_tensor = transform(raw_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    pred = output.argmax(1).item()
                    probs = F.softmax(output, dim=1)[0]
                    conf = probs[pred].item()

                try:
                    lime_img, lime_mask, lime_heatmap = generate_lime_explanation(
                        model, raw_np, transform, device, pred
                    )
                except Exception as e:
                    print(f"    ⚠ LIME failed for {img_path.name}: {e}")
                    lime_img = raw_np
                    lime_mask = np.zeros((IMG_SIZE, IMG_SIZE))
                    lime_heatmap = np.zeros((IMG_SIZE, IMG_SIZE))

                results.append({
                    "class": cls, "env": env, "path": str(img_path),
                    "pred": CLASSES[pred], "conf": conf,
                    "raw_img": raw_np, "lime_img": lime_img,
                    "lime_mask": lime_mask, "lime_heatmap": lime_heatmap,
                    "cls_idx": cls_idx, "correct": pred == cls_idx,
                })
                print(f"\r    LIME: {count}/{total}", end="", flush=True)

    print()
    return results


# ── Sample Collection ──────────────────────────────────────────────────────
def collect_sample_images(split="Test", n_per_env=2):
    """Collect sample images: n_per_env from Lab and Field for each class."""
    samples = {}
    for cls in CLASSES:
        samples[cls] = {}
        for env in ENVS:
            d = DATA_DIR / split / cls / env
            if not d.exists():
                samples[cls][env] = []
                continue
            files = [d / f for f in sorted(os.listdir(d))
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            random.shuffle(files)
            samples[cls][env] = files[:n_per_env]
    return samples


def build_shap_background(transform, n=50):
    """Build a small background dataset for SHAP from train data."""
    bg_images = []
    for cls in CLASSES:
        for env in ENVS:
            d = DATA_DIR / "Train" / cls / env
            if not d.exists():
                continue
            files = [d / f for f in sorted(os.listdir(d))
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            random.shuffle(files)
            for f in files[:max(1, n // (NUM_CLASSES * 2))]:
                img = Image.open(f).convert("RGB")
                bg_images.append(transform(img))
                if len(bg_images) >= n:
                    return torch.stack(bg_images)
    return torch.stack(bg_images) if bg_images else None


# ── Grad-CAM ───────────────────────────────────────────────────────────────
def generate_gradcam_heatmaps(model, device, samples, transform):
    """Generate Grad-CAM heatmaps for sample images."""
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape_transform)

    results = []
    for cls_idx, cls in enumerate(CLASSES):
        for env in ENVS:
            for img_path in samples[cls][env]:
                raw_img = Image.open(img_path).convert("RGB")
                raw_img_resized = raw_img.resize((IMG_SIZE, IMG_SIZE))
                raw_np = np.array(raw_img_resized).astype(np.float32) / 255.0

                input_tensor = transform(raw_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    pred = output.argmax(1).item()
                    probs = F.softmax(output, dim=1)[0]
                    conf = probs[pred].item()
                    top3_idx = probs.topk(3).indices.cpu().numpy()
                    top3_probs = probs.topk(3).values.cpu().numpy()

                # Grad-CAM for predicted class
                targets = [ClassifierOutputTarget(pred)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                cam_image = show_cam_on_image(raw_np, grayscale_cam[0], use_rgb=True)

                # Grad-CAM for true class (if different from pred)
                true_cam_image = None
                if pred != cls_idx:
                    true_targets = [ClassifierOutputTarget(cls_idx)]
                    true_grayscale_cam = cam(input_tensor=input_tensor, targets=true_targets)
                    true_cam_image = show_cam_on_image(raw_np, true_grayscale_cam[0], use_rgb=True)

                results.append({
                    "class": cls, "env": env, "path": str(img_path),
                    "pred": CLASSES[pred], "conf": conf,
                    "top3": [(CLASSES[idx], float(p)) for idx, p in zip(top3_idx, top3_probs)],
                    "raw_img": raw_np, "cam_img": cam_image,
                    "true_cam_img": true_cam_image,
                    "cam_mask": grayscale_cam[0], "cls_idx": cls_idx,
                    "correct": pred == cls_idx,
                })
    return results


# ── Attention Rollout ──────────────────────────────────────────────────────
def generate_attention_maps(model, device, samples, transform):
    """Generate attention rollout maps for sample images."""
    results = []
    for cls_idx, cls in enumerate(CLASSES):
        for env in ENVS:
            for img_path in samples[cls][env]:
                raw_img = Image.open(img_path).convert("RGB")
                raw_img_resized = raw_img.resize((IMG_SIZE, IMG_SIZE))
                raw_np = np.array(raw_img_resized).astype(np.float32) / 255.0

                input_tensor = transform(raw_img).unsqueeze(0)

                attn_mask = get_attention_rollout(model, input_tensor, device)
                if attn_mask is not None:
                    attn_colored = cv2.applyColorMap(
                        (attn_mask * 255).astype(np.uint8), cv2.COLORMAP_JET
                    )
                    attn_colored = cv2.cvtColor(attn_colored, cv2.COLOR_BGR2RGB)
                    overlay = (0.5 * raw_np * 255 + 0.5 * attn_colored).astype(np.uint8)
                else:
                    overlay = (raw_np * 255).astype(np.uint8)
                    attn_mask = np.zeros((IMG_SIZE, IMG_SIZE))

                with torch.no_grad():
                    output = model(input_tensor.to(device))
                    pred = output.argmax(1).item()

                results.append({
                    "class": cls, "env": env, "path": str(img_path),
                    "pred": CLASSES[pred], "raw_img": raw_np,
                    "attn_overlay": overlay, "attn_mask": attn_mask,
                })
    return results


# ── Plotting Functions ─────────────────────────────────────────────────────

def plot_gradcam_grid(gradcam_results):
    """Publication-ready grid: one row per class, columns = Original | Grad-CAM."""
    n_classes = len(CLASSES)
    fig, axes = plt.subplots(n_classes, 4, figsize=(18, n_classes * 3.8))

    col_titles = ["Lab Original", "Lab Grad-CAM", "Field Original", "Field Grad-CAM"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontweight="bold", fontsize=11)

    for i, cls in enumerate(CLASSES):
        cls_results = [r for r in gradcam_results if r["class"] == cls]
        lab_results = [r for r in cls_results if r["env"] == "Lab"]
        field_results = [r for r in cls_results if r["env"] == "Field"]

        if lab_results:
            r = lab_results[0]
            axes[i, 0].imshow(r["raw_img"])
            axes[i, 1].imshow(r["cam_img"])
            status = "✓" if r["correct"] else "✗"
            axes[i, 1].set_xlabel(f"{status} Pred: {r['pred']} ({r['conf']:.1%})", fontsize=8,
                                   color="green" if r["correct"] else "red")
        else:
            for j in [0, 1]:
                axes[i, j].text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=axes[i, j].transAxes, fontsize=14, color="gray")

        if field_results:
            r = field_results[0]
            axes[i, 2].imshow(r["raw_img"])
            axes[i, 3].imshow(r["cam_img"])
            status = "✓" if r["correct"] else "✗"
            axes[i, 3].set_xlabel(f"{status} Pred: {r['pred']} ({r['conf']:.1%})", fontsize=8,
                                   color="green" if r["correct"] else "red")
        else:
            for j in [2, 3]:
                axes[i, j].text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=axes[i, j].transAxes, fontsize=14, color="gray")

        axes[i, 0].set_ylabel(cls, fontweight="bold", fontsize=11, rotation=90, labelpad=15)

        for j in range(4):
            axes[i, j].axis("off")

    fig.suptitle("Grad-CAM Heatmaps: Lab vs Field (per class)\n"
                 "Red regions = high influence on prediction",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "gradcam_grid_all_classes.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ gradcam_grid_all_classes.png")


def plot_attention_grid(attn_results):
    """Grid of attention rollout maps."""
    n_classes = len(CLASSES)
    fig, axes = plt.subplots(n_classes, 4, figsize=(18, n_classes * 3.8))

    col_titles = ["Lab Original", "Lab Attention", "Field Original", "Field Attention"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontweight="bold", fontsize=11)

    for i, cls in enumerate(CLASSES):
        cls_results = [r for r in attn_results if r["class"] == cls]
        lab_results = [r for r in cls_results if r["env"] == "Lab"]
        field_results = [r for r in cls_results if r["env"] == "Field"]

        if lab_results:
            r = lab_results[0]
            axes[i, 0].imshow(r["raw_img"])
            axes[i, 1].imshow(r["attn_overlay"])
        else:
            for j in [0, 1]:
                axes[i, j].text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=axes[i, j].transAxes, fontsize=14, color="gray")

        if field_results:
            r = field_results[0]
            axes[i, 2].imshow(r["raw_img"])
            axes[i, 3].imshow(r["attn_overlay"])
        else:
            for j in [2, 3]:
                axes[i, j].text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=axes[i, j].transAxes, fontsize=14, color="gray")

        axes[i, 0].set_ylabel(cls, fontweight="bold", fontsize=11, rotation=90, labelpad=15)
        for j in range(4):
            axes[i, j].axis("off")

    fig.suptitle("Attention Rollout Maps: Lab vs Field (per class)\n"
                 "Shows where the ViT 'looks' across all 12 transformer layers",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "attention_rollout_grid.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ attention_rollout_grid.png")


def plot_shap_grid(shap_results):
    """Grid of SHAP value heatmaps: Lab vs Field per class."""
    n_classes = len(CLASSES)
    fig, axes = plt.subplots(n_classes, 4, figsize=(18, n_classes * 3.8))

    col_titles = ["Lab Original", "Lab SHAP", "Field Original", "Field SHAP"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontweight="bold", fontsize=11)

    for i, cls in enumerate(CLASSES):
        cls_results = [r for r in shap_results if r["class"] == cls]
        lab_results = [r for r in cls_results if r["env"] == "Lab"]
        field_results = [r for r in cls_results if r["env"] == "Field"]

        if lab_results:
            r = lab_results[0]
            axes[i, 0].imshow(r["raw_img"])
            axes[i, 1].imshow(r["shap_overlay"])
            status = "✓" if r["correct"] else "✗"
            axes[i, 1].set_xlabel(f"{status} Pred: {r['pred']} ({r['conf']:.1%})", fontsize=8,
                                   color="green" if r["correct"] else "red")
        else:
            for j in [0, 1]:
                axes[i, j].text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=axes[i, j].transAxes, fontsize=14, color="gray")

        if field_results:
            r = field_results[0]
            axes[i, 2].imshow(r["raw_img"])
            axes[i, 3].imshow(r["shap_overlay"])
            status = "✓" if r["correct"] else "✗"
            axes[i, 3].set_xlabel(f"{status} Pred: {r['pred']} ({r['conf']:.1%})", fontsize=8,
                                   color="green" if r["correct"] else "red")
        else:
            for j in [2, 3]:
                axes[i, j].text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=axes[i, j].transAxes, fontsize=14, color="gray")

        axes[i, 0].set_ylabel(cls, fontweight="bold", fontsize=11, rotation=90, labelpad=15)
        for j in range(4):
            axes[i, j].axis("off")

    fig.suptitle("SHAP Value Heatmaps: Lab vs Field (per class)\n"
                 "Bright = high Shapley value (pixel importance for prediction)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_grid_all_classes.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ shap_grid_all_classes.png")


def plot_lime_grid(lime_results):
    """Grid of LIME explanations: Lab vs Field per class."""
    n_classes = len(CLASSES)
    fig, axes = plt.subplots(n_classes, 4, figsize=(18, n_classes * 3.8))

    col_titles = ["Lab Original", "Lab LIME", "Field Original", "Field LIME"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontweight="bold", fontsize=11)

    for i, cls in enumerate(CLASSES):
        cls_results = [r for r in lime_results if r["class"] == cls]
        lab_results = [r for r in cls_results if r["env"] == "Lab"]
        field_results = [r for r in cls_results if r["env"] == "Field"]

        if lab_results:
            r = lab_results[0]
            axes[i, 0].imshow(r["raw_img"])
            axes[i, 1].imshow(r["lime_img"])
            status = "✓" if r["correct"] else "✗"
            axes[i, 1].set_xlabel(f"{status} Pred: {r['pred']} ({r['conf']:.1%})", fontsize=8,
                                   color="green" if r["correct"] else "red")
        else:
            for j in [0, 1]:
                axes[i, j].text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=axes[i, j].transAxes, fontsize=14, color="gray")

        if field_results:
            r = field_results[0]
            axes[i, 2].imshow(r["raw_img"])
            axes[i, 3].imshow(r["lime_img"])
            status = "✓" if r["correct"] else "✗"
            axes[i, 3].set_xlabel(f"{status} Pred: {r['pred']} ({r['conf']:.1%})", fontsize=8,
                                   color="green" if r["correct"] else "red")
        else:
            for j in [2, 3]:
                axes[i, j].text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=axes[i, j].transAxes, fontsize=14, color="gray")

        axes[i, 0].set_ylabel(cls, fontweight="bold", fontsize=11, rotation=90, labelpad=15)
        for j in range(4):
            axes[i, j].axis("off")

    fig.suptitle("LIME Explanations: Lab vs Field (per class)\n"
                 "Green boundaries = most influential superpixels for prediction",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "lime_grid_all_classes.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ lime_grid_all_classes.png")


def plot_4method_comparison(gradcam_results, attn_results, shap_results, lime_results):
    """Side-by-side comparison: Original | Grad-CAM | SHAP | LIME | Attention (one row per class)."""
    n_classes = len(CLASSES)

    # Two rows per class: Lab and Field
    fig, axes = plt.subplots(n_classes, 5, figsize=(25, n_classes * 3.8))

    col_titles = ["Original", "Grad-CAM", "SHAP", "LIME", "Attention Rollout"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontweight="bold", fontsize=11)

    for i, cls in enumerate(CLASSES):
        # Use Lab image for the comparison
        gc = [r for r in gradcam_results if r["class"] == cls and r["env"] == "Lab"]
        at = [r for r in attn_results if r["class"] == cls and r["env"] == "Lab"]
        sh = [r for r in shap_results if r["class"] == cls and r["env"] == "Lab"]
        lm = [r for r in lime_results if r["class"] == cls and r["env"] == "Lab"]

        # Fall back to Field if no Lab
        if not gc:
            gc = [r for r in gradcam_results if r["class"] == cls and r["env"] == "Field"]
        if not at:
            at = [r for r in attn_results if r["class"] == cls and r["env"] == "Field"]
        if not sh:
            sh = [r for r in shap_results if r["class"] == cls and r["env"] == "Field"]
        if not lm:
            lm = [r for r in lime_results if r["class"] == cls and r["env"] == "Field"]

        if gc:
            axes[i, 0].imshow(gc[0]["raw_img"])
            axes[i, 1].imshow(gc[0]["cam_img"])
        if sh:
            axes[i, 2].imshow(sh[0]["shap_overlay"])
        if lm:
            axes[i, 3].imshow(lm[0]["lime_img"])
        if at:
            axes[i, 4].imshow(at[0]["attn_overlay"])

        axes[i, 0].set_ylabel(cls, fontweight="bold", fontsize=10, rotation=90, labelpad=10)
        for j in range(5):
            axes[i, j].axis("off")

    fig.suptitle("XAI Method Comparison: 4 Explainability Approaches\n"
                 "Grad-CAM (gradient) | SHAP (Shapley) | LIME (perturbation) | Attention (rollout)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "xai_4method_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ xai_4method_comparison.png")


def plot_comparison_panels(gradcam_results, attn_results):
    """Side-by-side: Original | Grad-CAM | Attention for each class (Lab + Field rows)."""
    n_classes = len(CLASSES)
    fig, axes = plt.subplots(n_classes, 6, figsize=(24, n_classes * 3.5))

    col_titles = ["Lab Original", "Lab Grad-CAM", "Lab Attention",
                  "Field Original", "Field Grad-CAM", "Field Attention"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontweight="bold", fontsize=9)

    for i, cls in enumerate(CLASSES):
        gc_lab = [r for r in gradcam_results if r["class"] == cls and r["env"] == "Lab"]
        gc_field = [r for r in gradcam_results if r["class"] == cls and r["env"] == "Field"]
        at_lab = [r for r in attn_results if r["class"] == cls and r["env"] == "Lab"]
        at_field = [r for r in attn_results if r["class"] == cls and r["env"] == "Field"]

        # Lab
        if gc_lab:
            axes[i, 0].imshow(gc_lab[0]["raw_img"])
            axes[i, 1].imshow(gc_lab[0]["cam_img"])
        if at_lab:
            axes[i, 2].imshow(at_lab[0]["attn_overlay"])

        # Field
        if gc_field:
            axes[i, 3].imshow(gc_field[0]["raw_img"])
            axes[i, 4].imshow(gc_field[0]["cam_img"])
        if at_field:
            axes[i, 5].imshow(at_field[0]["attn_overlay"])

        axes[i, 0].set_ylabel(cls, fontweight="bold", fontsize=10, rotation=90, labelpad=10)
        for j in range(6):
            axes[i, j].axis("off")

    fig.suptitle("XAI Comparison: Grad-CAM vs Attention Rollout (Lab & Field)\n"
                 "Grad-CAM = gradient-based, Attention = pure attention flow",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "xai_comparison_grid.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ xai_comparison_grid.png")


def generate_misclassified_explanations(model, device, transform):
    """Find misclassified test images and explain them with Grad-CAM."""
    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape_transform)

    misclassified = []
    for cls_idx, cls in enumerate(CLASSES):
        for env in ENVS:
            d = DATA_DIR / "Test" / cls / env
            if not d.exists():
                continue
            for fname in sorted(os.listdir(d)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = d / fname
                raw_img = Image.open(img_path).convert("RGB")
                raw_resized = raw_img.resize((IMG_SIZE, IMG_SIZE))
                raw_np = np.array(raw_resized).astype(np.float32) / 255.0
                input_tensor = transform(raw_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    pred = output.argmax(1).item()
                    probs = F.softmax(output, dim=1)[0]
                    conf = probs[pred].item()
                    true_conf = probs[cls_idx].item()

                if pred != cls_idx:
                    # Grad-CAM for predicted class
                    targets = [ClassifierOutputTarget(pred)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                    cam_image_pred = show_cam_on_image(raw_np, grayscale_cam[0], use_rgb=True)

                    # Grad-CAM for true class
                    targets_true = [ClassifierOutputTarget(cls_idx)]
                    grayscale_cam_true = cam(input_tensor=input_tensor, targets=targets_true)
                    cam_image_true = show_cam_on_image(raw_np, grayscale_cam_true[0], use_rgb=True)

                    misclassified.append({
                        "true": cls, "pred": CLASSES[pred],
                        "pred_conf": conf, "true_conf": true_conf,
                        "env": env, "raw_img": raw_np,
                        "cam_img_pred": cam_image_pred,
                        "cam_img_true": cam_image_true,
                    })

    if not misclassified:
        print("  No misclassified samples found!")
        return

    # Sort by confidence (most confidently wrong first)
    misclassified.sort(key=lambda x: x["pred_conf"], reverse=True)

    n = min(len(misclassified), 12)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 9, figsize=(28, rows * 4))
    if rows == 1:
        axes = axes[np.newaxis, :]

    for idx in range(n):
        r = misclassified[idx]
        row = idx // 3
        col_base = (idx % 3) * 3

        axes[row, col_base].imshow(r["raw_img"])
        axes[row, col_base].set_title(f"True: {r['true']}\n({r['env']})", fontsize=8, color="green")

        axes[row, col_base + 1].imshow(r["cam_img_pred"])
        axes[row, col_base + 1].set_title(f"Pred CAM: {r['pred']}\n({r['pred_conf']:.1%})",
                                            fontsize=8, color="red")

        axes[row, col_base + 2].imshow(r["cam_img_true"])
        axes[row, col_base + 2].set_title(f"True CAM: {r['true']}\n({r['true_conf']:.1%})",
                                            fontsize=8, color="blue")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(f"Misclassified Samples with Dual Grad-CAM ({n}/{len(misclassified)} shown)\n"
                 f"Left=Original | Middle=Pred class CAM | Right=True class CAM",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "misclassified_explanations.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ misclassified_explanations.png ({n}/{len(misclassified)} samples)")

    # Save misclassification summary
    summary = []
    for m in misclassified:
        summary.append({
            "true": m["true"], "pred": m["pred"], "env": m["env"],
            "pred_conf": round(m["pred_conf"], 4),
            "true_conf": round(m["true_conf"], 4),
        })
    with open(OUTPUT_DIR / "misclassified_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ misclassified_summary.json ({len(misclassified)} errors)")


def plot_attention_statistics(gradcam_results, shap_results=None):
    """Statistical analysis of Grad-CAM and SHAP activation patterns per class."""
    n_plots = 3 if shap_results else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    x = np.arange(len(CLASSES))

    # ── Panel 1: Mean Grad-CAM activation per class ──
    class_means = {}
    class_stds = {}
    for cls in CLASSES:
        masks = [r["cam_mask"] for r in gradcam_results if r["class"] == cls]
        if masks:
            class_means[cls] = np.mean([m.mean() for m in masks])
            class_stds[cls] = np.std([m.mean() for m in masks])

    ax = axes[0]
    bars = ax.bar(x, [class_means.get(c, 0) for c in CLASSES], 0.6,
                  yerr=[class_stds.get(c, 0) for c in CLASSES],
                  color=colors, edgecolor="white", capsize=5)
    ax.set_xlabel("Disease Class")
    ax.set_ylabel("Mean Grad-CAM Activation")
    ax.set_title("Mean Grad-CAM Activation by Class\n(Higher = more focused attention)",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Grad-CAM Lab vs Field ──
    ax = axes[1]
    lab_means = []
    field_means = []
    for cls in CLASSES:
        lab_masks = [r["cam_mask"].mean() for r in gradcam_results
                     if r["class"] == cls and r["env"] == "Lab"]
        field_masks = [r["cam_mask"].mean() for r in gradcam_results
                       if r["class"] == cls and r["env"] == "Field"]
        lab_means.append(np.mean(lab_masks) if lab_masks else 0)
        field_means.append(np.mean(field_masks) if field_masks else 0)

    w = 0.3
    ax.bar(x - w/2, lab_means, w, label="Lab", color="#3498db", edgecolor="white")
    ax.bar(x + w/2, field_means, w, label="Field", color="#e67e22", edgecolor="white")
    ax.set_xlabel("Disease Class")
    ax.set_ylabel("Mean Grad-CAM Activation")
    ax.set_title("Grad-CAM Activation by Environment\n(Compares model focus in Lab vs Field)",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: SHAP statistics (if available) ──
    if shap_results:
        ax = axes[2]
        shap_lab_means = []
        shap_field_means = []
        for cls in CLASSES:
            lab_maps = [r["shap_map"].mean() for r in shap_results
                        if r["class"] == cls and r["env"] == "Lab"]
            field_maps = [r["shap_map"].mean() for r in shap_results
                          if r["class"] == cls and r["env"] == "Field"]
            shap_lab_means.append(np.mean(lab_maps) if lab_maps else 0)
            shap_field_means.append(np.mean(field_maps) if field_maps else 0)

        ax.bar(x - w/2, shap_lab_means, w, label="Lab", color="#8e44ad", edgecolor="white")
        ax.bar(x + w/2, shap_field_means, w, label="Field", color="#d35400", edgecolor="white")
        ax.set_xlabel("Disease Class")
        ax.set_ylabel("Mean SHAP Value")
        ax.set_title("SHAP Feature Importance by Environment\n(Shapley-based pixel attribution)",
                     fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, rotation=15, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "attention_statistics.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ attention_statistics.png")


def plot_individual_detailed(gradcam_results, attn_results):
    """Create detailed individual explanation panels for each class."""
    individual_dir = OUTPUT_DIR / "individual"
    individual_dir.mkdir(exist_ok=True)

    for cls in CLASSES:
        gc_cls = [r for r in gradcam_results if r["class"] == cls]
        at_cls = [r for r in attn_results if r["class"] == cls]

        if not gc_cls:
            continue

        n_samples = len(gc_cls)
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, n_samples * 3.5))
        if n_samples == 1:
            axes = axes[np.newaxis, :]

        axes[0, 0].set_title("Original", fontweight="bold", fontsize=11)
        axes[0, 1].set_title("Grad-CAM", fontweight="bold", fontsize=11)
        axes[0, 2].set_title("Attention Rollout", fontweight="bold", fontsize=11)
        axes[0, 3].set_title("Prediction Info", fontweight="bold", fontsize=11)

        for idx, gc_r in enumerate(gc_cls):
            axes[idx, 0].imshow(gc_r["raw_img"])
            axes[idx, 0].set_ylabel(f"{gc_r['env']}", fontweight="bold", fontsize=10)

            axes[idx, 1].imshow(gc_r["cam_img"])

            # Find matching attention result
            at_match = [r for r in at_cls if r["path"] == gc_r["path"]]
            if at_match:
                axes[idx, 2].imshow(at_match[0]["attn_overlay"])
            else:
                axes[idx, 2].text(0.5, 0.5, "N/A", ha="center", va="center",
                                  transform=axes[idx, 2].transAxes)

            # Prediction info
            axes[idx, 3].axis("off")
            info_text = f"True: {gc_r['class']}\n"
            info_text += f"Pred: {gc_r['pred']}\n"
            info_text += f"Conf: {gc_r['conf']:.1%}\n"
            info_text += f"Status: {'✓ Correct' if gc_r['correct'] else '✗ Wrong'}\n\n"
            info_text += "Top-3:\n"
            for name, prob in gc_r["top3"]:
                info_text += f"  {name}: {prob:.1%}\n"
            axes[idx, 3].text(0.1, 0.5, info_text, transform=axes[idx, 3].transAxes,
                              fontsize=9, va="center", fontfamily="monospace",
                              bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

            for j in range(3):
                axes[idx, j].axis("off")

        safe_cls = cls.replace(" ", "_")
        fig.suptitle(f"Detailed Explainability: {cls}", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(individual_dir / f"detail_{safe_cls}.png", bbox_inches="tight")
        plt.close(fig)

    print(f"  ✓ {len(CLASSES)} individual detail panels → individual/")


def plot_xai_method_summary(gradcam_results, shap_results, lime_results, attn_results):
    """Summary chart comparing the 4 XAI methods' spatial agreement/overlap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Mean activation intensity per method per class
    methods = ["Grad-CAM", "SHAP", "LIME", "Attention"]
    method_colors = ["#e74c3c", "#8e44ad", "#27ae60", "#3498db"]

    ax = axes[0]
    x = np.arange(len(CLASSES))
    bar_w = 0.18
    for m_idx, (method_name, results, key) in enumerate([
        ("Grad-CAM", gradcam_results, "cam_mask"),
        ("SHAP", shap_results, "shap_map"),
        ("LIME", lime_results, "lime_heatmap"),
        ("Attention", attn_results, "attn_mask"),
    ]):
        means = []
        for cls in CLASSES:
            vals = [r[key].mean() for r in results if r["class"] == cls]
            means.append(np.mean(vals) if vals else 0)
        ax.bar(x + m_idx * bar_w - 1.5 * bar_w, means, bar_w,
               label=method_name, color=method_colors[m_idx], edgecolor="white", alpha=0.85)

    ax.set_xlabel("Disease Class")
    ax.set_ylabel("Mean Activation / Importance")
    ax.set_title("Mean XAI Activation by Method & Class", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Spatial correlation between methods (how similar are the heatmaps?)
    ax = axes[1]
    from itertools import combinations
    method_pairs = list(combinations(range(4), 2))
    pair_labels = [f"{methods[a]} vs\n{methods[b]}" for a, b in method_pairs]

    all_results_list = [gradcam_results, shap_results, lime_results, attn_results]
    all_keys = ["cam_mask", "shap_map", "lime_heatmap", "attn_mask"]

    correlations = []
    for a, b in method_pairs:
        cors = []
        for cls in CLASSES:
            maps_a = [r for r in all_results_list[a] if r["class"] == cls]
            maps_b = [r for r in all_results_list[b] if r["class"] == cls]
            for ra in maps_a:
                for rb in maps_b:
                    if ra["path"] == rb["path"]:
                        ma = ra[all_keys[a]].flatten()
                        mb = rb[all_keys[b]].flatten()
                        if ma.std() > 0 and mb.std() > 0:
                            cor = np.corrcoef(ma, mb)[0, 1]
                            cors.append(cor)
        correlations.append(np.mean(cors) if cors else 0)

    pair_colors = ["#e74c3c", "#8e44ad", "#f39c12", "#3498db", "#2ecc71", "#1abc9c"]
    bars = ax.bar(range(len(correlations)), correlations, color=pair_colors[:len(correlations)],
                  edgecolor="white")
    ax.set_xlabel("Method Pair")
    ax.set_ylabel("Mean Spatial Correlation")
    ax.set_title("Spatial Agreement Between XAI Methods\n(Higher = methods agree on important regions)",
                 fontweight="bold")
    ax.set_xticks(range(len(pair_labels)))
    ax.set_xticklabels(pair_labels, rotation=0, ha="center", fontsize=8)
    for bar, val in zip(bars, correlations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "xai_method_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ xai_method_summary.png")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(" Rice Disease — Explainability (Grad-CAM + Attention + SHAP + LIME)")
    print("=" * 70)

    # Load model
    print("\n[1/11] Loading model...")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"  Loaded: epoch {ckpt.get('epoch', '?')}")

    transform = get_eval_transform()

    # Collect samples (2 per environment — balances speed vs coverage)
    print("\n[2/11] Collecting sample images...")
    samples = collect_sample_images("Test", n_per_env=2)
    total = sum(len(samples[c][e]) for c in CLASSES for e in ENVS)
    print(f"  Collected {total} sample images")
    for cls in CLASSES:
        lab_n = len(samples[cls]["Lab"])
        field_n = len(samples[cls]["Field"])
        print(f"    {cls:<20}: Lab={lab_n}, Field={field_n}")

    # Grad-CAM
    print("\n[3/11] Generating Grad-CAM heatmaps...")
    gradcam_results = generate_gradcam_heatmaps(model, device, samples, transform)
    plot_gradcam_grid(gradcam_results)

    # Save individual Grad-CAM images
    print("\n[4/11] Saving individual Grad-CAM images...")
    gradcam_dir = OUTPUT_DIR / "gradcam_individual"
    gradcam_dir.mkdir(exist_ok=True)
    for r in gradcam_results:
        safe_cls = r['class'].replace(' ', '_')
        fname = f"gradcam_{safe_cls}_{r['env']}.png"
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(r["cam_img"])
        ax.axis("off")
        status = "✓" if r["correct"] else "✗"
        ax.set_title(f"{r['class']} ({r['env']})\n{status} Pred: {r['pred']} ({r['conf']:.1%})",
                     fontsize=10)
        plt.tight_layout()
        fig.savefig(gradcam_dir / fname, bbox_inches="tight")
        plt.close(fig)
    print(f"  ✓ {len(gradcam_results)} individual Grad-CAM images → gradcam_individual/")

    # Attention Rollout
    print("\n[5/11] Generating attention rollout maps...")
    attn_results = generate_attention_maps(model, device, samples, transform)
    plot_attention_grid(attn_results)

    # SHAP
    print("\n[6/11] Generating SHAP explanations...")
    print("  Building background dataset for SHAP...")
    background_tensors = build_shap_background(transform, n=50)
    if background_tensors is not None:
        background_tensors = background_tensors.to(device)
        print(f"  Background: {background_tensors.shape[0]} images")
        shap_results = generate_shap_explanations(model, device, samples, transform, background_tensors)
        plot_shap_grid(shap_results)
    else:
        print("  ⚠ Could not build SHAP background — skipping SHAP")
        shap_results = []

    # LIME
    print("\n[7/11] Generating LIME explanations...")
    lime_results = generate_lime_explanations(model, device, samples, transform)
    plot_lime_grid(lime_results)

    # Comparison panels
    print("\n[8/11] Creating XAI comparison panels...")
    plot_comparison_panels(gradcam_results, attn_results)

    # 4-method comparison
    print("\n[9/11] Creating 4-method XAI comparison...")
    if shap_results and lime_results:
        plot_4method_comparison(gradcam_results, attn_results, shap_results, lime_results)
        plot_xai_method_summary(gradcam_results, shap_results, lime_results, attn_results)
    else:
        print("  ⚠ Skipping 4-method comparison (missing SHAP or LIME results)")

    # Individual detailed panels
    print("\n[10/11] Creating individual detail panels...")
    plot_individual_detailed(gradcam_results, attn_results)

    # Attention + SHAP statistics
    plot_attention_statistics(gradcam_results, shap_results if shap_results else None)

    # Misclassified
    print("\n[11/11] Explaining misclassified samples...")
    generate_misclassified_explanations(model, device, transform)

    # ── Save XAI summary JSON ──
    xai_summary = {
        "methods_used": ["Grad-CAM", "Attention Rollout", "SHAP (GradientExplainer)", "LIME"],
        "samples_per_class_per_env": 2,
        "total_samples_explained": total,
        "shap_background_size": background_tensors.shape[0] if background_tensors is not None else 0,
        "lime_num_perturbations": 300,
        "per_class_correct": {},
    }
    for cls in CLASSES:
        gc_cls = [r for r in gradcam_results if r["class"] == cls]
        xai_summary["per_class_correct"][cls] = {
            "correct": sum(1 for r in gc_cls if r["correct"]),
            "total": len(gc_cls),
        }
    with open(OUTPUT_DIR / "xai_summary.json", "w") as f:
        json.dump(xai_summary, f, indent=2)
    print("  ✓ xai_summary.json")

    print(f"\n{'='*70}")
    print(f" ✅ Explainability complete! → {OUTPUT_DIR}/")
    print(f"    Outputs:")
    print(f"      gradcam_grid_all_classes.png     — All classes Grad-CAM grid")
    print(f"      attention_rollout_grid.png        — All classes attention rollout")
    print(f"      shap_grid_all_classes.png         — All classes SHAP heatmaps")
    print(f"      lime_grid_all_classes.png         — All classes LIME explanations")
    print(f"      xai_4method_comparison.png        — 4-method side-by-side comparison")
    print(f"      xai_comparison_grid.png           — Grad-CAM vs Attention comparison")
    print(f"      xai_method_summary.png            — Method agreement statistics")
    print(f"      misclassified_explanations.png    — Error analysis with dual CAMs")
    print(f"      attention_statistics.png           — Activation pattern analysis")
    print(f"      gradcam_individual/               — Individual heatmaps per image")
    print(f"      individual/                       — Detailed panels per class")
    print(f"      misclassified_summary.json         — Error breakdown")
    print(f"      xai_summary.json                   — XAI run metadata")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
