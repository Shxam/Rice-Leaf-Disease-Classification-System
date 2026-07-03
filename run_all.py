"""
run_all.py — Master Pipeline Runner
=====================================
Runs all stages sequentially: explore → train → evaluate → explain
"""

import subprocess
import sys
import time
from pathlib import Path

PYTHON = sys.executable
SCRIPTS = [
    ("Dataset Exploration", "explore_dataset.py"),
    ("ViT Training", "train.py"),
    ("Evaluation", "evaluate.py"),
    ("Explainability", "explain.py"),
]


def run_stage(name, script):
    print(f"\n{'='*70}")
    print(f" STAGE: {name}")
    print(f" Script: {script}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, script],
        cwd=str(Path(__file__).parent),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n ❌ {name} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        sys.exit(1)
    else:
        print(f"\n ✅ {name} completed in {elapsed:.1f}s")

    return elapsed


def main():
    print("=" * 70)
    print(" RICE PLANT DISEASE CLASSIFICATION — FULL PIPELINE")
    print(" Model: ViT-Base/16 (vit_base_patch16_224)")
    print(" Focus: Explainable AI + Class Imbalance Handling")
    print("=" * 70)

    total_start = time.time()
    timings = {}

    for name, script in SCRIPTS:
        elapsed = run_stage(name, script)
        timings[name] = elapsed

    total_time = time.time() - total_start

    print(f"\n{'='*70}")
    print(" PIPELINE COMPLETE — TIMING SUMMARY")
    print(f"{'='*70}")
    for name, t in timings.items():
        print(f"  {name:<25}: {t:>8.1f}s ({t/60:.1f}m)")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<25}: {total_time:>8.1f}s ({total_time/60:.1f}m)")
    print(f"\n  All outputs saved to: outputs/")
    print(f"    ├── exploration/    — Dataset analysis plots")
    print(f"    ├── training/       — Model checkpoint, curves, configs")
    print(f"    ├── evaluation/     — Metrics, confusion matrices, ROC/PR curves")
    print(f"    └── explainability/ — Grad-CAM, attention rollout, XAI comparison")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
