"""
Evaluate and compare all trained models.

Reports: mAP@50, mAP@50:95, per-class AP, precision, recall.
Generates: comparison table, per-class bar chart, confusion analysis.

Usage:
    python evaluate.py                              # auto-discover all models
    python evaluate.py --models yolo11x_full rtdetrv2_full
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────
PROJECT = os.path.expanduser("~/PPE")
DATASET_YAML = os.path.join(PROJECT, "sh17.yaml")
RUNS_DIR = os.path.join(PROJECT, "runs")
EVAL_DIR = os.path.join(PROJECT, "evaluation")

CLASS_NAMES = [
    "person", "ear", "ear-mufs", "face", "face-guard",
    "face-mask-medical", "foot", "tools", "glasses", "gloves",
    "helmet", "hands", "head", "medical-suit", "shoes",
    "safety-suit", "safety-vest",
]

# reference baseline from paper
BASELINE = {"model": "YOLOv9-e (paper)", "mAP50": 70.9, "mAP50_95": 48.7}


# ── Model discovery ─────────────────────────────────────────

def discover_models():
    """Find all trained models in runs/."""
    models = []
    if not os.path.isdir(RUNS_DIR):
        return models

    for name in sorted(os.listdir(RUNS_DIR)):
        run_dir = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(run_dir):
            continue

        # ultralytics models (YOLO / RT-DETR)
        best_pt = os.path.join(run_dir, "weights", "best.pt")
        if os.path.isfile(best_pt):
            model_type = "rtdetr" if "rtdetr" in name.lower() else "yolo"
            models.append({"name": name, "type": model_type, "weights": best_pt})
            continue

        # mmdetection models (DINO) — look for best_*.pth
        for f in sorted(os.listdir(run_dir)):
            if f.startswith("best_") and f.endswith(".pth"):
                configs = [c for c in os.listdir(run_dir) if c.endswith(".py")]
                cfg_path = os.path.join(run_dir, configs[0]) if configs else None
                models.append({
                    "name": name, "type": "dino",
                    "weights": os.path.join(run_dir, f),
                    "config": cfg_path,
                })
                break

    return models


# ── Evaluation ──────────────────────────────────────────────

def eval_ultralytics(name, weights, model_type):
    """Evaluate a YOLO or RT-DETR model."""
    print(f"\n  Evaluating {name}...")

    if model_type == "rtdetr":
        from ultralytics import RTDETR
        model = RTDETR(weights)
    else:
        from ultralytics import YOLO
        model = YOLO(weights)

    metrics = model.val(
        data=DATASET_YAML,
        imgsz=1280,
        batch=8,
        device="0",
        plots=True,
        save_json=True,
    )

    result = {
        "model": name,
        "mAP50": round(float(metrics.box.map50) * 100, 2),
        "mAP50_95": round(float(metrics.box.map) * 100, 2),
        "precision": round(float(metrics.box.mp) * 100, 2),
        "recall": round(float(metrics.box.mr) * 100, 2),
        "per_class_ap50": {},
        "per_class_ap50_95": {},
    }

    # per-class
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        for i, ap in enumerate(metrics.box.ap50):
            if i < len(CLASS_NAMES):
                result["per_class_ap50"][CLASS_NAMES[i]] = round(float(ap) * 100, 2)

    if hasattr(metrics.box, "ap") and metrics.box.ap is not None:
        for i, ap in enumerate(metrics.box.ap):
            if i < len(CLASS_NAMES):
                result["per_class_ap50_95"][CLASS_NAMES[i]] = round(float(ap) * 100, 2)

    return result


def eval_dino(name, weights, config):
    """Evaluate a DINO model via mmdetection."""
    print(f"\n  Evaluating {name}...")

    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
    except ImportError:
        print(f"    mmdetection not installed, skipping {name}")
        return None

    if config is None:
        print(f"    No config found for {name}, skipping")
        return None

    cfg = Config.fromfile(config)
    cfg.work_dir = os.path.join(EVAL_DIR, name)
    cfg.load_from = weights

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    result = {
        "model": name,
        "mAP50": round(metrics.get("coco/bbox_mAP_50", 0) * 100, 2),
        "mAP50_95": round(metrics.get("coco/bbox_mAP", 0) * 100, 2),
        "precision": 0,  # mmdet doesn't directly report P/R the same way
        "recall": 0,
        "per_class_ap50": {},
        "per_class_ap50_95": {},
    }

    return result


# ── Visualization ───────────────────────────────────────────

def print_comparison(results):
    """Print a clean comparison table."""
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*70}")

    # include baseline
    all_rows = [BASELINE] + results

    df = pd.DataFrame(all_rows)
    cols = ["model", "mAP50", "mAP50_95"]
    if "precision" in df.columns:
        cols += ["precision", "recall"]
    df = df[cols]

    print(df.to_string(index=False))
    print()

    # highlight best
    best_50 = max(results, key=lambda x: x["mAP50"])
    best_95 = max(results, key=lambda x: x["mAP50_95"])
    print(f"  Best mAP@50:    {best_50['model']} ({best_50['mAP50']})")
    print(f"  Best mAP@50:95: {best_95['model']} ({best_95['mAP50_95']})")

    # vs baseline
    for r in results:
        diff50 = r["mAP50"] - BASELINE["mAP50"]
        diff95 = r["mAP50_95"] - BASELINE["mAP50_95"]
        sign50 = "+" if diff50 >= 0 else ""
        sign95 = "+" if diff95 >= 0 else ""
        print(f"  {r['model']} vs baseline: {sign50}{diff50:.1f} mAP@50, {sign95}{diff95:.1f} mAP@50:95")

    # save csv
    csv_path = os.path.join(EVAL_DIR, "comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    return df


def plot_per_class(results):
    """Bar chart comparing per-class AP across models."""
    # collect data
    data = {}
    for r in results:
        if r["per_class_ap50"]:
            data[r["model"]] = r["per_class_ap50"]

    if len(data) < 1:
        return

    df = pd.DataFrame(data)
    df = df.reindex(CLASS_NAMES)  # ensure order

    fig, ax = plt.subplots(figsize=(16, 8))
    df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("AP@50 (%)")
    ax.set_title("Per-Class AP@50 Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    # highlight minority classes
    minority = {"face-guard", "face-mask-medical", "medical-suit", "safety-suit", "ear-mufs", "tools"}
    for i, label in enumerate(ax.get_xticklabels()):
        if label.get_text() in minority:
            label.set_color("red")
            label.set_fontweight("bold")

    plt.tight_layout()
    path = os.path.join(EVAL_DIR, "per_class_ap50.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_bars(results):
    """Simple bar chart of mAP@50 and mAP@50:95 for all models."""
    all_rows = [BASELINE] + results

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_rows))
    w = 0.35

    map50 = [r["mAP50"] for r in all_rows]
    map95 = [r["mAP50_95"] for r in all_rows]
    names = [r["model"] for r in all_rows]

    bars1 = ax.bar(x - w/2, map50, w, label="mAP@50", color="#3498db")
    bars2 = ax.bar(x + w/2, map95, w, label="mAP@50:95", color="#e74c3c")

    ax.set_ylabel("mAP (%)")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(EVAL_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def analyze_hard_classes(results):
    """Find which classes each model struggles with most."""
    print(f"\n{'='*70}")
    print(f"  HARD CLASS ANALYSIS")
    print(f"{'='*70}")

    for r in results:
        ap = r.get("per_class_ap50", {})
        if not ap:
            continue

        sorted_cls = sorted(ap.items(), key=lambda x: x[1])
        print(f"\n  {r['model']}:")
        print(f"    Hardest 5 classes:")
        for name, val in sorted_cls[:5]:
            print(f"      {name:<25s} {val:>6.1f}%")
        print(f"    Easiest 3 classes:")
        for name, val in sorted_cls[-3:]:
            print(f"      {name:<25s} {val:>6.1f}%")


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific model names to evaluate (default: all)")
    args = parser.parse_args()

    os.makedirs(EVAL_DIR, exist_ok=True)

    # discover
    discovered = discover_models()
    if not discovered:
        print(f"No trained models found in {RUNS_DIR}")
        print(f"Train a model first, then run this script.")
        return

    print(f"Found {len(discovered)} model(s):")
    for m in discovered:
        print(f"  • {m['name']} ({m['type']})")

    # filter
    if args.models:
        discovered = [m for m in discovered if m["name"] in args.models]

    # evaluate each
    results = []
    for m in discovered:
        try:
            if m["type"] in ("yolo", "rtdetr"):
                r = eval_ultralytics(m["name"], m["weights"], m["type"])
            elif m["type"] == "dino":
                r = eval_dino(m["name"], m["weights"], m.get("config"))
            else:
                continue
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ERROR evaluating {m['name']}: {e}")

    if not results:
        print("No models evaluated successfully.")
        return

    # report
    print_comparison(results)
    plot_summary_bars(results)
    plot_per_class(results)
    analyze_hard_classes(results)

    # save everything
    json_path = os.path.join(EVAL_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results: {json_path}")
    print(f"  All outputs:  {EVAL_DIR}/")


if __name__ == "__main__":
    main()