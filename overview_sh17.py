"""
SH17 Dataset Overview
=====================
Quick summary for slides / presentation.
Prints a clean, compact overview of the dataset.

Usage:
    python overview_sh17.py
"""

import os
import glob
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

# ── Config ──────────────────────────────────────────────────
DATASET_ROOT = os.path.expanduser("~/PPE/sh17dataset")
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
TRAIN_TXT = os.path.join(DATASET_ROOT, "train_files.txt")
VAL_TXT = os.path.join(DATASET_ROOT, "val_files.txt")
META_DIR = os.path.join(DATASET_ROOT, "meta-data")


def read_lines(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def find_image(name, img_dir):
    full = os.path.join(img_dir, name)
    if os.path.isfile(full):
        return full
    stem = os.path.splitext(name)[0]
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"]:
        p = os.path.join(img_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def parse_label(path):
    boxes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    boxes.append((int(parts[0]), *map(float, parts[1:5])))
                except ValueError:
                    pass
    return boxes


def try_load_class_names():
    """Try to read class names from meta-data."""
    if not os.path.isdir(META_DIR):
        return {}
    for fname in os.listdir(META_DIR):
        fpath = os.path.join(META_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.endswith((".txt", ".names", ".csv")):
            continue
        with open(fpath) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) >= 10:  # likely a class list
            names = {}
            for i, line in enumerate(lines):
                if ":" in line:
                    k, v = line.split(":", 1)
                    names[int(k.strip())] = v.strip()
                elif "," in line:
                    parts = line.split(",", 1)
                    try:
                        names[int(parts[0].strip())] = parts[1].strip()
                    except ValueError:
                        names[i] = line
                else:
                    names[i] = line
            return names
    return {}


def main():
    # resolve splits
    train_names = read_lines(TRAIN_TXT)
    val_names = read_lines(VAL_TXT)

    train_imgs = [p for n in train_names if (p := find_image(n, IMAGES_DIR))]
    val_imgs = [p for n in val_names if (p := find_image(n, IMAGES_DIR))]
    all_imgs = train_imgs + val_imgs

    # count instances + classes
    class_counts = Counter()
    total_instances = 0
    objs_per_img = []

    for img_path in all_imgs:
        stem = Path(img_path).stem
        lbl = os.path.join(LABELS_DIR, stem + ".txt")
        if not os.path.isfile(lbl):
            objs_per_img.append(0)
            continue
        boxes = parse_label(lbl)
        objs_per_img.append(len(boxes))
        total_instances += len(boxes)
        for cls_id, *_ in boxes:
            class_counts[cls_id] += 1

    # class names
    class_names = try_load_class_names()
    num_classes = len(class_counts)

    # sample image dimensions
    sample_imgs = all_imgs[:200] if len(all_imgs) > 200 else all_imgs
    dims = []
    for p in sample_imgs:
        try:
            w, h = Image.open(p).size
            dims.append((w, h))
        except:
            pass

    # imbalance
    sorted_cls = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    max_count = sorted_cls[0][1]
    min_count = sorted_cls[-1][1]
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    # ── Print ───────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              SH17 DATASET — OVERVIEW                   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    print(f"""
  Images
    Total:          {len(all_imgs)}
    Train:          {len(train_imgs)} ({100*len(train_imgs)/len(all_imgs):.1f}%)
    Val:            {len(val_imgs)} ({100*len(val_imgs)/len(all_imgs):.1f}%)

  Annotations
    Total instances: {total_instances}
    Classes:         {num_classes}
    Avg obj/image:   {np.mean(objs_per_img):.1f}
    Max obj/image:   {max(objs_per_img)}""")

    if dims:
        widths = [d[0] for d in dims]
        heights = [d[1] for d in dims]
        common_res = Counter(dims).most_common(1)[0]
        print(f"""
  Image Size
    Most common:    {common_res[0][0]} x {common_res[0][1]} ({common_res[1]} images)
    Width range:    {min(widths)} - {max(widths)}
    Height range:   {min(heights)} - {max(heights)}""")

    print(f"""
  Class Imbalance
    Ratio (max/min): {imbalance_ratio:.0f}x
    Largest class:   {class_names.get(sorted_cls[0][0], f'class_{sorted_cls[0][0]}')} ({max_count} instances)
    Smallest class:  {class_names.get(sorted_cls[-1][0], f'class_{sorted_cls[-1][0]}')} ({min_count} instances)
""")

    # class table
    print("  Classes")
    print(f"  {'#':>4s}  {'Name':<25s}  {'Count':>7s}  {'%':>6s}")
    print(f"  {'─'*4}  {'─'*25}  {'─'*7}  {'─'*6}")
    for cls_id, count in sorted_cls:
        name = class_names.get(cls_id, f"class_{cls_id}")
        pct = 100 * count / total_instances
        print(f"  {cls_id:>4d}  {name:<25s}  {count:>7d}  {pct:>5.1f}%")

    print(f"\n  Path: {DATASET_ROOT}")
    print()


if __name__ == "__main__":
    main()