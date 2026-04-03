"""
SH17 Dataset Analysis
=====================
Run this to get a full picture of the dataset before training.
Covers: splits, class distribution, object sizes, image stats, label issues, etc.

Usage:
    python analyze_sh17.py
"""

import os
import glob
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # headless server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────────
DATASET_ROOT = os.path.expanduser("~/PPE/sh17dataset")
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
TRAIN_TXT = os.path.join(DATASET_ROOT, "train_files.txt")
VAL_TXT = os.path.join(DATASET_ROOT, "val_files.txt")
META_DIR = os.path.join(DATASET_ROOT, "meta-data")
VOC_DIR = os.path.join(DATASET_ROOT, "voc_labels")

OUTPUT_DIR = os.path.expanduser("~/PPE/dataset_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────

def read_filelist(txt_path):
    """Read a file list (.txt) and return stripped lines."""
    with open(txt_path) as f:
        return [l.strip() for l in f if l.strip()]


def find_image(name, img_dir):
    """
    Given a filename (might or might not have extension),
    find the actual image file. Returns full path or None.
    """
    # if it already has an extension and exists
    full = os.path.join(img_dir, name)
    if os.path.isfile(full):
        return full

    # try common extensions
    stem = os.path.splitext(name)[0]
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"]:
        candidate = os.path.join(img_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def find_label(image_path, lbl_dir):
    """Find the YOLO label file matching an image."""
    stem = Path(image_path).stem
    lbl = os.path.join(lbl_dir, stem + ".txt")
    return lbl if os.path.isfile(lbl) else None


def parse_yolo_label(label_path):
    """
    Parse a YOLO-format label file.
    Each line: class_id  x_center  y_center  width  height  (all normalized 0-1)
    Returns list of tuples: (cls_id, xc, yc, w, h)
    """
    boxes = []
    with open(label_path) as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                # malformed line, skip but note it
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append((cls_id, xc, yc, w, h))
            except ValueError:
                continue
    return boxes


def coco_size_category(abs_w, abs_h):
    """Classify object size by COCO convention: small / medium / large."""
    area = abs_w * abs_h
    if area < 32**2:
        return "small"
    elif area < 96**2:
        return "medium"
    else:
        return "large"


def print_header(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def print_bar(label, count, max_count, bar_width=40):
    """Print a simple text-based bar chart row."""
    filled = int(bar_width * count / max_count) if max_count > 0 else 0
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"  {label:>22s} │ {bar} │ {count:>6d}")


# ── Step 1: Basic file structure ────────────────────────────

def step1_file_structure():
    print_header("1. FILE STRUCTURE")

    dirs = ["images", "labels", "meta-data", "voc_labels"]
    for d in dirs:
        full = os.path.join(DATASET_ROOT, d)
        if os.path.isdir(full):
            n = len(os.listdir(full))
            print(f"  ✓ {d:15s}  →  {n} files")
        else:
            print(f"  ✗ {d:15s}  →  NOT FOUND")

    for f in ["train_files.txt", "val_files.txt"]:
        full = os.path.join(DATASET_ROOT, f)
        if os.path.isfile(full):
            n = len(read_filelist(full))
            print(f"  ✓ {f:15s}  →  {n} entries")
        else:
            print(f"  ✗ {f:15s}  →  NOT FOUND")

    # peek at meta-data
    if os.path.isdir(META_DIR):
        meta_files = os.listdir(META_DIR)
        print(f"\n  meta-data contents: {meta_files[:10]}")
        for mf in meta_files:
            mf_path = os.path.join(META_DIR, mf)
            if os.path.isfile(mf_path) and os.path.getsize(mf_path) < 50000:
                print(f"\n  --- {mf} (first 30 lines) ---")
                with open(mf_path) as f:
                    for i, line in enumerate(f):
                        if i >= 30:
                            print(f"  ... (truncated)")
                            break
                        print(f"  {line.rstrip()}")

    # peek at train_files.txt format
    print(f"\n  train_files.txt — first 5 lines:")
    train_lines = read_filelist(TRAIN_TXT)
    for line in train_lines[:5]:
        print(f"    '{line}'")

    print(f"\n  val_files.txt — first 5 lines:")
    val_lines = read_filelist(VAL_TXT)
    for line in val_lines[:5]:
        print(f"    '{line}'")

    # peek at a label file
    all_labels = sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt")))
    if all_labels:
        sample = all_labels[0]
        print(f"\n  sample label ({os.path.basename(sample)}) — first 5 lines:")
        with open(sample) as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"    {line.rstrip()}")


# ── Step 2: Resolve splits ──────────────────────────────────

def step2_resolve_splits():
    print_header("2. TRAIN / VAL SPLITS")

    train_names = read_filelist(TRAIN_TXT)
    val_names = read_filelist(VAL_TXT)

    # resolve actual image paths
    train_imgs, train_missing = [], []
    for name in train_names:
        img = find_image(name, IMAGES_DIR)
        if img:
            train_imgs.append(img)
        else:
            train_missing.append(name)

    val_imgs, val_missing = [], []
    for name in val_names:
        img = find_image(name, IMAGES_DIR)
        if img:
            val_imgs.append(img)
        else:
            val_missing.append(name)

    print(f"  Train:  {len(train_imgs)} images found, {len(train_missing)} missing")
    print(f"  Val:    {len(val_imgs)} images found, {len(val_missing)} missing")
    print(f"  Total:  {len(train_imgs) + len(val_imgs)}")
    print(f"  Ratio:  {len(train_imgs)/(len(train_imgs)+len(val_imgs))*100:.1f}% train / "
          f"{len(val_imgs)/(len(train_imgs)+len(val_imgs))*100:.1f}% val")

    if train_missing:
        print(f"\n  ⚠ Missing train images (first 5): {train_missing[:5]}")
    if val_missing:
        print(f"  ⚠ Missing val images (first 5): {val_missing[:5]}")

    # check overlap
    train_set = set(os.path.basename(p) for p in train_imgs)
    val_set = set(os.path.basename(p) for p in val_imgs)
    overlap = train_set & val_set
    if overlap:
        print(f"  ⚠ WARNING: {len(overlap)} files appear in BOTH train and val!")
        print(f"    Examples: {list(overlap)[:5]}")
    else:
        print(f"  ✓ No overlap between train and val — good.")

    # check images without labels
    train_no_label = [p for p in train_imgs if find_label(p, LABELS_DIR) is None]
    val_no_label = [p for p in val_imgs if find_label(p, LABELS_DIR) is None]
    print(f"\n  Images with no label file:")
    print(f"    Train: {len(train_no_label)}")
    print(f"    Val:   {len(val_no_label)}")

    # also check: are there images in the folder NOT in either split?
    all_images_on_disk = set(os.listdir(IMAGES_DIR))
    referenced = set(os.path.basename(p) for p in train_imgs + val_imgs)
    orphans = all_images_on_disk - referenced
    if orphans:
        print(f"\n  ⚠ {len(orphans)} images on disk are NOT in train or val txt files.")
        print(f"    Examples: {list(orphans)[:5]}")
    else:
        print(f"  ✓ All images on disk are referenced in train/val splits.")

    return train_imgs, val_imgs


# ── Step 3: Deep label analysis ─────────────────────────────

def step3_label_analysis(train_imgs, val_imgs):
    print_header("3. LABEL ANALYSIS")

    all_data = {}  # split_name -> list of per-image info

    for split_name, img_list in [("train", train_imgs), ("val", val_imgs)]:
        records = []
        for img_path in img_list:
            lbl_path = find_label(img_path, LABELS_DIR)
            boxes = parse_yolo_label(lbl_path) if lbl_path else []
            records.append({
                "image": img_path,
                "label": lbl_path,
                "boxes": boxes,
                "n_objects": len(boxes),
            })
        all_data[split_name] = records

    # overall counts
    for split_name, records in all_data.items():
        total_obj = sum(r["n_objects"] for r in records)
        empty = sum(1 for r in records if r["n_objects"] == 0)
        print(f"\n  [{split_name.upper()}]")
        print(f"    Images:     {len(records)}")
        print(f"    Instances:  {total_obj}")
        print(f"    Empty:      {empty} images with 0 objects")
        if records:
            objs = [r["n_objects"] for r in records]
            print(f"    Objects/img: min={min(objs)}, max={max(objs)}, "
                  f"mean={np.mean(objs):.1f}, median={np.median(objs):.0f}")

    # check label value ranges (catch bad labels)
    print(f"\n  Checking label value ranges...")
    issues = {"out_of_range": 0, "negative_size": 0, "huge_class_id": 0}
    all_class_ids = set()

    for split_name, records in all_data.items():
        for r in records:
            for cls_id, xc, yc, w, h in r["boxes"]:
                all_class_ids.add(cls_id)
                if not (0 <= xc <= 1 and 0 <= yc <= 1):
                    issues["out_of_range"] += 1
                if w <= 0 or h <= 0:
                    issues["negative_size"] += 1
                if w > 1.0 or h > 1.0:
                    issues["out_of_range"] += 1
                if cls_id < 0 or cls_id > 50:
                    issues["huge_class_id"] += 1

    print(f"    Class IDs found: {sorted(all_class_ids)}")
    print(f"    Num unique classes: {len(all_class_ids)}")
    print(f"    Coords out of [0,1]: {issues['out_of_range']}")
    print(f"    Negative w/h: {issues['negative_size']}")
    print(f"    Suspicious class IDs: {issues['huge_class_id']}")

    if all(v == 0 for v in issues.values()):
        print(f"    ✓ All labels look clean.")

    return all_data, sorted(all_class_ids)


# ── Step 4: Class distribution ──────────────────────────────

def step4_class_distribution(all_data, class_ids):
    print_header("4. CLASS DISTRIBUTION")

    # try to load class names from meta-data
    class_names = _try_load_class_names(class_ids)

    for split_name, records in all_data.items():
        counts = Counter()
        img_counts = defaultdict(set)  # class -> set of image paths
        for r in records:
            for cls_id, *_ in r["boxes"]:
                counts[cls_id] += 1
                img_counts[cls_id].add(r["image"])

        total = sum(counts.values())
        sorted_cls = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        max_count = sorted_cls[0][1] if sorted_cls else 1
        min_count = sorted_cls[-1][1] if sorted_cls else 1

        print(f"\n  [{split_name.upper()}] — {total} total instances")
        print(f"  {'Class':>22s} │ {'Distribution':^42s} │ {'Count':>6s} │ {'%':>5s} │ {'Imgs':>5s}")
        print(f"  {'─'*22}─┼─{'─'*42}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*5}")

        for cls_id, count in sorted_cls:
            name = class_names.get(cls_id, f"class_{cls_id}")
            pct = 100.0 * count / total
            n_imgs = len(img_counts[cls_id])
            filled = int(40 * count / max_count)
            bar = "█" * filled + "░" * (40 - filled)
            print(f"  {name:>22s} │ {bar} │ {count:>6d} │ {pct:>4.1f}% │ {n_imgs:>5d}")

        # imbalance stats
        ratio = max_count / min_count if min_count > 0 else float("inf")
        print(f"\n  Imbalance ratio (max/min): {ratio:.1f}x")
        print(f"  Most common:  {class_names.get(sorted_cls[0][0], sorted_cls[0][0])} ({sorted_cls[0][1]})")
        print(f"  Least common: {class_names.get(sorted_cls[-1][0], sorted_cls[-1][0])} ({sorted_cls[-1][1]})")

    # plot
    _plot_class_distribution(all_data, class_names)

    return class_names


def _try_load_class_names(class_ids):
    """
    Try to figure out class names from meta-data folder.
    If we can't, just use 'class_0', 'class_1', etc.
    """
    names = {}

    # check common files in meta-data
    candidates = [
        os.path.join(META_DIR, "classes.txt"),
        os.path.join(META_DIR, "classes.names"),
        os.path.join(META_DIR, "obj.names"),
        os.path.join(META_DIR, "labels.txt"),
    ]

    # also try any .txt or .names file in meta-data
    if os.path.isdir(META_DIR):
        for f in os.listdir(META_DIR):
            fp = os.path.join(META_DIR, f)
            if f.endswith((".txt", ".names", ".csv")) and os.path.isfile(fp):
                if fp not in candidates:
                    candidates.append(fp)

    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        # heuristic: if line count roughly matches class count, it's probably class names
        if len(lines) >= len(class_ids):
            print(f"  Found class names in: {os.path.basename(path)}")
            for i, line in enumerate(lines):
                # handle "id: name" or "id,name" or just "name"
                if ":" in line:
                    parts = line.split(":", 1)
                    names[int(parts[0].strip())] = parts[1].strip()
                elif "," in line:
                    parts = line.split(",", 1)
                    try:
                        names[int(parts[0].strip())] = parts[1].strip()
                    except ValueError:
                        names[i] = line
                else:
                    names[i] = line
            break

    # fallback
    if not names:
        print(f"  Could not find class names file in meta-data, using generic names.")
        for cid in class_ids:
            names[cid] = f"class_{cid}"

    return names


def _plot_class_distribution(all_data, class_names):
    """Bar chart of class counts, train vs val side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (split_name, records) in zip(axes, all_data.items()):
        counts = Counter()
        for r in records:
            for cls_id, *_ in r["boxes"]:
                counts[cls_id] += 1

        sorted_cls = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        labels = [class_names.get(c, f"cls_{c}") for c, _ in sorted_cls]
        values = [v for _, v in sorted_cls]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
        bars = ax.barh(range(len(labels)), values, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Instance count")
        ax.set_title(f"{split_name.upper()} — Class Distribution", fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # add count labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{val}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Saved: {path}")


# ── Step 5: Object size analysis ────────────────────────────

def step5_object_sizes(all_data, class_names):
    print_header("5. OBJECT SIZE ANALYSIS")

    # we need actual image dimensions to convert normalized coords to pixels
    # sample a subset if dataset is huge
    size_data = {"small": 0, "medium": 0, "large": 0}
    size_per_class = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})
    all_areas = []
    all_wh = []  # (abs_w, abs_h) for aspect ratio analysis
    img_dims = []

    # combine train + val
    all_records = []
    for records in all_data.values():
        all_records.extend(records)

    # sample up to 2000 images for speed (reading image dims is slow)
    sample = all_records if len(all_records) <= 2000 else random.sample(all_records, 2000)
    print(f"  Analyzing {len(sample)} images (sampled from {len(all_records)})...")

    for r in sample:
        if r["n_objects"] == 0:
            continue
        try:
            img = Image.open(r["image"])
            img_w, img_h = img.size
            img_dims.append((img_w, img_h))
        except Exception:
            continue

        for cls_id, xc, yc, w, h in r["boxes"]:
            abs_w = w * img_w
            abs_h = h * img_h
            area = abs_w * abs_h
            cat = coco_size_category(abs_w, abs_h)

            size_data[cat] += 1
            size_per_class[cls_id][cat] += 1
            all_areas.append(area)
            all_wh.append((abs_w, abs_h))

    total = sum(size_data.values())
    print(f"\n  Size distribution (COCO criteria):")
    print(f"    Small  (area < 32²):   {size_data['small']:>6d}  ({100*size_data['small']/total:.1f}%)")
    print(f"    Medium (32² - 96²):    {size_data['medium']:>6d}  ({100*size_data['medium']/total:.1f}%)")
    print(f"    Large  (area > 96²):   {size_data['large']:>6d}  ({100*size_data['large']/total:.1f}%)")

    if all_areas:
        areas = np.array(all_areas)
        print(f"\n  Object area stats (pixels²):")
        print(f"    Min:    {areas.min():.0f}")
        print(f"    Max:    {areas.max():.0f}")
        print(f"    Mean:   {areas.mean():.0f}")
        print(f"    Median: {np.median(areas):.0f}")
        print(f"    Std:    {areas.std():.0f}")

    # per-class size breakdown
    print(f"\n  Per-class size breakdown:")
    print(f"  {'Class':>22s} │ {'Small':>7s} │ {'Medium':>7s} │ {'Large':>7s} │ {'%Small':>7s}")
    print(f"  {'─'*22}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")

    for cls_id in sorted(size_per_class.keys()):
        s = size_per_class[cls_id]
        cls_total = s["small"] + s["medium"] + s["large"]
        pct_small = 100 * s["small"] / cls_total if cls_total > 0 else 0
        name = class_names.get(cls_id, f"class_{cls_id}")
        print(f"  {name:>22s} │ {s['small']:>7d} │ {s['medium']:>7d} │ {s['large']:>7d} │ {pct_small:>6.1f}%")

    # image dimensions
    if img_dims:
        widths = [d[0] for d in img_dims]
        heights = [d[1] for d in img_dims]
        print(f"\n  Image dimensions:")
        print(f"    Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
        print(f"    Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
        # unique resolutions
        unique_res = Counter(img_dims)
        print(f"    Unique resolutions: {len(unique_res)}")
        for res, cnt in unique_res.most_common(5):
            print(f"      {res[0]}x{res[1]}: {cnt} images")

    # plots
    _plot_size_analysis(all_areas, all_wh, size_per_class, class_names)

    return size_data, size_per_class


def _plot_size_analysis(all_areas, all_wh, size_per_class, class_names):
    """Generate size-related plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1) area histogram
    ax = axes[0, 0]
    if all_areas:
        areas = np.array(all_areas)
        # clip for visualization
        clipped = np.clip(areas, 0, np.percentile(areas, 99))
        ax.hist(clipped, bins=80, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.axvline(32**2, color="red", linestyle="--", alpha=0.7, label="Small threshold (32²)")
        ax.axvline(96**2, color="orange", linestyle="--", alpha=0.7, label="Medium threshold (96²)")
        ax.legend(fontsize=9)
    ax.set_xlabel("Object area (pixels²)")
    ax.set_ylabel("Count")
    ax.set_title("Object Area Distribution", fontweight="bold")

    # 2) width vs height scatter
    ax = axes[0, 1]
    if all_wh:
        wh = np.array(all_wh)
        # subsample if too many points
        if len(wh) > 5000:
            idx = np.random.choice(len(wh), 5000, replace=False)
            wh_plot = wh[idx]
        else:
            wh_plot = wh
        ax.scatter(wh_plot[:, 0], wh_plot[:, 1], alpha=0.15, s=5, c="#4C72B0")
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.plot([0, max(wh_plot[:, 0])], [0, max(wh_plot[:, 0])], "r--", alpha=0.3, label="1:1")
        ax.legend()
    ax.set_title("Object Width vs Height", fontweight="bold")

    # 3) stacked bar: size per class
    ax = axes[1, 0]
    sorted_cls = sorted(size_per_class.keys())
    labels = [class_names.get(c, f"cls_{c}") for c in sorted_cls]
    smalls = [size_per_class[c]["small"] for c in sorted_cls]
    mediums = [size_per_class[c]["medium"] for c in sorted_cls]
    larges = [size_per_class[c]["large"] for c in sorted_cls]

    x = np.arange(len(labels))
    ax.barh(x, smalls, label="Small", color="#e74c3c", alpha=0.85)
    ax.barh(x, mediums, left=smalls, label="Medium", color="#f39c12", alpha=0.85)
    ax.barh(x, larges, left=np.array(smalls)+np.array(mediums), label="Large", color="#2ecc71", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title("Size Breakdown per Class", fontweight="bold")

    # 4) aspect ratio distribution
    ax = axes[1, 1]
    if all_wh:
        wh = np.array(all_wh)
        ratios = wh[:, 0] / np.maximum(wh[:, 1], 1e-6)
        ratios_clipped = np.clip(ratios, 0, np.percentile(ratios, 99))
        ax.hist(ratios_clipped, bins=60, color="#9b59b6", edgecolor="white", linewidth=0.3)
        ax.axvline(1.0, color="red", linestyle="--", alpha=0.5, label="1:1 (square)")
    ax.set_xlabel("Aspect ratio (w/h)")
    ax.set_ylabel("Count")
    ax.set_title("Aspect Ratio Distribution", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "size_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ── Step 6: Objects per image ───────────────────────────────

def step6_objects_per_image(all_data, class_names):
    print_header("6. OBJECTS PER IMAGE")

    for split_name, records in all_data.items():
        counts = [r["n_objects"] for r in records]
        print(f"\n  [{split_name.upper()}]")
        print(f"    Min:    {min(counts)}")
        print(f"    Max:    {max(counts)}")
        print(f"    Mean:   {np.mean(counts):.1f}")
        print(f"    Median: {np.median(counts):.0f}")
        print(f"    Std:    {np.std(counts):.1f}")

        # histogram of object counts
        hist = Counter()
        for c in counts:
            if c == 0:
                hist["0"] += 1
            elif c <= 5:
                hist["1-5"] += 1
            elif c <= 10:
                hist["6-10"] += 1
            elif c <= 20:
                hist["11-20"] += 1
            elif c <= 50:
                hist["21-50"] += 1
            else:
                hist["50+"] += 1

        print(f"    Distribution:")
        for bucket in ["0", "1-5", "6-10", "11-20", "21-50", "50+"]:
            n = hist.get(bucket, 0)
            print(f"      {bucket:>6s} objects: {n} images")

        # top 5 most crowded images
        sorted_by_count = sorted(records, key=lambda x: x["n_objects"], reverse=True)
        print(f"    Top 5 most crowded images:")
        for r in sorted_by_count[:5]:
            print(f"      {os.path.basename(r['image'])}: {r['n_objects']} objects")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (split_name, records) in zip(axes, all_data.items()):
        counts = [r["n_objects"] for r in records]
        ax.hist(counts, bins=range(0, max(counts)+2), color="#3498db", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Objects per image")
        ax.set_ylabel("Number of images")
        ax.set_title(f"{split_name.upper()} — Objects per Image", fontweight="bold")
        ax.axvline(np.mean(counts), color="red", linestyle="--", label=f"Mean: {np.mean(counts):.1f}")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "objects_per_image.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Saved: {path}")


# ── Step 7: Co-occurrence analysis ──────────────────────────

def step7_cooccurrence(all_data, class_names):
    print_header("7. CLASS CO-OCCURRENCE")

    # how often do classes appear together in the same image?
    all_records = []
    for records in all_data.values():
        all_records.extend(records)

    n_classes = max(class_names.keys()) + 1 if class_names else 17
    cooccurrence = np.zeros((n_classes, n_classes), dtype=int)

    for r in all_records:
        classes_in_image = set(cls_id for cls_id, *_ in r["boxes"])
        for c1 in classes_in_image:
            for c2 in classes_in_image:
                if c1 < n_classes and c2 < n_classes:
                    cooccurrence[c1][c2] += 1

    # plot heatmap
    labels = [class_names.get(i, f"cls_{i}") for i in range(n_classes)]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cooccurrence, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Co-occurrence count (images)")
    ax.set_title("Class Co-occurrence Matrix", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cooccurrence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")

    # print some interesting pairs
    print(f"\n  Top co-occurring class pairs:")
    pairs = []
    for i in range(n_classes):
        for j in range(i+1, n_classes):
            pairs.append((cooccurrence[i][j], i, j))
    pairs.sort(reverse=True)
    for count, i, j in pairs[:10]:
        n1 = class_names.get(i, f"cls_{i}")
        n2 = class_names.get(j, f"cls_{j}")
        print(f"    {n1} + {n2}: {count} images")

    # classes that rarely appear together (potential issue for context)
    print(f"\n  Rarely co-occurring (0 shared images):")
    zero_pairs = [(i, j) for count, i, j in pairs if count == 0]
    for i, j in zero_pairs[:10]:
        n1 = class_names.get(i, f"cls_{i}")
        n2 = class_names.get(j, f"cls_{j}")
        print(f"    {n1} + {n2}")
    if len(zero_pairs) > 10:
        print(f"    ... and {len(zero_pairs)-10} more pairs")


# ── Step 8: Summary & training recommendations ──────────────

def step8_summary(all_data, class_names, size_data):
    print_header("8. SUMMARY & RECOMMENDATIONS")

    total_imgs = sum(len(r) for r in all_data.values())
    total_instances = sum(r["n_objects"] for records in all_data.values() for r in records)

    # class imbalance
    all_counts = Counter()
    for records in all_data.values():
        for r in records:
            for cls_id, *_ in r["boxes"]:
                all_counts[cls_id] += 1

    sorted_cls = sorted(all_counts.items(), key=lambda x: x[1], reverse=True)
    max_c, min_c = sorted_cls[0][1], sorted_cls[-1][1]
    ratio = max_c / min_c if min_c > 0 else float("inf")

    # small object percentage
    total_sized = sum(size_data.values())
    pct_small = 100 * size_data["small"] / total_sized if total_sized > 0 else 0

    print(f"""
  Dataset overview:
    Total images:     {total_imgs}
    Total instances:  {total_instances}
    Classes:          {len(all_counts)}
    Avg obj/image:    {total_instances/total_imgs:.1f}

  Key challenges:
    Class imbalance:  {ratio:.0f}x (most: {class_names.get(sorted_cls[0][0])}, "
                      f"least: {class_names.get(sorted_cls[-1][0])})
    Small objects:    {pct_small:.1f}% of all objects

  Recommendations:
    1. Image size:    Use imgsz=1280 (many small objects)
    2. Augmentation:  Enable copy_paste=0.3 for minority classes
    3. Mosaic:        Keep mosaic=1.0, close_mosaic=20
    4. Scale aug:     scale=0.5 (important for size variation)
    5. LR:            Start low (0.0005 for YOLO, 0.0001 for transformers)
    6. Patience:      Set early stopping patience=30 (imbalanced data converges slowly)
    """)

    # identify problem classes (small count + small size)
    print(f"  ⚠ Problem classes (low count AND high %small):")
    for cls_id, count in sorted_cls:
        name = class_names.get(cls_id, f"class_{cls_id}")
        if cls_id in size_data:
            # skip — this is the global size_data, not per-class
            pass
        if count < total_instances * 0.03:  # less than 3% of total
            print(f"    {name}: {count} instances ({100*count/total_instances:.1f}%)")


# ── Main ────────────────────────────────────────────────────

def main():
    print("\n" + "━" * 65)
    print("  SH17 DATASET — FULL ANALYSIS")
    print("━" * 65)
    print(f"  Root: {DATASET_ROOT}")
    print(f"  Output: {OUTPUT_DIR}")
    print("━" * 65)

    step1_file_structure()
    train_imgs, val_imgs = step2_resolve_splits()
    all_data, class_ids = step3_label_analysis(train_imgs, val_imgs)
    class_names = step4_class_distribution(all_data, class_ids)
    size_data, size_per_class = step5_object_sizes(all_data, class_names)
    step6_objects_per_image(all_data, class_names)
    step7_cooccurrence(all_data, class_names)
    step8_summary(all_data, class_names, size_data)

    print(f"\n{'━'*65}")
    print(f"  Done! All plots saved to: {OUTPUT_DIR}")
    print(f"{'━'*65}\n")


if __name__ == "__main__":
    main()