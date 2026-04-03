"""
Prepare SH17 dataset for training.

What this does:
  1. Create Ultralytics-compatible folder layout (symlinks, no copying)
  2. Generate dataset YAML for YOLO11x and RT-DETRv2
  3. Convert annotations to COCO JSON for DINO (mmdetection)
  4. Validate everything

Usage:
    python prepare_data.py
"""

import os
import sys
import json
import yaml
from pathlib import Path
from collections import Counter

from PIL import Image
from tqdm import tqdm

# ── Paths ───────────────────────────────────────────────────
DATASET_ROOT = os.path.expanduser("~/PPE/sh17dataset")
PROJECT_ROOT = os.path.expanduser("~/PPE")

IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
TRAIN_TXT  = os.path.join(DATASET_ROOT, "train_files.txt")
VAL_TXT    = os.path.join(DATASET_ROOT, "val_files.txt")

# output dirs
ULTR_DIR   = os.path.join(PROJECT_ROOT, "data_ultralytics")   # for YOLO11x + RT-DETRv2
COCO_DIR   = os.path.join(PROJECT_ROOT, "data_coco")          # for DINO

# ── Class mapping (verified with check_class_mapping.py) ───
CLASS_NAMES = [
    "person",             # 0
    "ear",                # 1
    "ear-mufs",           # 2
    "face",               # 3
    "face-guard",         # 4
    "face-mask-medical",  # 5
    "foot",               # 6
    "tools",              # 7
    "glasses",            # 8
    "gloves",             # 9
    "helmet",             # 10
    "hands",              # 11
    "head",               # 12
    "medical-suit",       # 13
    "shoes",              # 14
    "safety-suit",        # 15
    "safety-vest",        # 16
]
NUM_CLASSES = len(CLASS_NAMES)


# ── Helpers ─────────────────────────────────────────────────

def read_filelist(txt_path):
    with open(txt_path) as f:
        return [line.strip() for line in f if line.strip()]


def find_image(name, img_dir):
    """Resolve image filename to full path."""
    full = os.path.join(img_dir, name)
    if os.path.isfile(full):
        return full
    stem = os.path.splitext(name)[0]
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"]:
        p = os.path.join(img_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def resolve_split(txt_path, img_dir):
    """Read a split file and resolve all image paths."""
    names = read_filelist(txt_path)
    found, missing = [], []
    for name in names:
        img = find_image(name, img_dir)
        if img:
            found.append(img)
        else:
            missing.append(name)
    return found, missing


# ── Step 1: Ultralytics layout ──────────────────────────────

def create_ultralytics_layout(train_imgs, val_imgs):
    """
    Ultralytics expects:
        data_ultralytics/
            images/train/  -> symlinks to actual images
            images/val/
            labels/train/  -> symlinks to actual labels
            labels/val/
    """
    print("\n[Step 1] Creating Ultralytics directory layout...")

    dirs = {
        "img_train": os.path.join(ULTR_DIR, "images", "train"),
        "img_val":   os.path.join(ULTR_DIR, "images", "val"),
        "lbl_train": os.path.join(ULTR_DIR, "labels", "train"),
        "lbl_val":   os.path.join(ULTR_DIR, "labels", "val"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    def symlink_split(img_paths, img_dst, lbl_dst):
        created, skipped, no_label = 0, 0, 0
        for img_path in img_paths:
            fname = os.path.basename(img_path)
            stem = Path(img_path).stem

            # image symlink
            link_img = os.path.join(img_dst, fname)
            if not os.path.exists(link_img):
                os.symlink(os.path.abspath(img_path), link_img)
                created += 1
            else:
                skipped += 1

            # label symlink
            label_src = os.path.join(LABELS_DIR, stem + ".txt")
            link_lbl = os.path.join(lbl_dst, stem + ".txt")
            if os.path.isfile(label_src):
                if not os.path.exists(link_lbl):
                    os.symlink(os.path.abspath(label_src), link_lbl)
            else:
                no_label += 1

        return created, skipped, no_label

    c1, s1, n1 = symlink_split(train_imgs, dirs["img_train"], dirs["lbl_train"])
    print(f"  Train: {c1} symlinks created, {s1} already exist, {n1} missing labels")

    c2, s2, n2 = symlink_split(val_imgs, dirs["img_val"], dirs["lbl_val"])
    print(f"  Val:   {c2} symlinks created, {s2} already exist, {n2} missing labels")

    # sanity check: count files
    for name, path in dirs.items():
        n = len(os.listdir(path))
        print(f"  {name}: {n} files")


# ── Step 2: Dataset YAML ────────────────────────────────────

def create_dataset_yaml():
    """Generate the YAML config that Ultralytics reads."""
    print("\n[Step 2] Creating dataset YAML...")

    config = {
        "path": os.path.abspath(ULTR_DIR),
        "train": "images/train",
        "val": "images/val",
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }

    yaml_path = os.path.join(PROJECT_ROOT, "sh17.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"  Saved: {yaml_path}")

    # print it so user can verify
    print(f"  Contents:")
    with open(yaml_path) as f:
        for line in f:
            print(f"    {line.rstrip()}")

    return yaml_path


# ── Step 3: COCO JSON (for DINO) ────────────────────────────

def convert_to_coco(img_paths, split_name):
    """
    Convert YOLO labels to COCO JSON format.
    YOLO: class_id x_center y_center width height (normalized)
    COCO: {"bbox": [x_min, y_min, width, height]} (absolute pixels)
    """
    print(f"\n  Converting {split_name}...")

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)],
    }

    ann_id = 1
    skipped = 0

    for img_id, img_path in enumerate(tqdm(img_paths, desc=f"  {split_name}"), start=1):
        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except Exception:
            skipped += 1
            continue

        coco["images"].append({
            "id": img_id,
            "file_name": os.path.basename(img_path),
            "width": img_w,
            "height": img_h,
        })

        # read yolo label
        stem = Path(img_path).stem
        label_path = os.path.join(LABELS_DIR, stem + ".txt")
        if not os.path.isfile(label_path):
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                except ValueError:
                    continue

                # normalized -> absolute
                abs_w = w * img_w
                abs_h = h * img_h
                x_min = max(0, (xc - w / 2) * img_w)
                y_min = max(0, (yc - h / 2) * img_h)

                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [round(x_min, 2), round(y_min, 2), round(abs_w, 2), round(abs_h, 2)],
                    "area": round(abs_w * abs_h, 2),
                    "iscrowd": 0,
                })
                ann_id += 1

    # save
    out_path = os.path.join(COCO_DIR, f"{split_name}.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)

    n_imgs = len(coco["images"])
    n_anns = len(coco["annotations"])
    print(f"  {split_name}: {n_imgs} images, {n_anns} annotations -> {out_path}")
    if skipped:
        print(f"  (skipped {skipped} unreadable images)")

    return out_path


def create_coco_annotations(train_imgs, val_imgs):
    """Convert both splits to COCO format."""
    print("\n[Step 3] Converting to COCO JSON (for DINO)...")
    os.makedirs(COCO_DIR, exist_ok=True)

    # DINO needs images accessible from a common path
    # we'll symlink images dir into data_coco for convenience
    coco_img_link = os.path.join(COCO_DIR, "images")
    if not os.path.exists(coco_img_link):
        os.symlink(os.path.abspath(IMAGES_DIR), coco_img_link)
        print(f"  Symlinked images: {coco_img_link} -> {IMAGES_DIR}")

    train_json = convert_to_coco(train_imgs, "train")
    val_json = convert_to_coco(val_imgs, "val")

    return train_json, val_json


# ── Step 4: Validate ────────────────────────────────────────

def validate(train_imgs, val_imgs, yaml_path):
    """Quick sanity checks before training."""
    print("\n[Step 4] Validation...")

    errors = 0

    # check yaml loads correctly
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["nc"] == NUM_CLASSES, f"nc mismatch: {cfg['nc']} vs {NUM_CLASSES}"
    assert len(cfg["names"]) == NUM_CLASSES

    # check ultralytics dirs have correct counts
    train_img_dir = os.path.join(ULTR_DIR, "images", "train")
    val_img_dir = os.path.join(ULTR_DIR, "images", "val")
    train_lbl_dir = os.path.join(ULTR_DIR, "labels", "train")
    val_lbl_dir = os.path.join(ULTR_DIR, "labels", "val")

    n_train_img = len(os.listdir(train_img_dir))
    n_val_img = len(os.listdir(val_img_dir))
    n_train_lbl = len(os.listdir(train_lbl_dir))
    n_val_lbl = len(os.listdir(val_lbl_dir))

    print(f"  Ultralytics layout:")
    print(f"    Train images: {n_train_img}, labels: {n_train_lbl}")
    print(f"    Val images:   {n_val_img}, labels: {n_val_lbl}")

    if n_train_img != n_train_lbl:
        print(f"    ⚠ Train image/label count mismatch!")
        errors += 1
    if n_val_img != n_val_lbl:
        print(f"    ⚠ Val image/label count mismatch!")
        errors += 1

    # check COCO JSON
    for split in ["train", "val"]:
        json_path = os.path.join(COCO_DIR, f"{split}.json")
        if os.path.isfile(json_path):
            with open(json_path) as f:
                data = json.load(f)
            print(f"  COCO {split}.json: {len(data['images'])} images, "
                  f"{len(data['annotations'])} annotations, "
                  f"{len(data['categories'])} categories")
        else:
            print(f"  ⚠ Missing: {json_path}")
            errors += 1

    # spot check: pick a random label and verify class ids are in range
    import random
    sample_imgs = random.sample(train_imgs, min(100, len(train_imgs)))
    bad_ids = 0
    for img_path in sample_imgs:
        stem = Path(img_path).stem
        lbl = os.path.join(LABELS_DIR, stem + ".txt")
        if not os.path.isfile(lbl):
            continue
        with open(lbl) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id < 0 or cls_id >= NUM_CLASSES:
                        bad_ids += 1

    if bad_ids:
        print(f"  ⚠ Found {bad_ids} annotations with out-of-range class IDs!")
        errors += 1
    else:
        print(f"  Spot check: all class IDs in [0, {NUM_CLASSES-1}] ✓")

    if errors == 0:
        print("\n  ✓ All checks passed. Ready to train!")
    else:
        print(f"\n  ⚠ {errors} issue(s) found. Check above.")

    return errors


# ── Main ────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SH17 — Data Preparation")
    print("=" * 55)
    print(f"  Dataset:  {DATASET_ROOT}")
    print(f"  Output:   {PROJECT_ROOT}")
    print(f"  Classes:  {NUM_CLASSES}")

    # resolve splits
    train_imgs, train_miss = resolve_split(TRAIN_TXT, IMAGES_DIR)
    val_imgs, val_miss = resolve_split(VAL_TXT, IMAGES_DIR)
    print(f"\n  Train: {len(train_imgs)} images ({len(train_miss)} missing)")
    print(f"  Val:   {len(val_imgs)} images ({len(val_miss)} missing)")

    # do it
    create_ultralytics_layout(train_imgs, val_imgs)
    yaml_path = create_dataset_yaml()
    create_coco_annotations(train_imgs, val_imgs)
    validate(train_imgs, val_imgs, yaml_path)

    # summary
    print(f"\n{'='*55}")
    print(f"  Done! Created:")
    print(f"    {ULTR_DIR}/          <- for YOLO11x & RT-DETRv2")
    print(f"    {PROJECT_ROOT}/sh17.yaml  <- dataset config")
    print(f"    {COCO_DIR}/              <- for DINO")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()