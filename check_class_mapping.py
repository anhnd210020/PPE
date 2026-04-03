"""
Figure out class_id -> class_name mapping.
Strategy: compare VOC XML labels (has class names) with YOLO txt labels (has class ids)
for the same images to build the mapping.

Usage:
    python check_class_mapping.py
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = os.path.expanduser("~/PPE/sh17dataset")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
VOC_DIR = os.path.join(DATASET_ROOT, "voc_labels")


def parse_voc(xml_path):
    """Parse VOC XML, return list of (class_name, xmin, ymin, xmax, ymax)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)
        objects.append((name, xmin, ymin, xmax, ymax))

    return objects, img_w, img_h


def parse_yolo(txt_path):
    """Parse YOLO txt, return list of (class_id, xc, yc, w, h)."""
    boxes = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append((int(parts[0]), *map(float, parts[1:5])))
    return boxes


def match_boxes(voc_objs, yolo_boxes, img_w, img_h):
    """
    Match VOC boxes to YOLO boxes by center-point proximity.
    Returns list of (class_name, class_id) pairs.
    """
    matches = []
    used = set()

    for name, xmin, ymin, xmax, ymax in voc_objs:
        # VOC center (absolute pixels)
        voc_cx = (xmin + xmax) / 2.0
        voc_cy = (ymin + ymax) / 2.0

        best_dist = float("inf")
        best_idx = -1

        for i, (cls_id, xc, yc, w, h) in enumerate(yolo_boxes):
            if i in used:
                continue
            # YOLO center (convert to absolute)
            yolo_cx = xc * img_w
            yolo_cy = yc * img_h

            dist = ((voc_cx - yolo_cx)**2 + (voc_cy - yolo_cy)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        # only accept close matches (within 5px)
        if best_idx >= 0 and best_dist < 5.0:
            used.add(best_idx)
            matches.append((name, yolo_boxes[best_idx][0]))

    return matches


def main():
    # find files that exist in both voc_labels/ and labels/
    voc_files = {Path(f).stem for f in os.listdir(VOC_DIR) if f.endswith(".xml")}
    yolo_files = {Path(f).stem for f in os.listdir(LABELS_DIR) if f.endswith(".txt")}
    common = sorted(voc_files & yolo_files)

    print(f"VOC label files:  {len(voc_files)}")
    print(f"YOLO label files: {len(yolo_files)}")
    print(f"Common (matched): {len(common)}")

    # build mapping by checking many files
    mapping_votes = defaultdict(lambda: defaultdict(int))  # name -> {id: count}

    sample = common[:500]  # check up to 500 files
    for stem in sample:
        voc_path = os.path.join(VOC_DIR, stem + ".xml")
        yolo_path = os.path.join(LABELS_DIR, stem + ".txt")

        try:
            voc_objs, img_w, img_h = parse_voc(voc_path)
            yolo_boxes = parse_yolo(yolo_path)
            matches = match_boxes(voc_objs, yolo_boxes, img_w, img_h)
        except Exception as e:
            continue

        for name, cls_id in matches:
            mapping_votes[name][cls_id] += 1

    # pick the most-voted id for each name
    print(f"\n{'='*50}")
    print(f"  CLASS MAPPING (sorted by index)")
    print(f"{'='*50}")

    final_mapping = {}
    for name, votes in mapping_votes.items():
        best_id = max(votes, key=votes.get)
        confidence = votes[best_id]
        total = sum(votes.values())
        final_mapping[best_id] = (name, confidence, total)

    for cls_id in sorted(final_mapping.keys()):
        name, conf, total = final_mapping[cls_id]
        pct = 100 * conf / total
        print(f"  {cls_id:>2d}  →  {name:<25s}  (confidence: {conf}/{total} = {pct:.0f}%)")

    # also print as a python list (copy-paste ready)
    print(f"\n{'='*50}")
    print(f"  COPY-PASTE READY")
    print(f"{'='*50}")
    print("CLASS_NAMES = [")
    for i in range(max(final_mapping.keys()) + 1):
        if i in final_mapping:
            print(f'    "{final_mapping[i][0]}",  # {i}')
        else:
            print(f'    "unknown_{i}",  # {i}')
    print("]")


if __name__ == "__main__":
    main()