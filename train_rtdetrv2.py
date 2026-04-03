"""
Train RT-DETRv2-X on SH17.

RT-DETR is a transformer-based detector (no NMS, end-to-end).
Good at capturing global context between body parts and PPE.
Uses Ultralytics framework, so dataset setup is shared with YOLO11x.

Usage:
    python train_rtdetrv2.py --mode quick
    python train_rtdetrv2.py --mode full
    python train_rtdetrv2.py --mode full --batch 2         # if OOM
    python train_rtdetrv2.py --mode full --device 0,1      # both GPUs
    python train_rtdetrv2.py --mode finetune --weights runs/rtdetrv2_full/weights/best.pt
    python train_rtdetrv2.py --mode full --resume
"""

import os
import argparse
from ultralytics import RTDETR

# ── Config ──────────────────────────────────────────────────
PROJECT = os.path.expanduser("~/PPE")
DATASET_YAML = os.path.join(PROJECT, "sh17.yaml")
RUNS_DIR = os.path.join(PROJECT, "runs")


def get_config(mode, device, batch_override=None, weights=None):

    if mode == "quick":
        cfg = dict(
            model="rtdetr-x.pt",
            data=DATASET_YAML,
            epochs=30,
            imgsz=640,
            batch=8,               # RT-DETR is heavier than YOLO
            device=device,
            project=RUNS_DIR,
            name="rtdetrv2_quick",
            optimizer="AdamW",
            lr0=0.0001,            # transformers need lower LR
            lrf=0.01,
            weight_decay=0.0001,
            warmup_epochs=3,
            save=True,
            val=True,
            plots=True,
        )

    elif mode == "full":
        cfg = dict(
            model="rtdetr-x.pt",
            data=DATASET_YAML,
            epochs=150,            # transformers converge faster than YOLO
            imgsz=1280,
            # RT-DETR at 1280 uses more VRAM than YOLO at 1280.
            # start with 4, drop to 2 if OOM.
            batch=4,
            device=device,
            project=RUNS_DIR,
            name="rtdetrv2_full",

            # --- optimizer ---
            # transformers are sensitive to LR. 1e-4 is standard for DETR family.
            # backbone gets even lower LR via internal scaling.
            optimizer="AdamW",
            lr0=0.0001,
            lrf=0.01,
            weight_decay=0.0001,
            warmup_epochs=5,
            warmup_bias_lr=0.001,
            cos_lr=True,
            nbs=64,

            # --- augmentation ---
            # RT-DETR benefits from similar augmentation as YOLO,
            # but we keep it slightly milder since the transformer
            # encoder already provides regularization via attention.
            mosaic=1.0,
            close_mosaic=15,
            mixup=0.1,
            copy_paste=0.2,
            scale=0.5,
            degrees=3.0,
            translate=0.1,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
            erasing=0.15,

            # --- training control ---
            label_smoothing=0.01,
            patience=25,
            save=True,
            save_period=15,
            val=True,
            plots=True,
        )

    elif mode == "finetune":
        model_path = weights or os.path.join(RUNS_DIR, "rtdetrv2_full", "weights", "best.pt")
        if not os.path.isfile(model_path):
            print(f"ERROR: weights not found at {model_path}")
            exit(1)

        cfg = dict(
            model=model_path,
            data=DATASET_YAML,
            epochs=40,
            imgsz=1280,
            batch=4,
            device=device,
            project=RUNS_DIR,
            name="rtdetrv2_finetune",
            optimizer="AdamW",
            lr0=0.00003,
            lrf=0.01,
            weight_decay=0.0001,
            warmup_epochs=2,
            cos_lr=True,
            freeze=15,
            mosaic=0.3,
            mixup=0.05,
            scale=0.3,
            patience=15,
            save=True,
            val=True,
            plots=True,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    if batch_override is not None:
        cfg["batch"] = batch_override

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full", "finetune"], default="quick")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.mode, args.device, args.batch, args.weights)

    print(f"\n{'='*55}")
    print(f"  RT-DETRv2-X — {args.mode.upper()}")
    print(f"{'='*55}")
    print(f"  epochs:  {cfg['epochs']}")
    print(f"  imgsz:   {cfg['imgsz']}")
    print(f"  batch:   {cfg['batch']}")
    print(f"  device:  {cfg['device']}")
    print(f"  output:  {cfg['project']}/{cfg['name']}")
    print(f"{'='*55}\n")

    if args.resume:
        last_pt = os.path.join(RUNS_DIR, cfg["name"], "weights", "last.pt")
        print(f"  Resuming from: {last_pt}")
        model = RTDETR(last_pt)
        model.train(resume=True)
    else:
        model_id = cfg.pop("model")
        model = RTDETR(model_id)
        model.train(**cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()