"""
Train YOLOv9-e on SH17.

This is the paper's best baseline (70.9 mAP@50, 48.7 mAP@50:95).
We replicate and attempt to surpass it.

Usage:
    python train_yolov9e.py --mode quick
    python train_yolov9e.py --mode full
    python train_yolov9e.py --mode full --device 1         # GPU 1 only
    python train_yolov9e.py --mode full --batch 4           # if OOM
    python train_yolov9e.py --mode finetune --weights runs/yolov9e_full/weights/best.pt
    python train_yolov9e.py --mode full --resume
"""

import os
import argparse
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────
PROJECT = os.path.expanduser("~/PPE")
DATASET_YAML = os.path.join(PROJECT, "sh17.yaml")
RUNS_DIR = os.path.join(PROJECT, "runs")


def get_config(mode, device, batch_override=None, weights=None):

    if mode == "quick":
        cfg = dict(
            model="yolov9e.pt",
            data=DATASET_YAML,
            epochs=30,
            imgsz=640,
            batch=16,
            device=device,
            project=RUNS_DIR,
            name="yolov9e_quick",
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.0,
            save=True,
            save_period=10,
            val=True,
            plots=True,
            verbose=True,
        )

    elif mode == "full":
        cfg = dict(
            model="yolov9e.pt",
            data=DATASET_YAML,
            epochs=120,
            imgsz=640,
            # YOLOv9-e is heavier than YOLO11x (~58M params + aux branch).
            # batch=8 should fit on 4090 at 1280. drop to 4 if OOM.
            batch=16,
            device=device,
            project=RUNS_DIR,
            name="yolov9e_full",

            # --- optimizer ---
            # SGD is the default for YOLOv9 in the original repo.
            # sticking with it for fair comparison with the paper.
            optimizer="SGD",
            lr0=0.01,
            lrf=0.01,              # cosine decay to lr0 * lrf = 1e-4
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            cos_lr=True,
            nbs=64,

            # --- augmentation ---
            mosaic=1.0,
            close_mosaic=15,
            mixup=0.15,
            copy_paste=0.3,        # help minority classes
            scale=0.5,             # size variation for small objects
            degrees=0.0,           # no rotation (keep it close to paper defaults)
            translate=0.1,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            erasing=0.0,

            # --- loss ---
            cls=0.5,
            box=7.5,
            dfl=1.5,
            label_smoothing=0.0,

            # --- training control ---
            patience=50,
            save=True,
            save_period=20,
            val=True,
            plots=True,
            verbose=True,
        )

    elif mode == "finetune":
        model_path = weights or os.path.join(RUNS_DIR, "yolov9e_full", "weights", "best.pt")
        if not os.path.isfile(model_path):
            print(f"ERROR: weights not found at {model_path}")
            print("  Train full mode first, or pass --weights explicitly.")
            exit(1)

        cfg = dict(
            model=model_path,
            data=DATASET_YAML,
            epochs=50,
            imgsz=1280,
            batch=8,
            device=device,
            project=RUNS_DIR,
            name="yolov9e_finetune",
            optimizer="SGD",
            lr0=0.001,             # 10x lower than full
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=2,
            cos_lr=True,
            freeze=10,
            mosaic=0.5,
            mixup=0.05,
            copy_paste=0.2,
            scale=0.3,
            patience=15,
            save=True,
            save_period=10,
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
    print(f"  YOLOv9-e — {args.mode.upper()}")
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
        model = YOLO(last_pt)
        model.train(resume=True)
    else:
        model_id = cfg.pop("model")
        model = YOLO(model_id)
        model.train(**cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()