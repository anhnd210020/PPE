"""
Train YOLO11x on SH17.

Three modes:
  quick    - 30 epochs, imgsz=640, fast sanity check (~1-2 hours on 4090)
  full     - 200 epochs, imgsz=1280, production training (~12-18 hours)
  finetune - 50 epochs, lower LR, freeze backbone, from best.pt

Usage:
    python train_yolo11x.py --mode quick
    python train_yolo11x.py --mode full
    python train_yolo11x.py --mode full --batch 4          # if OOM
    python train_yolo11x.py --mode full --device 0,1       # both GPUs
    python train_yolo11x.py --mode finetune --weights runs/yolo11x_full/weights/best.pt
    python train_yolo11x.py --mode full --resume            # resume interrupted training
"""

import os
import argparse
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────
PROJECT = os.path.expanduser("~/PPE")
DATASET_YAML = os.path.join(PROJECT, "sh17.yaml")
RUNS_DIR = os.path.join(PROJECT, "runs")


def get_config(mode, device, batch_override=None, weights=None):
    """
    Build training config dict for each mode.
    All the decisions here are tuned for SH17's specific challenges:
    high-res images, small objects, 118x class imbalance, dense scenes.
    """

    if mode == "quick":
        # just enough to verify the pipeline works end-to-end
        cfg = dict(
            model="yolo11x.pt",
            data=DATASET_YAML,
            epochs=30,
            imgsz=640,
            batch=16,
            device=device,
            project=RUNS_DIR,
            name="yolo11x_quick",
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
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
            model="yolo11x.pt",
            data=DATASET_YAML,
            epochs=120,
            # --- imgsz=1280 is critical ---
            # original images are 4000x6000+, so even 1280 is a 3-5x downscale.
            # going higher than 1280 would be better but VRAM is the bottleneck.
            # 1280 is the sweet spot for 24GB GPUs.
            imgsz=640,
            batch=32,              # 8 works on 4090 at 1280. drop to 4 if OOM.
            device=device,
            project=RUNS_DIR,
            name="yolo11x_full",

            # --- optimizer ---
            # AdamW with lower LR since model is large and images are high-res.
            # cosine schedule decays to lr0 * lrf = 0.0005 * 0.01 = 5e-6.
            optimizer="AdamW",
            lr0=0.0005,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=5,
            warmup_bias_lr=0.01,
            cos_lr=True,
            nbs=64,               # nominal batch size for LR scaling

            # --- augmentation (tuned for SH17) ---
            # mosaic: combines 4 images into 1. great for:
            #   - seeing more small objects per batch
            #   - mixing minority class images with majority ones
            mosaic=1.0,
            close_mosaic=20,      # turn off mosaic last 20 epochs for fine detail

            # mixup: blends two images. mild regularization.
            mixup=0.15,

            # copy_paste: copies object instances onto other images.
            # this is THE key augmentation for class imbalance —
            # minority class objects get pasted into more scenes.
            copy_paste=0.3,

            # scale: randomly scales objects 0.5x-1.5x.
            # critical for small object robustness since SH17 has huge
            # variation in object sizes (PPE on nearby vs far workers).
            scale=0.5,

            # spatial transforms
            degrees=5.0,
            translate=0.1,
            shear=2.0,
            perspective=0.0001,
            flipud=0.1,
            fliplr=0.5,

            # color augmentation — helps generalize across lighting
            # conditions in different factory environments.
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,

            # random erasing: forces model to not rely on single features
            erasing=0.2,

            # --- loss ---
            cls=1.0,
            box=7.5,
            dfl=1.5,
            label_smoothing=0.01,

            # --- training control ---
            patience=30,          # early stopping (imbalanced data can be noisy)
            save=True,
            save_period=20,
            val=True,
            plots=True,
            verbose=True,
        )

    elif mode == "finetune":
        # start from best.pt, lower LR, freeze early layers,
        # lighter augmentation to avoid forgetting what was learned.
        model_path = weights or os.path.join(RUNS_DIR, "yolo11x_full", "weights", "best.pt")
        if not os.path.isfile(model_path):
            print(f"ERROR: weights not found at {model_path}")
            print("  Either train full mode first, or pass --weights explicitly.")
            exit(1)

        cfg = dict(
            model=model_path,
            data=DATASET_YAML,
            epochs=50,
            imgsz=1280,
            batch=8,
            device=device,
            project=RUNS_DIR,
            name="yolo11x_finetune",
            optimizer="AdamW",
            lr0=0.0001,           # 5x lower than full
            lrf=0.01,
            weight_decay=0.001,
            warmup_epochs=2,
            cos_lr=True,
            freeze=10,            # freeze first 10 layers
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

    # override batch if user specified
    if batch_override is not None:
        cfg["batch"] = batch_override

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full", "finetune"], default="quick")
    parser.add_argument("--device", type=str, default="0",
                        help="GPU device(s). '0' for single, '0,1' for both.")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights (finetune mode)")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt")
    args = parser.parse_args()

    cfg = get_config(args.mode, args.device, args.batch, args.weights)

    print(f"\n{'='*55}")
    print(f"  YOLO11x — {args.mode.upper()}")
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