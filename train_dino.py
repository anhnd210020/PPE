"""
Train DINO on SH17 via mmdetection.

DINO = DETR with Improved deNoising anchOr boxes.
Uses deformable attention + contrastive denoising → strong on small objects.

Backbone options:
  swinl  - Swin-L (strongest, needs ~35-40GB VRAM, use 2x 4090 with DDP)
  r50    - ResNet-50 (lighter, ~16GB, single 4090 is enough)

Prerequisites:
    pip install -U openmim
    mim install mmengine mmcv mmdet

Usage:
    python train_dino.py --mode quick                     # R50, 12 epochs, sanity check
    python train_dino.py --mode full --backbone swinl     # Swin-L, 2x GPU
    python train_dino.py --mode full --backbone r50       # R50, single GPU
    python train_dino.py --mode full --backbone swinl --gpus 2
"""

import os
import sys
import argparse
import tempfile

# ── Config ──────────────────────────────────────────────────
PROJECT = os.path.expanduser("~/PPE")
COCO_DIR = os.path.join(PROJECT, "data_coco")
IMAGES_DIR = os.path.join(COCO_DIR, "images")
TRAIN_JSON = os.path.join(COCO_DIR, "train.json")
VAL_JSON = os.path.join(COCO_DIR, "val.json")
RUNS_DIR = os.path.join(PROJECT, "runs")

CLASS_NAMES = [
    "person", "ear", "ear-mufs", "face", "face-guard",
    "face-mask-medical", "foot", "tools", "glasses", "gloves",
    "helmet", "hands", "head", "medical-suit", "shoes",
    "safety-suit", "safety-vest",
]
NUM_CLASSES = 17


# ── Config generation ──────────────────────────────────────

def generate_config(mode, backbone, work_dir):
    """
    Generate a mmdetection config file for DINO.
    Returns the path to the generated .py config.
    """
    max_epochs = {"quick": 12, "full": 36}[mode]
    batch_size = 2  # DINO is heavy, 2 per GPU is typical
    base_lr = 0.0001

    # backbone-specific settings
    if backbone == "swinl":
        backbone_cfg = _swinl_backbone()
        neck_channels = [192, 384, 768, 1536]
        num_feature_levels = 5
    else:
        backbone_cfg = _r50_backbone()
        neck_channels = [512, 1024, 2048]
        num_feature_levels = 4

    config_str = f"""
_base_ = ['mmdet::_base_/default_runtime.py']

model = dict(
    type='DINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),
{backbone_cfg}
    neck=dict(
        type='ChannelMapper',
        in_channels={neck_channels},
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs={num_feature_levels},
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels={num_feature_levels}, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels={num_feature_levels}, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        post_norm_cfg=None,
    ),
    positional_encoding=dict(num_feats=128, normalize=True, offset=0.0, temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes={NUM_CLASSES},
        sync_cls_avg_factor=True,
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
    ),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100),
    ),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0),
            ],
        ),
    ),
    test_cfg=dict(max_per_img=300),
)

# -- dataset --
dataset_type = 'CocoDataset'
metainfo = dict(classes={CLASS_NAMES})

# multi-scale training: important for small objects.
# scales range from 480 to 1280 on the short side.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [dict(type='RandomChoiceResize',
                  scales=[(s, 2048) for s in range(480, 1312, 32)],
                  keep_ratio=True)],
            [dict(type='RandomChoiceResize', scales=[(400, 4200), (500, 4200), (600, 4200)], keep_ratio=True),
             dict(type='RandomCrop', crop_type='absolute_range', crop_size=(384, 600), allow_negative_crop=True),
             dict(type='RandomChoiceResize',
                  scales=[(s, 2048) for s in range(480, 1312, 32)],
                  keep_ratio=True)],
        ],
    ),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1280, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size={batch_size},
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root='{COCO_DIR}',
        ann_file='train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size={batch_size},
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root='{COCO_DIR}',
        ann_file='val.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file='{VAL_JSON}',
    metric='bbox',
    format_only=False,
    classwise=True,
)
test_evaluator = val_evaluator

# -- schedule --
max_epochs = {max_epochs}
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr={base_lr}, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={{
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
        }},
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(type='MultiStepLR', begin=0, end=max_epochs,
         by_epoch=True, milestones=[int(max_epochs * 0.75)], gamma=0.1),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=3, max_keep_ckpts=5,
                    save_best='coco/bbox_mAP', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

auto_scale_lr = dict(enable=True, base_batch_size=16)
"""

    config_path = os.path.join(work_dir, f"dino_{backbone}_{mode}.py")
    os.makedirs(work_dir, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(config_str)

    return config_path


def _swinl_backbone():
    return """    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
        ),
    ),"""


def _r50_backbone():
    return """    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),"""


# ── Checks ──────────────────────────────────────────────────

def check_deps():
    try:
        import mmdet
        import mmengine
        print(f"  mmdetection: {mmdet.__version__}")
        print(f"  mmengine:    {mmengine.__version__}")
        return True
    except ImportError as e:
        print(f"  ERROR: {e}")
        print(f"  Install with:")
        print(f"    pip install -U openmim")
        print(f"    mim install mmengine mmcv mmdet")
        return False


def check_data():
    ok = True
    for f in [TRAIN_JSON, VAL_JSON]:
        if os.path.isfile(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} — run prepare_data.py first")
            ok = False
    return ok


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--backbone", choices=["swinl", "r50"], default="r50")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  DINO ({args.backbone}) — {args.mode.upper()}")
    print(f"{'='*55}")

    # checks
    if not check_deps():
        sys.exit(1)
    if not check_data():
        sys.exit(1)

    # generate config
    work_dir = os.path.join(RUNS_DIR, f"dino_{args.backbone}_{args.mode}")
    config_path = generate_config(args.mode, args.backbone, work_dir)
    print(f"  config:   {config_path}")
    print(f"  work_dir: {work_dir}")
    print(f"  GPUs:     {args.gpus}")

    if args.backbone == "swinl" and args.gpus == 1:
        print(f"\n  ⚠ Swin-L on single 4090 might OOM.")
        print(f"    Recommend: --gpus 2 or --backbone r50")

    print(f"{'='*55}\n")

    # train
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(config_path)
    cfg.work_dir = work_dir

    if args.resume:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.gpus > 1:
        # launch with torchrun for multi-GPU
        import subprocess
        cmd = (
            f"torchrun --nproc_per_node={args.gpus} "
            f"-m mmengine.runner --config {config_path} "
            f"--work-dir {work_dir} --launcher pytorch"
        )
        print(f"  Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    else:
        runner = Runner.from_cfg(cfg)
        runner.train()

    print(f"\n  Done! Checkpoints: {work_dir}")


if __name__ == "__main__":
    main()