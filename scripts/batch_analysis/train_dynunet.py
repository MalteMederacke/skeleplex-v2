"""Train a DynUNet with skeleton-recall loss on the patches from
``prepare_patches.py``.

    python train_dynunet.py --channel ssh

Patch files on disk are PATCH_SIZE^3 and a random CROP_SIZE^3 crop is taken
every time one is loaded, so the network sees a different window offset each
epoch. Three things here exist to stop the network merging neighbouring
branches in dense regions:

1. Random crops. With patch_shape == stride on network-sized files, each file
   yields exactly ONE fixed crop, so the network sees every dense region at a
   single offset for the whole run and never learns to be invariant to where
   the window lands -- which is exactly what breaks at inference.

2. Real augmentation. At prob=0.05 per transform, ~85% of samples go through
   untouched; the probabilities below actually bite.

3. deep_supr_num = 2. MONAI nearest-upsamples each deep-supervision head to the
   input size and compares it against the FULL-resolution label, so the 1/8 and
   1/16 heads are asked to reproduce 4-voxel gaps they cannot represent —
   filling the gap in is their only way to lower the loss.

CROP_SIZE must match the TRAIN_CROP entry for this channel in
``_constants.py``, which is what ``segment_batch.py`` infers with: DynUNet uses
InstanceNorm, whose statistics span the whole window, so inferring at a
different window size than training shifts every normalisation statistic.

Hyper-parameters live at the top of this file — edit them directly for a run.
"""

import argparse
import os
import sys
from pathlib import Path

# Must be set before torch initialises CUDA. Recovers ~4 GiB of allocator
# fragmentation at this batch/crop size (measured 20.2 -> 16.3 GiB reserved)
# and is ~20% faster.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pytorch_lightning as pl  # noqa: E402
from monai.data import DataLoader  # noqa: E402
from monai.transforms import (  # noqa: E402
    CenterSpatialCropd,
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ToTensord,
)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # noqa: E402
from pytorch_lightning.loggers import TensorBoardLogger  # noqa: E402

from morphospaces.datasets import StandardHDF5Dataset  # noqa: E402
from morphospaces.networks.semantic_unet import SemanticDynSkelLossUNet  # noqa: E402
from morphospaces.transforms.image import ExpandDimsd  # noqa: E402
from morphospaces.transforms.label import LabelsAsLong  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _constants import (  # noqa: E402
    CHANNELS,
    PATCH_SIZE,
    TRAIN_CROP,
    patches_dir_for,
)

# ── hyper-parameters ──────────────────────────────────────────────────────────
# lr was screened at batch_size=2; sqrt-scaled for batch_size=6 (2.5e-5 * sqrt(3)).
# If training looks unstable early, put it back to 2.5e-5.
lr = 4.3e-5
skeleton_recall_factor = 0.5
dropout_rate = 0.025
batch_size = 6
lr_scheduler_step = 50
max_epochs = 500000
# Measured on an A5000 through the real training loop: this config peaks at
# 16.3 GiB reserved of 23.5, at 0.83 s/step. batch_size=8 at this crop OOMs.
precision = "bf16-mixed"
num_workers = 8
# how many deep-supervision heads to supervise (MONAI's implicit default is 4)
deep_supr_num = 2
# ─────────────────────────────────────────────────────────────────────────────

# ── data ──────────────────────────────────────────────────────────────────────
# One patch per file; the random crop below is what varies the window.
patch_shape = (PATCH_SIZE,) * 3
patch_stride = (PATCH_SIZE,) * 3
patch_threshold = 0

image_key = "image_normalized"
labels_key = "labels"
skeleton_key = "tubular_skeleton"
# ─────────────────────────────────────────────────────────────────────────────

# ── logging ───────────────────────────────────────────────────────────────────
log_every_n_iterations = 5
log_image_every_n = 25
val_check_interval = 1.0
# ─────────────────────────────────────────────────────────────────────────────


def logdir_for(channel, crop_size):
    """Run directory, named after the settings that distinguish one run."""
    lr_string = f"{lr:.0e}".replace("-", "m").replace("+", "p")
    name = (
        f"log_{channel}"
        f"_lr_{lr_string}"
        f"_dr_{str(dropout_rate).replace('0.', '')}"
        f"_bs_{batch_size}"
        f"_crop{crop_size[0]}"
        f"_rec_{f'{skeleton_recall_factor:.1f}'.replace('.', 'p')}"
        f"_ds_{deep_supr_num}"
    )
    return str(patches_dir_for(channel).parent / name)


def build_transforms(crop_size):
    """Training and validation transform pipelines."""
    shared_keys = [image_key, labels_key, skeleton_key]
    # per-key interpolation: the image must not be nearest-neighbour resampled
    affine_mode = ("bilinear", "nearest", "nearest")

    train_transform = Compose(
        [
            LabelsAsLong(keys=labels_key),
            ExpandDimsd(keys=shared_keys),
            # the point of the large patches: a different crop every time
            RandSpatialCropd(keys=shared_keys, roi_size=crop_size, random_size=False),
            RandFlipd(keys=shared_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=shared_keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=shared_keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=shared_keys, prob=0.3, spatial_axes=(0, 1)),
            RandRotate90d(keys=shared_keys, prob=0.3, spatial_axes=(0, 2)),
            RandRotate90d(keys=shared_keys, prob=0.3, spatial_axes=(1, 2)),
            RandAffined(
                keys=shared_keys,
                prob=0.2,
                mode=affine_mode,
                rotate_range=(0.5, 0.5, 0.5),
                translate_range=(10, 10, 10),
                scale_range=0.1,
            ),
            RandAdjustContrastd(keys=image_key, prob=0.2),
            RandScaleIntensityd(keys=image_key, factors=0.2, prob=0.2),
            RandShiftIntensityd(keys=image_key, offsets=0.1, prob=0.2),
            RandGaussianNoised(keys=image_key, prob=0.15, std=0.02),
            # plain tensors: MetaTensor bookkeeping costs GPU memory and buys
            # nothing here, since nothing downstream reads the metadata
            ToTensord(keys=shared_keys, track_meta=False),
        ]
    )

    # deterministic centre crop so val_loss is comparable across epochs, and at
    # the same size as training so the InstanceNorm statistics match
    val_transform = Compose(
        [
            LabelsAsLong(keys=labels_key),
            ExpandDimsd(keys=shared_keys),
            CenterSpatialCropd(keys=shared_keys, roi_size=crop_size),
            ToTensord(keys=shared_keys, track_meta=False),
        ]
    )
    return shared_keys, train_transform, val_transform


def build_loader(pattern, shared_keys, transform, shuffle):
    """Dataset + DataLoader over one glob of patch files."""
    dataset = StandardHDF5Dataset.from_glob_pattern(
        glob_pattern=pattern,
        dataset_keys=shared_keys,
        transform=transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_ignore_index=(0,),
        patch_filter_key=labels_key,
        patch_threshold=patch_threshold,
        patch_slack_acceptance=0,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,
    )
    return dataset, loader


def main(channel: str, crop_size=None) -> None:
    """Train a segmentation model on one channel's patch set.

    Parameters
    ----------
    channel : str
        Channel to train on; selects the patch directory and the log directory.
    crop_size : tuple of int, optional
        Network input size. Defaults to the channel's TRAIN_CROP entry, which
        is what ``segment_batch.py`` will infer with.
    """
    crop_size = tuple(crop_size) if crop_size else tuple(TRAIN_CROP[channel])
    if any(c > PATCH_SIZE for c in crop_size):
        raise ValueError(
            f"crop_size {crop_size} exceeds the patch size on disk ({PATCH_SIZE})"
        )

    patches_dir = patches_dir_for(channel)
    train_data_pattern = str(patches_dir / "training" / "*.h5")
    val_data_pattern = str(patches_dir / "validation" / "*.h5")
    logdir_path = logdir_for(channel, crop_size)

    pl.seed_everything(42, workers=True)

    print("Training parameters:")
    print(f"  channel               : {channel}")
    print(f"  lr                    : {lr}")
    print(f"  skeleton_recall_factor: {skeleton_recall_factor}")
    print(f"  dropout_rate          : {dropout_rate}")
    print(f"  batch_size            : {batch_size}")
    print(f"  precision             : {precision}")
    print(f"  num_workers           : {num_workers}")
    print(f"  deep_supr_num         : {deep_supr_num}")
    print(f"  patch on disk         : {patch_shape}")
    print(f"  random crop           : {crop_size}")
    print(f"  logdir                : {logdir_path}")
    print(f"  train pattern         : {train_data_pattern}")
    print(f"  val pattern           : {val_data_pattern}")

    shared_keys, train_transform, val_transform = build_transforms(crop_size)
    train_ds, train_loader = build_loader(
        train_data_pattern, shared_keys, train_transform, shuffle=True
    )
    val_ds, val_loader = build_loader(
        val_data_pattern, shared_keys, val_transform, shuffle=False
    )
    print(f"  train patches         : {len(train_ds)}")
    print(f"  val patches           : {len(val_ds)}")

    net = SemanticDynSkelLossUNet(
        in_channels=1,
        out_channels=2,
        learning_rate=lr,
        lr_scheduler_interval="epoch",
        lr_scheduler_step=lr_scheduler_step,
        dropout_rate=dropout_rate,
        deep_supr_num=deep_supr_num,
        image_key=image_key,
        labels_key=labels_key,
        skeleton_key=skeleton_key,
        skeleton_recall_factor=skeleton_recall_factor,
        log_image_every_n=log_image_every_n,
    )

    best_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="seg-best",
    )
    last_checkpoint = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        dirpath=logdir_path,
        every_n_epochs=1,
        filename="seg-last",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(save_dir=logdir_path, name="lightning_logs")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=precision,
        callbacks=[best_checkpoint, last_checkpoint, lr_monitor],
        logger=logger,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_iterations,
        val_check_interval=val_check_interval,
    )

    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channel", default=CHANNELS[0], choices=list(CHANNELS),
        help="Channel to train on (selects the patch set).",
    )
    parser.add_argument(
        "--crop-size", type=int, default=None,
        help="Network input size. Defaults to the channel's TRAIN_CROP entry.",
    )
    args = parser.parse_args()

    main(
        channel=args.channel,
        crop_size=(args.crop_size,) * 3 if args.crop_size else None,
    )
