"""Train a DynUNet with skeleton-recall loss on the HDF5 patches prepared by
``prepare_patches.py``.

Data patterns and the log directory are derived from ``PATCHES_DIR`` in
``_constants.py``. Hyper-parameters live at the top of this file — edit them
directly for a training run.
"""

import sys
from pathlib import Path

import pytorch_lightning as pl
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandRotate90d,
)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from morphospaces.datasets import StandardHDF5Dataset
from morphospaces.networks.semantic_unet import SemanticDynSkelLossUNet
from morphospaces.transforms.image import ExpandDimsd
from morphospaces.transforms.label import LabelsAsLong

sys.path.insert(0, str(Path(__file__).parent))
from _constants import PATCHES_DIR  # noqa: E402

# ── hyper-parameters ──────────────────────────────────────────────────────────
lr = 2.5e-5
skeleton_recall_factor = 0.5
dropout_rate = 0.025
batch_size = 2
lr_scheduler_step = 50
max_epochs = 500000
# ─────────────────────────────────────────────────────────────────────────────

# ── data ──────────────────────────────────────────────────────────────────────
# stride == patch_size means one patch per file (patches are pre-cut on disk)
patch_shape = (128, 128, 128)
patch_stride = (128, 128, 128)
patch_threshold = 0

image_key = "image_normalized"
labels_key = "labels"
skeleton_key = "tubular_skeleton"

train_data_pattern = str(PATCHES_DIR / "training" / "*.h5")
val_data_pattern = str(PATCHES_DIR / "validation" / "*.h5")
# ─────────────────────────────────────────────────────────────────────────────

# ── logging ───────────────────────────────────────────────────────────────────
log_every_n_iterations = 5
log_image_every_n = 25
val_check_interval = 1.0

lr_string = f"{lr:.0e}".replace("-", "m").replace("+", "p")
logdir_path = str(
    PATCHES_DIR.parent
    / (
        f"log_dynunet_lr_{lr_string}"
        f"_dr_{str(dropout_rate).replace('0.', '')}"
        f"_bs_{batch_size}"
        f"_rec_{f'{skeleton_recall_factor:.1f}'.replace('.', 'p')}"
    )
)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pl.seed_everything(42, workers=True)

    print("Training parameters:")
    print(f"  lr                    : {lr}")
    print(f"  skeleton_recall_factor: {skeleton_recall_factor}")
    print(f"  dropout_rate          : {dropout_rate}")
    print(f"  batch_size            : {batch_size}")
    print(f"  logdir                : {logdir_path}")
    print(f"  train pattern         : {train_data_pattern}")
    print(f"  val pattern           : {val_data_pattern}")

    shared_keys = [image_key, labels_key, skeleton_key]

    train_transform = Compose(
        [
            LabelsAsLong(keys=labels_key),
            ExpandDimsd(keys=shared_keys),
            RandAdjustContrastd(keys=image_key, prob=0.05),
            RandFlipd(keys=shared_keys, prob=0.05),
            RandRotate90d(keys=shared_keys, prob=0.05, spatial_axes=(0, 1)),
            RandRotate90d(keys=shared_keys, prob=0.05, spatial_axes=(0, 2)),
            RandRotate90d(keys=shared_keys, prob=0.05, spatial_axes=(1, 2)),
            RandAffined(
                keys=shared_keys,
                prob=0.05,
                mode="nearest",
                rotate_range=(0.5, 0.5, 0.5),
                translate_range=(10, 10, 10),
                scale_range=0.1,
            ),
        ]
    )

    val_transform = Compose(
        [
            LabelsAsLong(keys=labels_key),
            ExpandDimsd(keys=shared_keys),
        ]
    )

    train_ds = StandardHDF5Dataset.from_glob_pattern(
        glob_pattern=train_data_pattern,
        dataset_keys=shared_keys,
        transform=train_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_ignore_index=(0,),
        patch_filter_key=labels_key,
        patch_threshold=patch_threshold,
        patch_slack_acceptance=0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_ds = StandardHDF5Dataset.from_glob_pattern(
        glob_pattern=val_data_pattern,
        dataset_keys=shared_keys,
        transform=val_transform,
        patch_shape=patch_shape,
        stride_shape=patch_stride,
        patch_filter_ignore_index=(0,),
        patch_filter_key=labels_key,
        patch_threshold=patch_threshold,
        patch_slack_acceptance=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    net = SemanticDynSkelLossUNet(
        in_channels=1,
        out_channels=2,
        learning_rate=lr,
        lr_scheduler_interval="epoch",
        lr_scheduler_step=lr_scheduler_step,
        dropout_rate=dropout_rate,
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
        callbacks=[best_checkpoint, last_checkpoint, lr_monitor],
        logger=logger,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_iterations,
        val_check_interval=val_check_interval,
    )

    trainer.fit(
        net,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
