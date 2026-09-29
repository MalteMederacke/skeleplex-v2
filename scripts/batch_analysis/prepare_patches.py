"""Extract 3-D training patches from a segmentation zarr and append them to a
train/validation patch set for ``train_dynunet.py``.

Reads one zarr group holding an image array and a binary label mask. Overlapping
``PATCH_SIZE`` cubes are extracted on a regular grid (``PATCH_STRIDE``), patches
with too little foreground are dropped, and the rest are split
train/validation and written as HDF5 files:

    <out>/training/<zarr-name>_patch_XXXX.h5
    <out>/validation/<zarr-name>_patch_XXXX.h5

New patches continue the existing ``patch_XXXX`` numbering, so calling this
script on several zarrs accumulates one combined patch set without overwriting
earlier patches.

The label mask has no skeleton channel, so the ``tubular_skeleton`` target is
derived on the fly as a dilated morphological skeleton of the mask. Each patch
holds the keys the training script expects:

    image_normalized  – float32 image
    labels            – uint8 label mask
    tubular_skeleton  – uint8 dilated skeleton
"""

import argparse
import os
import random
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr
from skimage.morphology import ball, binary_dilation, skeletonize

sys.path.insert(0, str(Path(__file__).parent))
from _constants import (  # noqa: E402
    IMAGE_KEY,
    LABEL_KEY,
    MIN_FOREGROUND_FRACTION,
    PATCH_SIZE,
    PATCH_STRIDE,
    PATCHES_DIR,
    RANDOM_SEED,
    SKELETON_DILATION_RADIUS,
    TRAIN_FRACTION,
)


def extract_patch_starts(shape, patch_size, stride):
    """List of (z0, y0, x0) start indices for every patch on the grid."""
    d, h, w = shape
    starts = []
    for z in range(0, d - patch_size + 1, stride):
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                starts.append((z, y, x))
    return starts


def tubular_skeleton_from_labels(labels, radius):
    """Dilated morphological skeleton of the binary mask (uint8)."""
    skel = skeletonize(labels > 0)
    if radius > 0:
        skel = binary_dilation(skel, ball(radius))
    return skel.astype(np.uint8)


def next_patch_index(out_dir):
    """Smallest free ``patch_XXXX`` index in ``out_dir`` (append, not overwrite)."""
    if not os.path.isdir(out_dir):
        return 0
    idxs = [
        int(m.group(1))
        for fn in os.listdir(out_dir)
        if (m := re.search(r"patch_(\d+)\.h5$", fn))
    ]
    return max(idxs) + 1 if idxs else 0


def main(
    source_zarr: Path,
    out_dir: Path,
    image_key: str = IMAGE_KEY,
    label_key: str = LABEL_KEY,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_STRIDE,
    min_foreground_fraction: float = MIN_FOREGROUND_FRACTION,
    skeleton_dilation_radius: int = SKELETON_DILATION_RADIUS,
    train_fraction: float = TRAIN_FRACTION,
    seed: int = RANDOM_SEED,
) -> None:
    """Extract patches from one zarr and append them to the patch set.

    Parameters
    ----------
    source_zarr : Path
        Zarr group holding the image and label arrays.
    out_dir : Path
        Patch-set root; ``training/`` and ``validation/`` are created inside it.
    image_key, label_key : str
        Array keys inside the zarr for the image and the binary mask.
    patch_size, stride : int
        Cube size and grid stride, in voxels.
    min_foreground_fraction : float
        Keep a patch only if at least this fraction of its voxels is foreground.
    skeleton_dilation_radius : int
        Ball radius used to dilate the derived skeleton (0 = no dilation).
    train_fraction : float
        Fraction of kept patches assigned to the training split.
    seed : int
        Seed for the shuffle / split.
    """
    random.seed(seed)
    np.random.seed(seed)

    name = Path(source_zarr).name.replace(".zarr", "")
    print(f"Loading {source_zarr} ...")
    root = zarr.open(str(source_zarr), mode="r")
    labels = np.asarray(root[label_key]).astype(np.uint8)
    image = np.asarray(root[image_key]).astype(np.float32)
    print(f"  image shape : {image.shape}  labels shape: {labels.shape}")

    print("Computing dilated skeleton ...")
    tubular_skeleton = tubular_skeleton_from_labels(labels, skeleton_dilation_radius)

    all_starts = extract_patch_starts(labels.shape, patch_size, stride)
    print(f"Candidate patches: {len(all_starts)}")

    threshold = min_foreground_fraction * patch_size ** 3
    kept = []
    for (z, y, x) in all_starts:
        sl = (
            slice(z, z + patch_size),
            slice(y, y + patch_size),
            slice(x, x + patch_size),
        )
        if labels[sl].sum() >= threshold:
            kept.append((z, y, x))
    print(
        f"Patches after foreground filter "
        f"(>= {min_foreground_fraction * 100:.1f}%): {len(kept)}"
    )

    random.shuffle(kept)
    n_train = int(len(kept) * train_fraction)
    splits = [("training", kept[:n_train]), ("validation", kept[n_train:])]
    print(f"Train: {len(splits[0][1])}  |  Validation: {len(splits[1][1])}")

    for split, starts in splits:
        split_dir = os.path.join(str(out_dir), split)
        os.makedirs(split_dir, exist_ok=True)
        start_idx = next_patch_index(split_dir)
        for offset, (z, y, x) in enumerate(starts):
            sl = (
                slice(z, z + patch_size),
                slice(y, y + patch_size),
                slice(x, x + patch_size),
            )
            out_path = os.path.join(
                split_dir, f"{name}_patch_{start_idx + offset:04d}.h5"
            )
            with h5py.File(out_path, "w") as f:
                f.create_dataset("image_normalized", data=image[sl], compression="gzip")
                f.create_dataset("labels", data=labels[sl], compression="gzip")
                f.create_dataset(
                    "tubular_skeleton", data=tubular_skeleton[sl], compression="gzip"
                )
        if starts:
            print(
                f"  Appended {len(starts)} patches to {split_dir}/ "
                f"({name}_patch_{start_idx:04d}.."
                f"{name}_patch_{start_idx + len(starts) - 1:04d})"
            )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Source zarr group."
    )
    parser.add_argument(
        "--out", type=Path, default=PATCHES_DIR,
        help=f"Patch-set root. Default: {PATCHES_DIR}",
    )
    parser.add_argument("--image-key", default=IMAGE_KEY)
    parser.add_argument("--label-key", default=LABEL_KEY)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    parser.add_argument(
        "--min-foreground-fraction", type=float, default=MIN_FOREGROUND_FRACTION
    )
    parser.add_argument(
        "--skeleton-dilation-radius", type=int, default=SKELETON_DILATION_RADIUS
    )
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    main(
        source_zarr=args.source,
        out_dir=args.out,
        image_key=args.image_key,
        label_key=args.label_key,
        patch_size=args.patch_size,
        stride=args.stride,
        min_foreground_fraction=args.min_foreground_fraction,
        skeleton_dilation_radius=args.skeleton_dilation_radius,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
