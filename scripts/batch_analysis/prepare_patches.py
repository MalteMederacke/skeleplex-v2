"""Cut 3-D training patches from curated volumes for ``train_dynunet.py``.

Input:  PATCH_SOURCES — hand-curated .zarr or .h5 volumes (image + label mask)
Output: <PATCHES_DIR>_<channel>/{training,validation}/patch_XXXX.h5

Each output patch has the keys the training script expects::

    image_normalized  float32 image, percentile-normalised per source volume
    labels            uint8 binary segmentation mask
    tubular_skeleton  uint8 dilated skeleton derived from the labels

Why large patches plus random cropping
--------------------------------------
Patches on disk are PATCH_SIZE^3 and the training script takes a random
CROP_SIZE^3 crop out of each one (RandSpatialCropd in ``train_dynunet.py``).

An earlier version cut patches at the network's input size and trained with
patch_shape == stride, so every file yielded exactly ONE fixed crop. The network
saw each dense region at a single offset for its whole training run and never
learned to be invariant to where the sliding window lands. At inference the
window lands wherever it lands, and in dense regions an unlucky offset makes the
network bridge neighbouring branches.

At PATCH_SIZE=192 with a 128^3 random crop there are (192-128)^3 = 262144
distinct offsets per patch instead of one. Keep PATCH_SIZE - CROP_SIZE at about
64: that span is what buys the invariance. Both sizes must stay divisible by 32,
the product of the DynUNet strides.

Only hand-curated segmentations belong in PATCH_SOURCES. Training on raw model
output teaches the new network to reproduce the old one's mistakes, merged
branches included.

The ``tubular_skeleton`` target is computed on the FULL volume before cutting,
so the skeleton stays consistent across patch borders.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr
from skimage.morphology import ball, binary_dilation, skeletonize

sys.path.insert(0, str(Path(__file__).parent))
from _constants import (  # noqa: E402
    CLEAN_PATCH_OUTPUT,
    MIN_FOREGROUND_FRACTION,
    PATCH_SIZE,
    PATCH_SOURCES,
    PATCH_STRIDE,
    RANDOM_SEED,
    SKELETON_DILATION_RADIUS,
    TRAIN_FRACTION,
    patches_dir_for,
)
from normalization import normalize_image  # noqa: E402

SPLITS = ("training", "validation")


def open_source(path):
    """Return (handle, file_to_close) for a .zarr or .h5 source."""
    path = str(path)
    if path.endswith(".zarr"):
        return zarr.open(path, mode="r"), None
    f = h5py.File(path, "r")
    return f, f


def patch_starts(shape, patch_size, stride):
    """Start indices covering the volume, with the last row flush to the edge.

    A plain range() drops up to patch_size-1 voxels at each far edge; clamping
    the final start keeps the volume boundary in the training set, which is
    where the sliding window is most error-prone at inference.
    """
    starts = []
    for dim in shape:
        if dim < patch_size:
            raise ValueError(f"volume axis {dim} smaller than patch {patch_size}")
        s = list(range(0, dim - patch_size + 1, stride))
        if s[-1] != dim - patch_size:
            s.append(dim - patch_size)
        starts.append(s)
    return [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]


def tubular_skeleton_from_labels(labels, radius):
    """Dilated morphological skeleton of the binary mask (uint8)."""
    skel = skeletonize(labels > 0)
    if radius > 0:
        skel = binary_dilation(skel, ball(radius))
    return skel.astype(np.uint8)


def write_patch(out_dir, index, image, labels, skeleton, sl):
    """Write one patch file with the three datasets the training script reads."""
    out_path = Path(out_dir) / f"patch_{index:04d}.h5"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("image_normalized", data=image[sl], compression="gzip")
        f.create_dataset("labels", data=labels[sl], compression="gzip")
        f.create_dataset("tubular_skeleton", data=skeleton[sl], compression="gzip")


def keep_dense_patches(labels, patch_size, stride, min_foreground_fraction):
    """Patch starts whose label mask is at least ``min_foreground_fraction`` full."""
    starts = patch_starts(labels.shape, patch_size, stride)
    threshold = min_foreground_fraction * patch_size ** 3
    return [
        (z, y, x)
        for (z, y, x) in starts
        if labels[
            z:z + patch_size, y:y + patch_size, x:x + patch_size
        ].sum() >= threshold
    ]


def main(
    sources=None,
    patch_size: int = PATCH_SIZE,
    stride: int = PATCH_STRIDE,
    min_foreground_fraction: float = MIN_FOREGROUND_FRACTION,
    train_fraction: float = TRAIN_FRACTION,
    clean: bool = CLEAN_PATCH_OUTPUT,
    seed: int = RANDOM_SEED,
) -> None:
    """Cut training patches from every curated source volume.

    The train/validation split is made PER SOURCE, so every source volume is
    represented in both splits — a global split can put a whole volume in
    validation and leave the network blind to it.

    Parameters
    ----------
    sources : list of dict, optional
        Curated source volumes; defaults to PATCH_SOURCES. Each entry has
        ``path``, ``label_key`` and ``images`` ({channel: dataset key}).
    patch_size : int
        Size of the cubic patches written to disk.
    stride : int
        Step between neighbouring patch origins.
    min_foreground_fraction : float
        Keep a patch only if at least this fraction of it is foreground.
    train_fraction : float
        Fraction of each source's patches assigned to training.
    clean : bool
        Delete the output directories first. Appending to a set cut at a
        different patch_size mixes sizes and breaks the random crop.
    seed : int
        Seed for the per-source shuffle.
    """
    sources = sources if sources is not None else PATCH_SOURCES
    if not sources:
        print(
            "PATCH_SOURCES is empty — list your curated volumes in the config "
            "(path, label_key, images) before running."
        )
        sys.exit(1)

    random.seed(seed)
    np.random.seed(seed)

    channels = sorted({ch for src in sources for ch in src["images"]})
    out_dirs = {ch: patches_dir_for(ch) for ch in channels}

    for base in out_dirs.values():
        if clean and base.is_dir():
            print(f"Removing existing {base}")
            shutil.rmtree(base)
        for split in SPLITS:
            (base / split).mkdir(parents=True, exist_ok=True)

    counters = {(ch, split): 0 for ch in channels for split in SPLITS}
    totals = {(ch, split): 0 for ch in channels for split in SPLITS}

    print(f"{len(sources)} source volume(s), channels {channels}\n")
    for src in sources:
        path = Path(src["path"])
        print(f"-- {path.name}")
        root, fh = open_source(path)
        try:
            labels = (np.asarray(root[src["label_key"]]) > 0).astype(np.uint8)

            if min(labels.shape) < patch_size:
                print(
                    f"   [skip] shape {labels.shape} smaller than "
                    f"patch_size={patch_size}"
                )
                continue

            kept = keep_dense_patches(
                labels, patch_size, stride, min_foreground_fraction
            )
            n_candidates = len(patch_starts(labels.shape, patch_size, stride))
            print(
                f"   shape {labels.shape}  fg {labels.mean() * 100:.1f}%  "
                f"candidates {n_candidates} -> kept {len(kept)}"
            )
            if not kept:
                continue

            skeleton = tubular_skeleton_from_labels(labels, SKELETON_DILATION_RADIUS)

            random.shuffle(kept)
            n_train = int(round(len(kept) * train_fraction))
            splits = [("training", kept[:n_train]), ("validation", kept[n_train:])]

            for channel, image_key in sorted(src["images"].items()):
                image = normalize_image(np.asarray(root[image_key]))
                for split, subset in splits:
                    out_dir = out_dirs[channel] / split
                    for (z, y, x) in subset:
                        sl = (
                            slice(z, z + patch_size),
                            slice(y, y + patch_size),
                            slice(x, x + patch_size),
                        )
                        write_patch(
                            out_dir, counters[(channel, split)],
                            image, labels, skeleton, sl,
                        )
                        counters[(channel, split)] += 1
                    totals[(channel, split)] += len(subset)
                del image
        finally:
            if fh is not None:
                fh.close()

    print()
    for channel in channels:
        print(
            f"{channel:8s} -> {out_dirs[channel]}   "
            f"training {totals[(channel, 'training')]}   "
            f"validation {totals[(channel, 'validation')]}"
        )
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    parser.add_argument(
        "--min-foreground-fraction", type=float, default=MIN_FOREGROUND_FRACTION
    )
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Append to the existing patch set instead of wiping it first.",
    )
    args = parser.parse_args()

    main(
        patch_size=args.patch_size,
        stride=args.stride,
        min_foreground_fraction=args.min_foreground_fraction,
        train_fraction=args.train_fraction,
        clean=CLEAN_PATCH_OUTPUT and not args.no_clean,
    )
