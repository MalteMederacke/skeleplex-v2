"""Run segmentation inference on every sample, keep large connected components,
binarize, and write the result into a zarr container.

Input   (flat layout)   <ZARR_DIR>/<name>.zarr          image array + voxel_size_um
        (nested layout) <condition>/<ISO_SUBDIR>/*.h5   one dataset per channel
Output  (flat layout)   the same container, arrays added in place
        (nested layout) <condition>/segmentation_<channel>/<stem>.zarr

Arrays written:
    <channel>             (Z, Y, X)  float32, the normalised image
    segmentation          (Z, Y, X)  uint8, binary 0/1 (full volume)
    image_cropped         cropped image (if cropping enabled)
    segmentation_cropped  cropped binary segmentation
    .attrs: {voxel_size_um: [z, y, x], segmentation: {...params...}}

Sparse volumes (mostly background) are cropped to a padded bounding box around
the sample before inference, which is much faster; the result is scattered back
into a full-size array so it stays aligned with the image. The cropped image and
segmentation are also stored so downstream steps can work on the smaller arrays.
Set ``CROP_ENABLED = False`` to segment the full volume.

The image is normalised with ``normalization.normalize_image`` — the same
function ``prepare_patches.py`` applies — so the network sees the intensity
distribution it was trained on.

Each channel is segmented with its own checkpoint, at the roi_size that
checkpoint was trained on (see TRAIN_CROP in ``_constants.py``).

Containers that already hold a complete segmentation are skipped, so an
interrupted run can simply be restarted; ``--overwrite`` re-segments everything,
which is what you want after switching to a new checkpoint.
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage as ndi
from skimage.measure import block_reduce
from skimage.morphology import label

sys.path.insert(0, str(Path(__file__).parent))
from _constants import (  # noqa: E402
    CHANNELS,
    CROP_ENABLED,
    CROP_FACTOR,
    CROP_MARGIN,
    CROP_MIN_BLOB_FRAC,
    CROP_THR_FRAC,
    DEFAULT_VOXEL_SIZE_UM,
    MIN_COMPONENT_SIZE,
    checkpoint_for,
    inference_kwargs,
)
from _layout import find_inputs, rel  # noqa: E402
from inference import run_inference  # noqa: E402
from normalization import normalize_image  # noqa: E402


def _chunks(shape, base=(64, 256, 256)):
    return tuple(min(b, s) for b, s in zip(base, shape))


# ── reading the source image ─────────────────────────────────────────────────

def source_shape(sample):
    """Shape of the sample's image, read from the file header only."""
    if sample.source.suffix == ".h5":
        import h5py

        with h5py.File(sample.source, "r") as f:
            return tuple(f[sample.source_key].shape)
    grp = zarr.open(str(sample.source), mode="r")
    return tuple(grp[sample.source_key].shape)


def read_source(sample):
    """Return (image, voxel_size_um) for a sample, whatever format it is in."""
    if sample.source.suffix == ".h5":
        import h5py

        with h5py.File(sample.source, "r") as f:
            ds = f[sample.source_key]
            image = ds[:]
            voxel_size_um = list(
                ds.attrs.get("voxel_size_um", list(DEFAULT_VOXEL_SIZE_UM))
            )
        return image, voxel_size_um

    grp = zarr.open(str(sample.source), mode="r")
    image = grp[sample.source_key][:]
    voxel_size_um = list(
        grp.attrs.get("voxel_size_um", list(DEFAULT_VOXEL_SIZE_UM))
    )
    return image, voxel_size_um


def is_complete(zarr_path, expected_shape):
    """True if ``zarr_path`` holds a finished segmentation of the expected shape.

    A run interrupted mid-write leaves a zarr directory that exists but has no
    'segmentation' array, or one of the wrong shape. Testing only for the
    directory would skip such a sample forever, so check the contents.
    """
    if not Path(zarr_path).exists():
        return False
    try:
        root = zarr.open(str(zarr_path), mode="r")
        if "segmentation" not in root:
            return False
        seg = root["segmentation"]
        if tuple(seg.shape) != tuple(expected_shape):
            return False
        seg[-1, -1, -1]  # touch the last chunk: a half-written array raises here
    except Exception:
        return False
    return True


# ── segmentation ─────────────────────────────────────────────────────────────

def foreground_bbox(image, factor, margin, thr_frac, min_blob_frac):
    """Conservative bounding box (tuple of slices) around the dense sample.

    Downsampling averages away isolated background specks; thresholding and
    keeping only sizeable connected components isolates the sample from noise;
    the box is then padded by ``margin`` full-resolution voxels.
    """
    small = block_reduce(image, (factor,) * image.ndim, np.mean)
    thr = max(1.0, thr_frac * small.max())
    mask = small > thr
    if not mask.any():
        return tuple(slice(0, s) for s in image.shape)  # empty -> no crop
    lbl, _ = ndi.label(mask)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    keep = sizes >= max(min_blob_frac * mask.sum(), 1)
    mask = keep[lbl]
    coords = np.array(np.nonzero(mask))
    lo = coords.min(axis=1)
    hi = coords.max(axis=1) + 1
    slices = []
    for ax in range(image.ndim):
        a = max(int(lo[ax]) * factor - margin, 0)
        b = min(int(hi[ax]) * factor + margin, image.shape[ax])
        slices.append(slice(a, b))
    return tuple(slices)


def size_filter_mask(segmentation_raw, min_component_size):
    """Binary mask keeping all components with >= ``min_component_size`` voxels."""
    labeled = label(segmentation_raw != 0)
    if labeled.max() == 0:
        return np.zeros_like(labeled, dtype=np.uint8)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background
    keep = np.flatnonzero(counts >= min_component_size)
    return np.isin(labeled, keep).astype(np.uint8)


def process_sample(sample, min_component_size, crop_enabled):
    """Segment one sample and write the result to its zarr container."""
    checkpoint = checkpoint_for(sample.channel)
    kwargs = inference_kwargs(sample.channel)

    image, voxel_size_um = read_source(sample)
    image = normalize_image(image)

    if crop_enabled:
        bbox = foreground_bbox(
            image, CROP_FACTOR, CROP_MARGIN, CROP_THR_FRAC, CROP_MIN_BLOB_FRAC
        )
    else:
        bbox = tuple(slice(0, s) for s in image.shape)
    crop = image[bbox]
    speedup = np.prod(image.shape) / max(np.prod(crop.shape), 1)
    print(f"  inference on crop {crop.shape} of {image.shape}  (~{speedup:.0f}x) ...")

    seg_crop = size_filter_mask(
        run_inference(crop, checkpoint_path=str(checkpoint), **kwargs),
        min_component_size,
    )
    # scatter back into a full-size mask so it stays aligned with the image
    seg = np.zeros(image.shape, dtype=np.uint8)
    seg[bbox] = seg_crop

    sample.zarr_path.parent.mkdir(parents=True, exist_ok=True)
    # 'a' rather than 'w': in the flat layout the container already holds the
    # image (and possibly a hand-drawn label) that must not be dropped.
    grp = zarr.open(str(sample.zarr_path), mode="a")

    if sample.source != sample.zarr_path:
        grp.create_array(
            sample.channel,
            data=image.astype(np.float32),
            chunks=_chunks(image.shape),
            overwrite=True,
        )
    grp.create_array(
        "segmentation", data=seg, chunks=_chunks(seg.shape), overwrite=True
    )
    if crop_enabled:
        grp.create_array(
            "image_cropped", data=crop.astype(np.float32),
            chunks=_chunks(crop.shape), overwrite=True,
        )
        grp.create_array(
            "segmentation_cropped", data=seg_crop,
            chunks=_chunks(seg_crop.shape), overwrite=True,
        )
    grp.attrs["voxel_size_um"] = voxel_size_um
    grp.attrs["segmentation"] = {
        "channel": sample.channel,
        "checkpoint": str(checkpoint),
        "min_component_size": min_component_size,
        "crop_bbox": [[int(s.start), int(s.stop)] for s in bbox],
        **{k: list(v) if isinstance(v, tuple) else v for k, v in kwargs.items()},
    }
    print(f"  saved -> {rel(sample.zarr_path)}  (fg voxels={int(seg.sum())})")


def main(
    channels=None,
    min_component_size: int = MIN_COMPONENT_SIZE,
    crop_enabled: bool = CROP_ENABLED,
    overwrite: bool = False,
) -> None:
    """Segment every sample of every requested channel.

    Parameters
    ----------
    channels : sequence of str, optional
        Channels to segment. Defaults to all of CHANNELS.
    min_component_size : int
        Discard connected components smaller than this many voxels.
    crop_enabled : bool
        Crop to a padded bounding box around the sample before inference.
    overwrite : bool
        Re-segment containers that already hold a complete segmentation.
    """
    channels = tuple(channels) if channels else tuple(CHANNELS)

    todo, done = [], []
    for channel in channels:
        for sample in find_inputs(channel):
            if not overwrite and is_complete(sample.zarr_path, source_shape(sample)):
                done.append(sample)
            else:
                todo.append(sample)

    if not todo and not done:
        print(f"No inputs found for channels {channels}.")
        sys.exit(1)

    print(
        f"{len(todo) + len(done)} sample(s) across {len(channels)} channel(s): "
        f"{len(todo)} to segment, {len(done)} already done."
    )
    if overwrite:
        print("--overwrite is on: existing segmentations will be redone.")
    for sample in done:
        print(f"[skip] {rel(sample.zarr_path)}")
    if not todo:
        print("\nNothing to do.")
        return

    print()
    errors = []
    for i, sample in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {rel(sample.source)}  ({sample.channel})")
        try:
            process_sample(sample, min_component_size, crop_enabled)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            errors.append((sample, e))

    print(f"\nDone. {len(todo) - len(errors)}/{len(todo)} samples segmented.")
    if errors:
        print("Errors:")
        for s, e in errors:
            print(f"  {rel(s.source)}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channel", action="append", dest="channels", choices=list(CHANNELS),
        help="Channel to segment (repeatable). Defaults to all configured channels.",
    )
    parser.add_argument("--min-component-size", type=int, default=MIN_COMPONENT_SIZE)
    parser.add_argument(
        "--no-crop", action="store_true", help="Segment the full volume (no cropping)."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        default=os.environ.get("SEGMENT_OVERWRITE", "0") == "1",
        help="Re-segment containers that are already done (SEGMENT_OVERWRITE=1).",
    )
    args = parser.parse_args()

    main(
        channels=args.channels,
        min_component_size=args.min_component_size,
        crop_enabled=CROP_ENABLED and not args.no_crop,
        overwrite=args.overwrite,
    )
