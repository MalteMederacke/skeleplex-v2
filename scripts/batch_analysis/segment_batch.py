"""Run segmentation inference on every zarr in a folder, keep large connected
components, binarize, and write the result back into each zarr container.

Input:  <ZARR_DIR>/<name>.zarr/    zarr group with:
            image            (Z, Y, X)  raw intensities
            label            (optional) manual annotation, left untouched
            .attrs: {voxel_size_um: [z, y, x]}
Output: same container, arrays added in place:
            segmentation           (Z, Y, X)  uint8, binary 0/1 (full volume)
            image_cropped          cropped image (if cropping enabled)
            segmentation_cropped   cropped binary segmentation
            .attrs: {segmentation: {...params...}}

Sparse volumes (mostly background) are cropped to a padded bounding box around
the sample before inference, which is much faster; the result is scattered back
into a full-size array so it stays aligned with image/label. The cropped image
and segmentation are also stored so downstream steps can work on the smaller
arrays. Set ``CROP_ENABLED = False`` in ``_constants.py`` to segment the full
volume.

The image is passed to the network without intensity normalisation — match this
to how the training patches were prepared.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage as ndi
from skimage.measure import block_reduce
from skimage.morphology import label

sys.path.insert(0, str(Path(__file__).parent))
from _constants import (  # noqa: E402
    CROP_ENABLED,
    CROP_FACTOR,
    CROP_MARGIN,
    CROP_MIN_BLOB_FRAC,
    CROP_THR_FRAC,
    DEFAULT_VOXEL_SIZE_UM,
    IMAGE_KEY,
    INFERENCE_KWARGS,
    MIN_COMPONENT_SIZE,
    SEG_CHECKPOINT,
    ZARR_DIR,
)
from inference import run_inference  # noqa: E402


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


def process_zarr(zarr_path, checkpoint, image_key, min_component_size, crop_enabled):
    grp = zarr.open(str(zarr_path), mode="a")
    if image_key not in grp:
        print(f"  no '{image_key}' array in {zarr_path.name}, skipping")
        return False

    image = grp[image_key][:]
    voxel_size_um = list(grp.attrs.get("voxel_size_um", list(DEFAULT_VOXEL_SIZE_UM)))

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
        run_inference(crop, checkpoint_path=str(checkpoint), **INFERENCE_KWARGS),
        min_component_size,
    )
    # scatter back into a full-size mask so it stays aligned with image/label
    seg = np.zeros(image.shape, dtype=np.uint8)
    seg[bbox] = seg_crop

    chunks = grp[image_key].chunks
    grp.create_array("segmentation", data=seg, chunks=chunks, overwrite=True)
    if crop_enabled:
        crop_chunks = tuple(min(c, s) for c, s in zip(chunks, crop.shape))
        grp.create_array(
            "image_cropped", data=crop, chunks=crop_chunks, overwrite=True
        )
        grp.create_array(
            "segmentation_cropped", data=seg_crop, chunks=crop_chunks, overwrite=True
        )
    grp.attrs["voxel_size_um"] = voxel_size_um
    grp.attrs["segmentation"] = {
        "checkpoint": str(checkpoint),
        "min_component_size": min_component_size,
        "crop_bbox": [[int(s.start), int(s.stop)] for s in bbox],
        **INFERENCE_KWARGS,
    }
    print(f"  saved 'segmentation' -> {zarr_path.name}  (fg voxels={int(seg.sum())})")
    return True


def main(
    root: Path,
    checkpoint: Path,
    image_key: str = IMAGE_KEY,
    min_component_size: int = MIN_COMPONENT_SIZE,
    crop_enabled: bool = CROP_ENABLED,
    overwrite: bool = False,
) -> None:
    """Segment every zarr container in ``root``.

    Parameters
    ----------
    root : Path
        Directory of ``<name>.zarr`` groups.
    checkpoint : Path
        Segmentation model checkpoint (``.ckpt``).
    image_key : str
        Array key inside each zarr holding the raw image.
    min_component_size : int
        Discard connected components smaller than this many voxels.
    crop_enabled : bool
        Crop to a padded bounding box around the sample before inference.
    overwrite : bool
        Re-segment containers that already have a ``segmentation`` array.
    """
    zarrs = sorted(p for p in root.glob("*.zarr") if p.is_dir())
    if not zarrs:
        print(f"No *.zarr containers found in {root}")
        sys.exit(1)
    print(f"Found {len(zarrs)} zarr containers.\n")

    errors = []
    for path in zarrs:
        grp = zarr.open(str(path), mode="r")
        if "segmentation" in grp and not overwrite:
            print(f"[skip] {path.name} (already segmented)")
            continue
        print(path.name)
        try:
            process_zarr(path, checkpoint, image_key, min_component_size, crop_enabled)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((path, e))

    print(f"\nDone. {len(zarrs) - len(errors)}/{len(zarrs)} containers segmented.")
    if errors:
        print("Errors:")
        for p, e in errors:
            print(f"  {p.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ZARR_DIR)
    parser.add_argument("--checkpoint", type=Path, default=SEG_CHECKPOINT)
    parser.add_argument("--image-key", default=IMAGE_KEY)
    parser.add_argument("--min-component-size", type=int, default=MIN_COMPONENT_SIZE)
    parser.add_argument(
        "--no-crop", action="store_true", help="Segment the full volume (no cropping)."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-segment containers that already have a 'segmentation' array.",
    )
    args = parser.parse_args()

    main(
        root=args.root,
        checkpoint=args.checkpoint,
        image_key=args.image_key,
        min_component_size=args.min_component_size,
        crop_enabled=CROP_ENABLED and not args.no_crop,
        overwrite=args.overwrite,
    )
