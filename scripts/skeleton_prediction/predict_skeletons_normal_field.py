"""Predict skeletons from a binary segmentation using the background vector field model.

The model was trained on the inward unit normal vector field: a 3-channel field
where each foreground voxel holds a unit vector pointing toward the nearest
boundary.  The field is recomputed here at inference time from the segmentation
(matching the on-the-fly approach used during training).

Input  : binary segmentation zarr, shape (Z, Y, X)
Output : skeleton prediction zarr, float32, shape (Z, Y, X)
"""
# isort: skip_file

import argparse
import gc
import sys
from pathlib import Path

import dask.array as da
import numpy as np
import zarr

from skeleplex.skeleton.distance_field import (
    inward_unit_normal_field_cpu,
    inward_unit_normal_field_gpu,
)
from skeleplex.skeleton._skeletonize import (
    load_normal_field_model,
    skeletonize_from_normal_field,
)
from skeleplex.utils._chunked import iteratively_process_chunks_3d

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import (
    CHECKPOINT_PATH,
    SEGMENTATION_ZARR,
    SKELETON_PREDICTIONS_ZARR,
)


def compute_normal_field_zarr(
    segmentation: np.ndarray,
    output_path: Path | str,
    chunk_shape: tuple[int, int, int] = (128, 128, 128),
    border_shape: tuple[int, int, int] = (32, 32, 32),
    backend: str = "gpu",
) -> None:
    """Compute the inward unit normal field chunkwise and save to disk as zarr.

    Each chunk is expanded by `border_shape` for EDT context; only the core
    result is written back.

    Parameters
    ----------
    segmentation : np.ndarray
        Binary array, shape (Z, Y, X).
    output_path : Path or str
        Destination zarr path. Written with shape (3, Z, Y, X), float32.
    chunk_shape : tuple[int, int, int]
        Core chunk size per iteration. Default (128, 128, 128).
    border_shape : tuple[int, int, int]
        Extra border around each chunk for EDT context. Default (32, 32, 32).
    backend : str
        "gpu" (CuPy) or "cpu" (NumPy/SciPy). Default "gpu".
    """
    field_fn = (
        inward_unit_normal_field_gpu if backend == "gpu"
        else inward_unit_normal_field_cpu
    )
    out_zarr = zarr.open(
        str(output_path),
        mode="w",
        shape=(3,) + segmentation.shape,
        dtype=np.float32,
        chunks=(3,) + chunk_shape,
    )
    seg_dask = da.from_array(segmentation, chunks=chunk_shape)
    iteratively_process_chunks_3d(
        input_array=seg_dask,
        output_zarr=out_zarr,
        function_to_apply=field_fn,
        chunk_shape=chunk_shape,
        extra_border=border_shape,
    )


def main(
    segmentation_zarr: Path,
    output_zarr: Path,
    checkpoint_path: Path,
    normal_field_zarr: Path | None = None,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    overlap: float = 0.5,
    field_backend: str = "gpu",
    chunk_shape: tuple[int, int, int] = (128, 128, 128),
    border_shape: tuple[int, int, int] = (32, 32, 32),
) -> None:
    """Run normal-field skeleton inference on a full segmentation volume.

    Step 1 - compute the inward unit normal field chunkwise and save to zarr.
    Step 2 - load the field and run sliding-window inference.

    Parameters
    ----------
    segmentation_zarr : Path
        Path to the input zarr containing the binary segmentation.
    output_zarr : Path
        Path at which to write the float32 skeleton prediction zarr.
    checkpoint_path : Path
        Path to the model checkpoint (.ckpt).
    normal_field_zarr : Path or None
        Where to save (or load, if it already exists) the normal field zarr.
        Defaults to <output_zarr>.parent / "normal_field.zarr".
    roi_size : tuple[int, int, int]
        Sliding-window tile size.
    overlap : float
        Fractional overlap between tiles.
    field_backend : str
        "gpu" or "cpu" for normal field computation.
    chunk_shape : tuple[int, int, int]
        Core chunk size for chunkwise field computation.
    border_shape : tuple[int, int, int]
        Extra border around each chunk for EDT context.
    """
    if normal_field_zarr is None:
        normal_field_zarr = Path(output_zarr).parent / "normal_field.zarr"

    # --- Step 1: normal field -------------------------------------------------
    if Path(normal_field_zarr).exists():
        print(f"Normal field zarr already exists, skipping: {normal_field_zarr}")
    else:
        print(f"Loading segmentation from {segmentation_zarr}")
        segmentation = np.asarray(
            zarr.open(str(segmentation_zarr), mode="r")[:]
        ).astype(bool)
        print(f"  Segmentation shape: {segmentation.shape}")

        print(
            f"Computing normal field "
            f"(backend={field_backend}, chunk={chunk_shape}, border={border_shape})..."
        )
        compute_normal_field_zarr(
            segmentation=segmentation,
            output_path=normal_field_zarr,
            chunk_shape=chunk_shape,
            border_shape=border_shape,
            backend=field_backend,
        )
        del segmentation
        gc.collect()
        print(f"  Normal field saved to {normal_field_zarr}")

    # --- Step 2: inference ----------------------------------------------------
    print(f"Loading normal field from {normal_field_zarr}")
    normal_field = np.asarray(
        zarr.open(str(normal_field_zarr), mode="r")[:]
    ).astype(np.float32)

    print(f"Loading model from {checkpoint_path}")
    model = load_normal_field_model(checkpoint_path)

    print("Running inference...")
    prediction = skeletonize_from_normal_field(
        normal_field=normal_field,
        model=model,
        roi_size=roi_size,
        overlap=overlap,
    )

    print(f"Saving prediction to {output_zarr}")
    store = zarr.open(
        str(output_zarr),
        mode="w",
        shape=prediction.shape,
        dtype=np.float32,
        chunks=(64, 64, 64),
    )
    store[:] = prediction
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict skeletons using the background vector field model."
    )
    parser.add_argument(
        "--segmentation",
        type=Path,
        default=SEGMENTATION_ZARR,
        help="Path to input binary segmentation zarr.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SKELETON_PREDICTIONS_ZARR,
        help="Path for output skeleton prediction zarr.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Path to model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--roi-size",
        type=int,
        nargs=3,
        default=[96, 96, 96],
        metavar=("Z", "Y", "X"),
        help="Sliding-window tile size. Default: 96 96 96.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Fractional overlap between tiles. Default: 0.5.",
    )
    parser.add_argument(
        "--field-backend",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Backend for normal field computation. Default: gpu.",
    )
    parser.add_argument(
        "--normal-field",
        type=Path,
        default=None,
        help="Path to save/load the precomputed normal field zarr. "
             "Defaults to <output>.parent/normal_field.zarr.",
    )
    parser.add_argument(
        "--chunk-shape",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        metavar=("Z", "Y", "X"),
        help="Chunk size for chunkwise normal field computation. Default: 128 128 128.",
    )
    parser.add_argument(
        "--border-shape",
        type=int,
        nargs=3,
        default=[32, 32, 32],
        metavar=("Z", "Y", "X"),
        help="Border size added around each chunk for EDT context. Default: 32 32 32.",
    )
    args = parser.parse_args()

    main(
        segmentation_zarr=args.segmentation,
        output_zarr=args.output,
        checkpoint_path=args.checkpoint,
        normal_field_zarr=args.normal_field,
        roi_size=tuple(args.roi_size),
        overlap=args.overlap,
        field_backend=args.field_backend,
        chunk_shape=tuple(args.chunk_shape),
        border_shape=tuple(args.border_shape),
    )
