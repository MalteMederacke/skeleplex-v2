"""Fusion Part 2.4.5: Label connected components then repair breaks in skeletonized scale images."""

import sys
import time
from pathlib import Path

from skeleplex.skeleton import repair_breaks_lazy
from skeleplex.skeleton._chunked_label import label_and_merge

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import (
    SCALE_RANGES_MANUAL,
    SCALED_IMAGE_ZARR,
    SKELETONIZED_LABELS_ON_SCALES_ZARR,
    SKELETONIZED_ON_SCALES_ZARR,
    SKELETONIZED_REPAIRED_ON_SCALES_ZARR,
    TMP_DIR,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-index", type=int, required=True,
        help="SLURM array task ID — indexes into sorted scale list",
    )
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    scales = sorted(SCALE_RANGES_MANUAL.keys())
    scale_number = scales[args.job_index]

    skeleton_path = f"{SKELETONIZED_ON_SCALES_ZARR}/scale{scale_number}"
    segmentation_path = f"{SCALED_IMAGE_ZARR}/scale{scale_number}"
    label_map_path = f"{SKELETONIZED_LABELS_ON_SCALES_ZARR}/scale{scale_number}"
    output_path = f"{SKELETONIZED_REPAIRED_ON_SCALES_ZARR}/scale{scale_number}"
    tmp_dir = f"{TMP_DIR}/label_scale{scale_number}"

    print(f"Job index:    {args.job_index}")
    print(f"Scale number: {scale_number}")
    print(f"Skeleton:     {skeleton_path}")
    print(f"Segmentation: {segmentation_path}")
    print(f"Label map:    {label_map_path}")
    print(f"Output:       {output_path}")

    # Step 1: precompute global connected-component labels
    # prevents false-positive repairs at chunk boundaries
    print("\n--- Step 1: label connected components ---")
    start_time = time.time()
    label_and_merge(
        input_path=skeleton_path,
        output_path=label_map_path,
        tmp_dir=tmp_dir,
        n_label_processes=args.workers,
        n_merge_processes=args.workers,
        chunk_shape=(256, 256, 256),
        backend="cupy",
    )
    print(f"Scale {scale_number}: labelling took {time.time() - start_time:.2f}s")

    # Step 2: repair breaks using the global label map
    print("\n--- Step 2: repair breaks ---")
    start_time = time.time()
    repair_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        output_path=output_path,
        repair_radius=40,
        chunk_shape=(256, 256, 256),
        label_map_path=label_map_path,
        backend="cupy",
    )
    print(f"Scale {scale_number}: repair took {time.time() - start_time:.2f}s")

    print("Finished repair breaks step.")
