"""Fusion Part 2.4.5: Repair breaks in skeletonized scale images."""

import sys
import time
from pathlib import Path

from skeleplex.skeleton import repair_breaks_lazy

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import (
    SCALE_RANGES_MANUAL,
    SCALED_IMAGE_ZARR,
    SKELETONIZED_ON_SCALES_ZARR,
    SKELETONIZED_REPAIRED_ON_SCALES_ZARR,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-index", type=int, required=True,
        help="SLURM array task ID — indexes into sorted scale list",
    )
    args = parser.parse_args()

    scales = sorted(SCALE_RANGES_MANUAL.keys())
    scale_number = scales[args.job_index]

    skeleton_path = f"{SKELETONIZED_ON_SCALES_ZARR}/scale{scale_number}"
    segmentation_path = f"{SCALED_IMAGE_ZARR}/scale{scale_number}"
    output_path = f"{SKELETONIZED_REPAIRED_ON_SCALES_ZARR}/scale{scale_number}"

    print(f"Job index: {args.job_index}")
    print(f"Scale number: {scale_number}")
    print(f"Skeleton:     {skeleton_path}")
    print(f"Segmentation: {segmentation_path}")
    print(f"Output:       {output_path}")

    start_time = time.time()
    repair_breaks_lazy(
        skeleton_path=skeleton_path,
        segmentation_path=segmentation_path,
        output_path=output_path,
        repair_radius=40,
        chunk_shape=(256, 256, 256),
        backend="cupy",
    )
    print(f"Scale {scale_number}: repair took {time.time() - start_time:.2f}s")
    print("Finished repair breaks step.")
