"""Fusion Part 2.5: Upscale skeletonized images back to native resolution."""

import shutil
import sys
import time
from pathlib import Path

from skeleplex.skeleton import upscale_skeleton_parallel

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import (
    SCALE_RANGES_MANUAL,
    SKELETONIZED_REPAIRED_ON_SCALES_ZARR,
    SKELETONIZED_RESCALED_ZARR,
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-index", type=int, required=True,
        help="SLURM array task ID — indexes into sorted scale list",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    scales = sorted(SCALE_RANGES_MANUAL.keys())
    target_scale = max(scales)  # finest scale = native resolution

    scale_number = scales[args.job_index]
    re_scale_factor = 2 ** (target_scale - scale_number)

    print(f"Job index: {args.job_index}")
    print(f"Scale number: {scale_number}  (target: {target_scale})")
    print(f"Re-scale factor: {re_scale_factor}")

    input_path = f"{SKELETONIZED_REPAIRED_ON_SCALES_ZARR}/scale{scale_number}"
    output_path = f"{SKELETONIZED_RESCALED_ZARR}/origin_scale{scale_number}"

    if re_scale_factor == 1:
        print(f"\nScale {scale_number} is already at native resolution, copying directly...")
        if Path(output_path).exists():
            shutil.rmtree(output_path)
        shutil.copytree(input_path, output_path)
        print(f"Copied {input_path} -> {output_path}")
    else:
        skeleton_upscale_factor = (re_scale_factor, re_scale_factor, re_scale_factor)
        n_processing_chunks = (2, 2, 2)
        border_size = (10, 10, 10)
        pool_type = "spawn"

        print("\nRunning upscale_skeleton_parallel...")
        start_time = time.time()

        upscale_skeleton_parallel(
            input_path=input_path,
            output_path=output_path,
            scale_factors=skeleton_upscale_factor,
            n_processing_chunks=n_processing_chunks,
            border_size=border_size,
            n_processes=args.workers,
            pool_type=pool_type,
        )

        run_time = time.time() - start_time
        print(
            f"\nScale {scale_number} -> {target_scale}: "
            f"factor={re_scale_factor}, took {run_time:.2f}s"
        )

    print("Image was re-scaled. End of fusion part 2.")
