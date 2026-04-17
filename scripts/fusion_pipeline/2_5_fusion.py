"""Benchmark script for chunked skeleton upscaling with zarr round-trip."""

import argparse
import time

import dask.array as da

from scripts.fusion_pipeline._constants import IMAGE_PREFIX
from skeleplex.skeleton import upscale_skeleton_parallel

if __name__ == "__main__":
    # Define the image prefix used to name the files
    image_prefix = IMAGE_PREFIX

    # Load the initial image (here: label)
    lung_image = da.from_zarr(f"/data/{image_prefix}.zarr")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-index", help="this is the index of the submitted job", type=int
    )
    parser.add_argument(
        "--job-index-offset",
        help="this number assists in getting the negative and positive"
        "scales required for the fusion algorithm (submit a positive integer)",
        type=int,
    )
    parser.add_argument("--workers", help="this sets the number of workers", type=int)

    args = parser.parse_args()

    print("Scale the image here:")
    scale_number = args.job_index - args.job_index_offset
    print("Job Index: ", args.job_index)
    print("Job Index Offset: ", args.job_index_offset)
    print("Scale number: ", scale_number)

    re_scale_factor = int(1 / (2 ** (scale_number)))

    # Configuration
    skeleton_upscale_factor = (re_scale_factor, re_scale_factor, re_scale_factor)
    n_processing_chunks = (2, 2, 2)
    border_size = (10, 10, 10)
    n_processes = 4
    pool_type = "spawn"

    input_path = f"/data/{image_prefix}_skeletonized_on_scales.zarr/scale{scale_number}"
    output_path = (
        f"/data/{image_prefix}_skeletonized_rescaled.zarr/origin_scale{scale_number}"
    )

    # Benchmark the upscaling function
    print("\nRunning upscale_skeleton_parallel...")
    start_time = time.time()

    upscale_skeleton_parallel(
        input_path=input_path,
        output_path=output_path,
        scale_factors=skeleton_upscale_factor,
        n_processing_chunks=n_processing_chunks,
        border_size=border_size,
        n_processes=n_processes,
        pool_type=pool_type,
    )

    run_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Scale factors: {skeleton_upscale_factor}")
    print(f"Processing chunks: {n_processing_chunks}")
    print(f"Border size: {border_size}")
    print(f"Number of processes: {n_processes}")
    print(f"Pool type: {pool_type}")
    print(f"Scale number: {scale_number}")
    print(f"Re-Scale Factor: {re_scale_factor, re_scale_factor, re_scale_factor}")
    print(f"\nUpscaling took: {run_time:.2f} seconds")
    print("=" * 60)


print("Image was re-scaled. End of fusion part 2.")
