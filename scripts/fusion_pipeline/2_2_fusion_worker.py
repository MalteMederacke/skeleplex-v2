"""Parallel worker for Fusion 2.2: Distance Field — one batch of chunks per SLURM task."""

import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from skeleplex.skeleton.distance_field import local_normalized_distance_gpu

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import DISTANCE_FIELD_ZARR, SCALED_IMAGE_ZARR
from _parallel_utils import get_slices_for_chunk

csv_path = sys.argv[1]
chunks_per_task = int(sys.argv[2]) if len(sys.argv) > 2 else 1
batch_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

job_df = pd.read_csv(csv_path)
start = batch_id * chunks_per_task
end = min(start + chunks_per_task, len(job_df))

if start >= len(job_df):
    print(f"batch {batch_id} out of range ({len(job_df)} chunks total), skipping")
    sys.exit(0)

fn = partial(local_normalized_distance_gpu, max_ball_radius=2)

# Cache open zarr handles per scale to avoid repeated opens
input_zarrs: dict[int, zarr.Array] = {}
output_zarrs: dict[int, zarr.Array] = {}

for chunk_id in range(start, end):
    expanded, core_in_result, core_out, scale_number = get_slices_for_chunk(job_df, chunk_id)

    if scale_number not in input_zarrs:
        input_zarrs[scale_number] = zarr.open(
            f"{SCALED_IMAGE_ZARR}/scale{scale_number}", mode="r"
        )
        output_zarrs[scale_number] = zarr.open(
            f"{DISTANCE_FIELD_ZARR}/scale{scale_number}_maxball_2", mode="a"
        )

    result = fn(input_zarrs[scale_number][expanded])
    output_zarrs[scale_number][core_out] = result[core_in_result]
    print(f"  chunk {chunk_id} (scale {scale_number}) done ({chunk_id - start + 1}/{end - start})")

print(f"Batch {batch_id} finished ({end - start} chunks).")
