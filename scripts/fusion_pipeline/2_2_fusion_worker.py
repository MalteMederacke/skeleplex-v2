"""Parallel worker for Fusion 2.2: Distance Field — one chunk per SLURM task."""

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
chunk_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

job_df = pd.read_csv(csv_path)
if chunk_id >= len(job_df):
    print(f"chunk {chunk_id} out of range ({len(job_df)} total), skipping")
    sys.exit(0)

expanded, core_in_result, core_out, scale_number = get_slices_for_chunk(job_df, chunk_id)

input_path = f"{SCALED_IMAGE_ZARR}/scale{scale_number}"
output_path = f"{DISTANCE_FIELD_ZARR}/scale{scale_number}_maxball_2"

image_chunk = zarr.open(input_path, mode="r")[expanded]
fn = partial(local_normalized_distance_gpu, max_ball_radius=2)
result = fn(image_chunk)

output = zarr.open(output_path, mode="a")
output[core_out] = result[core_in_result]

print(f"Finished chunk {chunk_id} (scale {scale_number}).")
