"""Parallel worker for Fusion 1.2: Scale Map — one chunk per SLURM task."""

import os
import sys
from functools import partial
from pathlib import Path

import pandas as pd
import zarr

from skeleplex.skeleton.fusion.scale_map import scale_map_generator_gpu

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import RADIUS_MAP_PATH, SCALE_MAP_PATH, SCALE_RANGES_MANUAL
from _parallel_utils import get_slices_for_chunk

csv_path = sys.argv[1]
chunk_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

job_df = pd.read_csv(csv_path)
if chunk_id >= len(job_df):
    print(f"chunk {chunk_id} out of range ({len(job_df)} total), skipping")
    sys.exit(0)

expanded, core_in_result, core_out, _ = get_slices_for_chunk(job_df, chunk_id)

image_chunk = zarr.open(RADIUS_MAP_PATH, mode="r")[expanded]
fn = partial(scale_map_generator_gpu, scale_ranges=SCALE_RANGES_MANUAL)
result = fn(image_chunk)

output = zarr.open(SCALE_MAP_PATH, mode="a")
output[core_out] = result[core_in_result]

print(f"Finished chunk {chunk_id}.")
