"""Parallel worker for Fusion 1.2: Scale Map — one batch of chunks per SLURM task."""

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
chunks_per_task = int(sys.argv[2]) if len(sys.argv) > 2 else 1
batch_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

job_df = pd.read_csv(csv_path)
start = batch_id * chunks_per_task
end = min(start + chunks_per_task, len(job_df))

if start >= len(job_df):
    print(f"batch {batch_id} out of range ({len(job_df)} chunks total), skipping")
    sys.exit(0)

fn = partial(scale_map_generator_gpu, scale_ranges=SCALE_RANGES_MANUAL)
input_zarr = zarr.open(RADIUS_MAP_PATH, mode="r")
output_zarr = zarr.open(SCALE_MAP_PATH, mode="a")

for chunk_id in range(start, end):
    expanded, core_in_result, core_out, _ = get_slices_for_chunk(job_df, chunk_id)
    result = fn(input_zarr[expanded])
    output_zarr[core_out] = result[core_in_result]
    print(f"  chunk {chunk_id} done ({chunk_id - start + 1}/{end - start})")

print(f"Batch {batch_id} finished ({end - start} chunks).")
