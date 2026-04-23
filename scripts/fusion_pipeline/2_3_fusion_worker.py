"""Parallel worker for Fusion 2.3: Skeleton Prediction — one batch of chunks per SLURM task.

The model is loaded once per SLURM task and reused across all chunks in the batch.
Supports both distance_field (1-channel, 3-D) and normal_field (3-channel, 4-D)
inputs, controlled by DISTANCE_FIELD_TYPE in _constants.py.
"""
# isort: skip_file — warnings.catch_warnings block prevents isort-compatible ordering

import os
import sys
from pathlib import Path

import pandas as pd
import zarr

from skeleplex.skeleton._skeletonize import load_normal_field_model, skeletonize
from skeleplex.skeleton._utils import get_skeletonization_model

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import (
    CHECKPOINT_PATH,
    DISTANCE_FIELD_TYPE,
    DISTANCE_FIELD_ZARR,
    SKELETON_PREDICTIONS_ZARR,
)
from _parallel_utils import get_slices_for_chunk, write_batch_marker

csv_path = sys.argv[1]
chunks_per_task = int(sys.argv[2]) if len(sys.argv) > 2 else 1
batch_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

job_df = pd.read_csv(csv_path)
start = batch_id * chunks_per_task
end = min(start + chunks_per_task, len(job_df))

if start >= len(job_df):
    print(f"batch {batch_id} out of range ({len(job_df)} chunks total), skipping")
    sys.exit(0)

# Load model once for the whole batch
if DISTANCE_FIELD_TYPE == "normal_field":
    model = load_normal_field_model(CHECKPOINT_PATH)
else:
    model = get_skeletonization_model()

input_zarrs: dict[int, zarr.Array] = {}
output_zarrs: dict[int, zarr.Array] = {}

for chunk_id in range(start, end):
    expanded, core_in_result, core_out, scale_number = get_slices_for_chunk(job_df, chunk_id)

    if scale_number not in input_zarrs:
        input_zarrs[scale_number] = zarr.open_array(
            f"{DISTANCE_FIELD_ZARR}/scale{scale_number}_{DISTANCE_FIELD_TYPE}", mode="r"
        )
        output_zarrs[scale_number] = zarr.open_array(
            f"{SKELETON_PREDICTIONS_ZARR}/scale{scale_number}", mode="a"
        )

    # For normal_field the zarr is (3, Z, Y, X) — read all channels for the tile
    if DISTANCE_FIELD_TYPE == "normal_field":
        field = input_zarrs[scale_number][(slice(None), *expanded)]
    else:
        field = input_zarrs[scale_number][expanded]

    result = skeletonize(field, model=model, roi_size=(192, 192, 192), progress_bar=False)
    output_zarrs[scale_number][core_out] = result[core_in_result]
    print(f"  chunk {chunk_id} (scale {scale_number}) done ({chunk_id - start + 1}/{end - start})")

write_batch_marker(csv_path, batch_id)
print(f"Batch {batch_id} finished ({end - start} chunks).")
