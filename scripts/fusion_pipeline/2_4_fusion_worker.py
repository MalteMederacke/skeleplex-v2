"""Parallel worker for Fusion 2.4: Mask + Threshold + Skeletonize — one batch per SLURM task."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from skimage.morphology import skeletonize as sk_skeletonize

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import (
    SCALED_IMAGE_ZARR,
    SKELETON_PREDICTIONS_ZARR,
    SKELETONIZED_ON_SCALES_ZARR,
    THRESHOLDS,
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

skel_zarrs: dict[int, zarr.Array] = {}
seg_zarrs: dict[int, zarr.Array] = {}
output_zarrs: dict[int, zarr.Array] = {}

for chunk_id in range(start, end):
    expanded, core_in_result, core_out, scale_number = get_slices_for_chunk(job_df, chunk_id)

    if scale_number not in skel_zarrs:
        skel_zarrs[scale_number] = zarr.open(
            f"{SKELETON_PREDICTIONS_ZARR}/scale{scale_number}", mode="r"
        )
        seg_zarrs[scale_number] = zarr.open(
            f"{SCALED_IMAGE_ZARR}/scale{scale_number}", mode="r"
        )
        output_zarrs[scale_number] = zarr.open(
            f"{SKELETONIZED_ON_SCALES_ZARR}/scale{scale_number}", mode="a"
        )

    skel_chunk = skel_zarrs[scale_number][expanded]
    seg_chunk = seg_zarrs[scale_number][expanded]

    masked = np.where(seg_chunk > 0, skel_chunk, 0).astype(skel_chunk.dtype)
    binary = masked > THRESHOLDS[scale_number]
    result = sk_skeletonize(binary)

    output_zarrs[scale_number][core_out] = result[core_in_result]
    print(f"  chunk {chunk_id} (scale {scale_number}) done ({chunk_id - start + 1}/{end - start})")

write_batch_marker(csv_path, batch_id)
print(f"Batch {batch_id} finished ({end - start} chunks).")
