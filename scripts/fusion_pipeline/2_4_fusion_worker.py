"""Parallel worker for Fusion 2.4: Mask + Threshold + Skeletonize — one chunk per SLURM task.

The three sub-steps of the sequential 2_4_fusion.py are collapsed into a single
in-memory pass per chunk, avoiding intermediate zarr round-trips.
"""

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
from _parallel_utils import get_slices_for_chunk

csv_path = sys.argv[1]
chunk_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

job_df = pd.read_csv(csv_path)
if chunk_id >= len(job_df):
    print(f"chunk {chunk_id} out of range ({len(job_df)} total), skipping")
    sys.exit(0)

expanded, core_in_result, core_out, scale_number = get_slices_for_chunk(job_df, chunk_id)

threshold = THRESHOLDS[scale_number]

skel_chunk = zarr.open(f"{SKELETON_PREDICTIONS_ZARR}/scale{scale_number}", mode="r")[expanded]
seg_chunk = zarr.open(f"{SCALED_IMAGE_ZARR}/scale{scale_number}", mode="r")[expanded]

# mask → threshold → skeletonize (all in-memory, no intermediate zarr)
masked = np.where(seg_chunk > 0, skel_chunk, 0).astype(skel_chunk.dtype)
binary = masked > threshold
result = sk_skeletonize(binary)

output = zarr.open(f"{SKELETONIZED_ON_SCALES_ZARR}/scale{scale_number}", mode="a")
output[core_out] = result[core_in_result]

print(f"Finished chunk {chunk_id} (scale {scale_number}).")
