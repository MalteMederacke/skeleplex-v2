"""Parallel worker for Fusion 2.3: Skeleton Prediction — one batch of chunks per SLURM task.

The model is loaded once per SLURM task and reused across all chunks in the batch.
"""
# isort: skip_file — warnings.catch_warnings block prevents isort-compatible ordering

import copy
import gc
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from morphospaces.networks.skeletonization import SkeletonizationRegressionDynUNet

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from monai.inferers import SlidingWindowInfererAdapt

from skeleplex.skeleton._utils import make_image_5d

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import CHECKPOINT_PATH, DISTANCE_FIELD_ZARR, SKELETON_PREDICTIONS_ZARR
from _parallel_utils import get_slices_for_chunk


def predict_chunk(image: np.ndarray, model, roi_size=(192, 192, 192)) -> np.ndarray:
    expanded = torch.from_numpy(make_image_5d(image))
    model.eval()
    inferer = SlidingWindowInfererAdapt(
        roi_size=roi_size,
        sw_device=torch.device("cuda"),
        sw_batch_size=1,
        overlap=0.5,
        mode="gaussian",
        progress=False,
    )
    with torch.no_grad():
        result = inferer(inputs=expanded, network=model)
    result_cpu = result.cpu()
    del result
    skel_pred = copy.deepcopy(
        torch.squeeze(torch.squeeze(result_cpu, dim=0), dim=0).numpy()
    )
    del expanded, result_cpu
    gc.collect()
    torch.cuda.empty_cache()
    return skel_pred


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
model = SkeletonizationRegressionDynUNet.load_from_checkpoint(CHECKPOINT_PATH)

input_zarrs: dict[int, zarr.Array] = {}
output_zarrs: dict[int, zarr.Array] = {}

for chunk_id in range(start, end):
    expanded, core_in_result, core_out, scale_number = get_slices_for_chunk(job_df, chunk_id)

    if scale_number not in input_zarrs:
        input_zarrs[scale_number] = zarr.open(
            f"{DISTANCE_FIELD_ZARR}/scale{scale_number}_maxball_2", mode="r"
        )
        output_zarrs[scale_number] = zarr.open(
            f"{SKELETON_PREDICTIONS_ZARR}/scale{scale_number}", mode="a"
        )

    result = predict_chunk(input_zarrs[scale_number][expanded], model)
    output_zarrs[scale_number][core_out] = result[core_in_result]
    print(f"  chunk {chunk_id} (scale {scale_number}) done ({chunk_id - start + 1}/{end - start})")

print(f"Batch {batch_id} finished ({end - start} chunks).")
