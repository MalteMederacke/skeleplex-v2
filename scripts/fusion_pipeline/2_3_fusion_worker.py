"""Parallel worker for Fusion 2.3: Skeleton Prediction — one chunk per SLURM task."""
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
        progress=True,
    )
    with torch.no_grad():
        result = inferer(inputs=expanded, network=model)
    result_cpu = result.cpu()
    del result
    skel_pred = copy.deepcopy(
        torch.squeeze(torch.squeeze(result_cpu, dim=0), dim=0).numpy()
    )
    del model, expanded, result_cpu
    gc.collect()
    torch.cuda.empty_cache()
    return skel_pred


csv_path = sys.argv[1]
chunk_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

job_df = pd.read_csv(csv_path)
if chunk_id >= len(job_df):
    print(f"chunk {chunk_id} out of range ({len(job_df)} total), skipping")
    sys.exit(0)

expanded, core_in_result, core_out, scale_number = get_slices_for_chunk(job_df, chunk_id)

input_path = f"{DISTANCE_FIELD_ZARR}/scale{scale_number}_maxball_2"
output_path = f"{SKELETON_PREDICTIONS_ZARR}/scale{scale_number}"

image_chunk = zarr.open(input_path, mode="r")[expanded]

model = SkeletonizationRegressionDynUNet.load_from_checkpoint(CHECKPOINT_PATH)
result = predict_chunk(image_chunk, model)

output = zarr.open(output_path, mode="a")
output[core_out] = result[core_in_result]

print(f"Finished chunk {chunk_id} (scale {scale_number}).")
