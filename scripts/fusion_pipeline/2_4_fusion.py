"""Fusion Part 2.4: Threshold and skeletonize the images on each scale."""

import argparse
import time

import dask.array as da
import numpy as np
import zarr
from skimage.morphology import skeletonize as sk_skeletonize

from skeleplex.skeleton._skeletonize import threshold_skeleton
from skeleplex.utils._chunked import iteratively_process_chunks_3d

from ._constants import IMAGE_PREFIX, THRESHOLDS

image_prefix = IMAGE_PREFIX

# Thresholds per scale for binary skeleton generation
thresholds = THRESHOLDS

parser = argparse.ArgumentParser()
parser.add_argument("--job-index", type=int)
parser.add_argument("--job-index-offset", type=int)
args = parser.parse_args()

scale_number = args.job_index - args.job_index_offset
print("Job Index: ", args.job_index)
print("Job Index Offset: ", args.job_index_offset)
print("Scale number: ", scale_number)
threshold = thresholds[scale_number]
print("Threshold: ", threshold)

skel_pred_image = da.from_zarr(
    f"/data/{image_prefix}_skeleton_predictions_on_scales.zarr/scale{scale_number}"
)
scaled_image = da.from_zarr(
    f"/data/{image_prefix}_image_scaled.zarr/scale{scale_number}"
)


def _mask_fn(skel: np.ndarray, seg: np.ndarray) -> np.ndarray:
    return np.where(seg > 0, skel, 0).astype(skel.dtype)


# Mask out the background in the skeleton prediction image
time_start_masking = time.time()
save_masked = zarr.open(
    f"/data/{image_prefix}_skeleton_predictions_on_scales.zarr/scale{scale_number}_masked",
    mode="w",
    shape=skel_pred_image.shape,
    chunks=(192, 192, 192),
    dtype=skel_pred_image.dtype,
)

iteratively_process_chunks_3d(
    input_arrays=(skel_pred_image, scaled_image),
    output_zarr=save_masked,
    function_to_apply=_mask_fn,
    chunk_shape=(192, 192, 192),
    extra_border=(0, 0, 0),
)
print(f"--- Masking took {time.time() - time_start_masking} seconds ---")

masked_segmentation_reloaded = da.from_zarr(
    f"/data/{image_prefix}_skeleton_predictions_on_scales.zarr/scale{scale_number}_masked"
)

# Threshold image
start_time3 = time.time()
lung_image_binary_skeleton = threshold_skeleton(
    masked_segmentation_reloaded, threshold=threshold
)
lung_image_binary_skeleton.to_zarr(
    f"/data/{image_prefix}_skeleton_predictions_on_scales.zarr/scale{scale_number}_ts",
    mode="w",
)
lung_image_binary_skeleton_reloaded = da.from_zarr(
    f"/data/{image_prefix}_skeleton_predictions_on_scales.zarr/scale{scale_number}_ts"
)
print(f"--- Thresholding took {time.time() - start_time3} seconds ---")

# Thinning / Skeletonizing
start_time4 = time.time()
lung_image_binary_skeleton_reloaded = lung_image_binary_skeleton_reloaded.rechunk(
    (192, 192, 192)
)

save_skel = zarr.open(
    f"/data/{image_prefix}_skeletonized_on_scales.zarr/scale{scale_number}",
    mode="w",
    shape=lung_image_binary_skeleton_reloaded.shape,
    chunks=(192, 192, 192),
    dtype=lung_image_binary_skeleton_reloaded.dtype,
)

iteratively_process_chunks_3d(
    input_arrays=lung_image_binary_skeleton_reloaded,
    output_zarr=save_skel,
    function_to_apply=sk_skeletonize,
    chunk_shape=(192, 192, 192),
    extra_border=(10, 10, 10),
)
print(f"--- Skeletonizing took {time.time() - start_time4} seconds ---")
