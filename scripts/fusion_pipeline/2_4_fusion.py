"""Fusion Part 2.4: Threshold and skeltonize the images on each scale."""

##################################################################################################
#                                           IMPORTS
##################################################################################################
import argparse
import time

import dask.array as da
from skimage.morphology import skeletonize as sk_skeletonize

from skeleplex.skeleton._skeletonize import threshold_skeleton

##################################################################################################
#                                           RUNNING FUSION
##################################################################################################

# Define the image prefix used to name the files
image_prefix = "LADAF-2021-17-left-v7_processed"  # ADAPT HERE


# Example: define scales and their valid ranges
scale_ranges_manual = {
    -1: (1, 10),
    -3: (10, 250),
}  # ADAPT HERE

# define threshold for the final fusion 3 step
thresholds = {
    -1: 0.55,
    -3: 0.7,
}

# Load the initial image (here: label)
lung_image = da.from_zarr(f"/data/{image_prefix}.zarr")  # ADAPT HERE


################ Fusion Part 2.4 ################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--job-index", help="this is the index of the submitted job", type=int
)
parser.add_argument(
    "--job-index-offset",
    help="this number assists in getting the negative "
    "and positive scales required for the fusion algorithm (submit a positive integer)",
    type=int,
)
parser.add_argument("--workers", help="this sets the number of workers", type=int)

args = parser.parse_args()


scale_number = args.job_index - args.job_index_offset
print("Job Index: ", args.job_index)
print("Job Index Offset: ", args.job_index_offset)
print("Scale number: ", scale_number)
threshold = thresholds[scale_number]
print("Threshold: ", threshold)

re_scale_factor = 1 / (2 ** (scale_number))

# load skeleton prediction image
skel_pred_image = da.from_zarr(
    f"/data/{image_prefix}_skeleton_predictions_on_scales.zarr/scale{scale_number}"
)

# load the scaled image to mask the skeleton prediction image
scaled_image = da.from_zarr(
    f"/data/{image_prefix}_image_scaled.zarr/scale{scale_number}"
)

# mask out the background in the skeleton prediction image
time_start_masking = time.time()
mask = scaled_image > 0
masked_segmentation = da.where(mask, skel_pred_image, 0)

masked_segmentation.to_zarr(
    f"/data/{image_prefix}_skeleton_predictions_on_scales.zarr/scale{scale_number}_masked",
    mode="w",
)
# print(f"--- Masking the skeleton prediction image
# with the scaled segmentation took {time.time() - time_start_masking} seconds ---")

masked_segmentation_reloaded = da.from_zarr(
    f"/data/{image_prefix}_skeleton_predictions_on_scales.zarr/scale{scale_number}_masked"
)


# Threshold Image
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
print(f"--- Thresholding optimal tree took {time.time() - start_time3} seconds ---")


# Perform Thinning / Skeletonizing
start_time4 = time.time()
lung_image_binary_skeleton_reloaded = lung_image_binary_skeleton_reloaded.rechunk(
    (192, 192, 192)
)
lung_image_skeletonized = lung_image_binary_skeleton_reloaded.map_overlap(
    sk_skeletonize,
    depth=10,
    boundary=0,
    dtype=lung_image_binary_skeleton_reloaded.dtype,
)

lung_image_skeletonized = lung_image_skeletonized.rechunk((192, 192, 192))
lung_image_skeletonized.to_zarr(
    f"/data/{image_prefix}_skeletonized_on_scales.zarr/scale{scale_number}",
    mode="w",
)

# print(f"--- Skeletonizing the thresholded
# image took {time.time() - start_time4} seconds ---")
