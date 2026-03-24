"""Fusion Part 2.2: Calculate Distance Field on Scaled Images."""

##################################################################################################
#                                           IMPORTS
##################################################################################################
import argparse
import time
from functools import partial

import dask
import dask.array as da
import numpy as np


def local_normalized_distance_gpu_mbf(
    image: np.ndarray,
    max_ball_radius: int = 30,
) -> np.ndarray:
    """
    Compute normalized distance transform for a binary image on GPU using CuPy.

    This algorithm computes the distance transform for each connected component
    of the binary image and normalizes the distances locally using a minimum filter.
    This ensures comparable distance measures across regions of varying sizes.
    It is accelerated on the GPU using CuPy.

    Parameters
    ----------
    image : np.ndarray
        Binary array where non-zero values are interpreted as foreground.
    max_ball_radius : int
        Maximum radius for the structuring element used in the minimum filter.
        Default is 30.

    Returns
    -------
    np.ndarray
        Array of same shape as input image, containing normalized distance values.
    """
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import (
            distance_transform_edt as distance_transform_edt_gpu,
        )
        from cupyx.scipy.ndimage import label
        from cupyx.scipy.ndimage import maximum_filter as maximum_filter_gpu

    except ImportError as err:
        raise ImportError(
            "local_normalized_distance_gpu requires CuPy. "
            "Please install it by following the CuPy "
            "installation instructions for your GPU."
        ) from err

    image = cp.asarray(image)  # move to GPU
    binary = image > 0
    labeled, num_labels = label(binary)
    normalized_distance = cp.zeros_like(image, dtype=cp.float32)

    for i in range(1, num_labels + 1):
        mask = labeled == i

        distance = distance_transform_edt_gpu(mask)
        radius = max_ball_radius

        # apply maximum filter to normalize distances locally
        local_max_distance = maximum_filter_gpu(distance, size=radius * 2 + 1)

        normalized_distance[mask] = distance[mask] / (local_max_distance[mask])

    return cp.asnumpy(normalized_distance)


##################################################################################################
#                                           RUNNING FUSION
##################################################################################################

# Define the image prefix used to name the files
image_prefix = "LADAF-2021-17-left-v7_processed"  # ADAPT HERE

# Example: define scales and their valid ranges
scale_ranges_manual = {
    -1: (1, 10),
    -3: (10, 150),
}  # ADAPT HERE

################ Fusion Part 2.2 ################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--job-index", help="this is the index of the submitted job", type=int
)
parser.add_argument(
    "--job-index-offset",
    help="this number assists in getting the negative and "
    "positive scales required for the fusion algorithm (submit a positive integer)",
    type=int,
)
parser.add_argument("--workers", help="this sets the number of workers", type=int)
args = parser.parse_args()


print("Scale the image here:")
scale_number = args.job_index - args.job_index_offset
print("Job Index: ", args.job_index)
print("Job Index Offset: ", args.job_index_offset)
print("Scale number: ", scale_number)


# Load scaled images form zarr to calculate distcance field on all scaled images
scaled_image = da.from_zarr(
    f"/data/{image_prefix}_image_scaled.zarr/scale{scale_number}"
)
scaled_image = scaled_image.rechunk((192, 192, 192))

print("Scaled image was loaded and rechunked")

# Calculate Distance Field on all scaled images
print(" Calculate distance field next")
start_time = time.time()
with dask.config.set(num_workers=args.workers):
    prefunction_partial = partial(local_normalized_distance_gpu_mbf, max_ball_radius=2)
    dist_image = scaled_image.map_overlap(
        prefunction_partial, depth=10, boundary=0, dtype=np.float32
    )

    # save as zarr
    dist_image = dist_image.rechunk((192, 192, 192))
    dist_image.to_zarr(
        f"/data/{image_prefix}_distance_field_on_scales.zarr/scale{scale_number}_maxball_2",
        overwrite=True,
    )
    """group = zarr.open_group(
        f"/data/{image_prefix}_distance_field_on_scales.zarr", mode="a"
    )
    scale_factor = 2**scale_number
    group[f"scale{scale_number}"].attrs["scale"] = [
        scale_factor,
        scale_factor,
        scale_factor,
    ]"""

# print(f"--- Calculating distance field on scale
#  {scale_number} took {time.time() - start_time} seconds ---")
