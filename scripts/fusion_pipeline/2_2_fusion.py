"""Fusion Part 2.2: Calculate Distance Field on Scaled Images."""

import argparse
import time
from functools import partial

import dask.array as da
import numpy as np
import zarr

from skeleplex.skeleton.distance_field import local_normalized_distance_gpu
from skeleplex.utils._chunked import iteratively_process_chunks_3d

from ._constants import IMAGE_PREFIX

# Define the image prefix used to name the files
image_prefix = IMAGE_PREFIX

parser = argparse.ArgumentParser()
parser.add_argument("--job-index", type=int)
parser.add_argument("--job-index-offset", type=int)
args = parser.parse_args()

scale_number = args.job_index - args.job_index_offset
print("Job Index: ", args.job_index)
print("Job Index Offset: ", args.job_index_offset)
print("Scale number: ", scale_number)

scaled_image = da.from_zarr(
    f"/data/{image_prefix}_image_scaled.zarr/scale{scale_number}"
)
scaled_image = scaled_image.rechunk((192, 192, 192))
print("Scaled image was loaded and rechunked")

print("Calculate distance field next")
start_time = time.time()
prefunction = partial(local_normalized_distance_gpu, max_ball_radius=2)

save_here = zarr.open(
    f"/data/{image_prefix}_distance_field_on_scales.zarr/scale{scale_number}_maxball_2",
    mode="w",
    shape=scaled_image.shape,
    chunks=(192, 192, 192),
    dtype=np.float32,
)

iteratively_process_chunks_3d(
    input_arrays=scaled_image,
    output_zarr=save_here,
    function_to_apply=prefunction,
    chunk_shape=(192, 192, 192),
    extra_border=(10, 10, 10),
)

print(
    f"--- Calculating distance field on scale {scale_number} "
    f"took {time.time() - start_time} seconds ---"
)
