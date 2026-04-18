"""Fusion Part 1.1: Generate Radius Map."""

import time

import dask.array as da
import numpy as np
import zarr

from skeleplex.skeleton.fusion.scale_map import radius_map_generator_gpu
from skeleplex.utils._chunked import iteratively_process_chunks_3d

from ._constants import INPUT_IMAGE_PATH, RADIUS_MAP_PATH

# Load the initial image (here: label)
lung_image = da.from_zarr(INPUT_IMAGE_PATH)
lung_image = lung_image.rechunk((192, 192, 192))


# Generate Radius Map
start_time = time.time()

save_here = zarr.open(
    RADIUS_MAP_PATH,
    mode="w",
    shape=lung_image.shape,
    chunks=(192, 192, 192),
    dtype=np.float32,
)

iteratively_process_chunks_3d(
    input_arrays=lung_image,
    output_zarr=save_here,
    function_to_apply=radius_map_generator_gpu,
    chunk_shape=(192, 192, 192),
    extra_border=(30, 30, 30),
)

print(f"--- Radius map creation took {time.time() - start_time} seconds ---")
print("End of Fusion part 1.1")
