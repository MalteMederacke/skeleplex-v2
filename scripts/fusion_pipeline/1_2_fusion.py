"""Fusion Part 1.2: Generate Scale Map."""

import time
from functools import partial

import dask.array as da
import zarr

from skeleplex.skeleton.fusion.scale_map import scale_map_generator_gpu
from skeleplex.utils._chunked import iteratively_process_chunks_3d

from ._constants import IMAGE_PREFIX, SCALE_RANGES_MANUAL

# Define the image prefix used to name the files
image_prefix = IMAGE_PREFIX  # ADAPT HERE

# Example: define scales and their valid ranges
scale_ranges_manual = SCALE_RANGES_MANUAL


lung_image_radius_map = da.from_zarr(
    f"/data/{image_prefix}_radius_map_new.zarr/scale_original"
)
lung_image_radius_map = lung_image_radius_map.rechunk((192, 192, 192))

# Generate Scale Map
start_time = time.time()
prefunction = partial(scale_map_generator_gpu, scale_ranges=scale_ranges_manual)

save_here = zarr.open(
    f"/data/{image_prefix}_image_scale_map.zarr/scale_original",
    mode="w",
    shape=lung_image_radius_map.shape,
    chunks=(192, 192, 192),
    dtype=lung_image_radius_map.dtype,
)

iteratively_process_chunks_3d(
    input_arrays=lung_image_radius_map,
    output_zarr=save_here,
    function_to_apply=prefunction,
    chunk_shape=(192, 192, 192),
    extra_border=(10, 10, 10),
)

print(f"--- Scale map creation took {time.time() - start_time} seconds ---")
print("End of Fusion part 1.2")
