"""Fusion Part 1.2: Generate Scale Map."""

import time
from functools import partial

import dask.array as da
import zarr

from skeleplex.skeleton.fusion.scale_map import scale_map_generator_gpu
from skeleplex.utils._chunked import iteratively_process_chunks_3d

from ._constants import RADIUS_MAP_PATH, SCALE_MAP_PATH, SCALE_RANGES_MANUAL

scale_ranges_manual = SCALE_RANGES_MANUAL

lung_image_radius_map = da.from_zarr(RADIUS_MAP_PATH)
lung_image_radius_map = lung_image_radius_map.rechunk((192, 192, 192))

# Generate Scale Map
start_time = time.time()
prefunction = partial(scale_map_generator_gpu, scale_ranges=scale_ranges_manual)

save_here = zarr.open(
    SCALE_MAP_PATH,
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
