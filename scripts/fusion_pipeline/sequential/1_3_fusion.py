"""Fusion Part 1.3: Process Scale Map."""

import sys
import time
from pathlib import Path

import dask.array as da
import zarr

from skeleplex.skeleton.fusion.scale_map import scale_map_processing_gpu
from skeleplex.utils._chunked import iteratively_process_chunks_3d

# isort: split
sys.path.insert(0, str(Path(__file__).parent.parent))
from _constants import (
    INPUT_IMAGE_PATH,
    RADIUS_MAP_PATH,
    SCALE_MAP_PATH,
    SCALE_MAP_PROCESSED_PATH,
)

lung_image = da.from_zarr(INPUT_IMAGE_PATH)
lung_image = lung_image.rechunk((192, 192, 192))

lung_image_radius_map = da.from_zarr(RADIUS_MAP_PATH)
lung_image_radius_map = lung_image_radius_map.rechunk((192, 192, 192))

lung_image_scale_map = da.from_zarr(SCALE_MAP_PATH)
lung_image_scale_map = lung_image_scale_map.rechunk((192, 192, 192))

# Process Scale Map
start_time = time.time()

save_here = zarr.open(
    SCALE_MAP_PROCESSED_PATH,
    mode="w",
    shape=lung_image_scale_map.shape,
    chunks=(192, 192, 192),
    dtype=lung_image_scale_map.dtype,
)

iteratively_process_chunks_3d(
    input_arrays=(lung_image, lung_image_scale_map, lung_image_radius_map),
    output_zarr=save_here,
    function_to_apply=scale_map_processing_gpu,
    chunk_shape=(192, 192, 192),
    extra_border=(20, 20, 20),
)

print(f"--- Scale map processing took {time.time() - start_time} seconds ---")
print("End of Fusion part 1.3")
