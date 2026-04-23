"""Fusion Part 3: Generate Optimal Tree from Scale Mapped Skeleton Predictions."""

import sys
import time
from pathlib import Path

import dask.array as da
import zarr
from skimage.morphology import skeletonize as sk_skeletonize

from skeleplex.skeleton import repair_breaks_lazy
from skeleplex.skeleton.fusion.scale_image import pad_to_match
from skeleplex.skeleton.fusion.tree_fusion import fused_tree_generator
from skeleplex.utils._chunked import iteratively_process_chunks_3d

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import (
    FINAL_SKELETON_PATH,
    FINAL_SKELETON_SKELETONIZED_PATH,
    FUSED_TREE_PATH,
    INPUT_IMAGE_PATH,
    SCALE_MAP_PROCESSED_PATH,
    SCALE_RANGES_MANUAL,
    SKELETONIZED_RESCALED_ZARR,
)

scale_ranges_manual = SCALE_RANGES_MANUAL

lung_image = da.from_zarr(INPUT_IMAGE_PATH)
lung_image = lung_image.rechunk((192, 192, 192))
print("Lung image shape: ", lung_image.shape)

start_time1 = time.time()

lung_image_scale_map = da.from_zarr(SCALE_MAP_PROCESSED_PATH)
lung_image_scale_map = lung_image_scale_map.rechunk((192, 192, 192))
print("Lung image scale map shape: ", lung_image_scale_map.shape)

multiscale_images = {}
for key in scale_ranges_manual.keys():
    scale_number = key
    image = da.from_zarr(f"{SKELETONIZED_RESCALED_ZARR}/origin_scale{scale_number}")
    image = image.rechunk((192, 192, 192))
    name = f"image_skeletonized_rescaled_from_{scale_number}"
    print(name)
    print("Image shape: ", image.shape)
    image = pad_to_match(lung_image, image, value=0)
    print("Padded image shape: ", image.shape)
    multiscale_images[name] = image

print(f"--- Loading all images took {time.time() - start_time1} seconds ---")

# Combine rescaled images via scale map to generate optimal tree
start_time2 = time.time()
fused_tree_generator(
    lung_image_scale_map, lung_image, scale_ranges_manual, multiscale_images,
    output_path=FUSED_TREE_PATH,
)
print(f"--- Generating optimal tree took {time.time() - start_time2} seconds ---")

lung_image_optimum = da.from_zarr(FUSED_TREE_PATH)

# Repair breaks in the final skeleton
start_time5 = time.time()
repair_breaks_lazy(
    skeleton_path=FUSED_TREE_PATH,
    segmentation_path=INPUT_IMAGE_PATH,
    output_path=FINAL_SKELETON_PATH,
    repair_radius=40,
    chunk_shape=(256, 256, 256),
    backend="cupy",
)

# Thinning / Skeletonizing on the repaired skeleton
start_time6 = time.time()
lung_image_repaired = da.from_zarr(FINAL_SKELETON_PATH)
lung_image_repaired = lung_image_repaired.rechunk((192, 192, 192))

save_here = zarr.open(
    FINAL_SKELETON_SKELETONIZED_PATH,
    mode="w",
    shape=lung_image_repaired.shape,
    chunks=(192, 192, 192),
    dtype=lung_image_repaired.dtype,
)

iteratively_process_chunks_3d(
    input_arrays=lung_image_repaired,
    output_zarr=save_here,
    function_to_apply=sk_skeletonize,
    chunk_shape=(192, 192, 192),
    extra_border=(10, 10, 10),
)
print(f"--- Skeletonizing final skeleton took {time.time() - start_time6} seconds ---")

print("Finished Fusion Part 3: Generated Fused Tree and final skeleton.")
