"""Fusion Part 2.1: Scale the image to the required scales."""

import argparse
import sys
import time
from pathlib import Path

import dask.array as da

from skeleplex.skeleton.fusion.scale_image import scale_image

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import IMAGE_PREFIX, INPUT_IMAGE_PATH

image_prefix = IMAGE_PREFIX

lung_image = da.from_zarr(INPUT_IMAGE_PATH)
lung_image = lung_image.rechunk((96, 96, 96))


parser = argparse.ArgumentParser()
parser.add_argument("--job-index", type=int)
parser.add_argument("--job-index-offset", type=int)
args = parser.parse_args()

scale_number = args.job_index - args.job_index_offset
print("Job Index: ", args.job_index)
print("Job Index Offset: ", args.job_index_offset)
print("Scale number: ", scale_number)

start_time = time.time()
scale_image(lung_image, scale_number=scale_number, image_prefix=image_prefix)
print(f"Scaling image to scale {scale_number} took {time.time() - start_time} s")
