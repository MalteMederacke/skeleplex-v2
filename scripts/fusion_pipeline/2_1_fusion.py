"""Fusion Part 2.1: Scale the image to the required scales."""

import argparse
import sys
import time
from pathlib import Path

import dask.array as da

from skeleplex.skeleton.fusion.scale_image import scale_image

# isort: split
sys.path.insert(0, str(Path(__file__).parent))
from _constants import INPUT_IMAGE_PATH, SCALE_RANGES_MANUAL, SCALED_IMAGE_ZARR

lung_image = da.from_zarr(INPUT_IMAGE_PATH)
lung_image = lung_image.rechunk((96, 96, 96))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--job-index", type=int, required=True,
    help="SLURM array task ID — indexes into sorted scale list",
)
args = parser.parse_args()

scales = sorted(SCALE_RANGES_MANUAL.keys())
scale_number = scales[args.job_index]
print(f"Job index: {args.job_index}, scale number: {scale_number}")

start_time = time.time()
scale_image(lung_image, zarr_root=SCALED_IMAGE_ZARR, scale_number=scale_number)
print(f"Scaling image to scale {scale_number} took {time.time() - start_time} s")
