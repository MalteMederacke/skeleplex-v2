"""Example script to threshold and skeletonize a prediction volume."""

import dask.array as da
import numpy as np
from skimage.morphology import skeletonize

path_to_skeleton = "../example_data/skeleton_prediction.zarr"

skeleton_prediction = da.from_zarr(path_to_skeleton)

binary_skeleton = (skeleton_prediction > 0.6).astype(np.uint8)


factor = np.array([4, 4, 4])  # even numbers
desired_chunksize = np.array(binary_skeleton.shape) // factor
skeleton_dask = binary_skeleton.rechunk(desired_chunksize)

depth = 20

skeleton = binary_skeleton.map_overlap(
    skeletonize, depth=depth, boundary="none", dtype=np.uint8
)

skeleton.to_zarr("../example_data/skeleton.zarr", overwrite=True)
