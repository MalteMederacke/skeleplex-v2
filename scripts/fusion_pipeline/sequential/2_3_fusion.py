"""Fusion Part 2.3: Predict Skeletons on Scaled Distance Field Images."""
# isort: skip_file  — warnings.catch_warnings block prevents isort-compatible ordering

##################################################################################################
#                                           IMPORTS
##################################################################################################
import argparse
import copy
import gc
import sys
import time
import uuid
import warnings
from functools import partial
from pathlib import Path
from typing import Literal

import dask.array as da
import numpy as np
import torch
import zarr
from morphospaces.networks.multiscale_skeletonization import (
    MultiscaleSkeletonizationNet,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from monai.inferers import SlidingWindowInfererAdapt

from morphospaces.networks.skeletonization import SkeletonizationRegressionDynUNet

from skeleplex.skeleton._utils import get_skeletonization_model, make_image_5d
from skeleplex.utils._chunked import iteratively_process_chunks_3d

# isort: split
sys.path.insert(0, str(Path(__file__).parent.parent))
from _constants import (
    CHECKPOINT_PATH,
    DISTANCE_FIELD_ZARR,
    SKELETON_PREDICTIONS_ZARR,
)

##################################################################################################
#                                           FUNCTIONS
##################################################################################################


def skeletonize_for_dask_simplified(
    image: np.ndarray,
    model: Literal["pretrained"] | MultiscaleSkeletonizationNet = "pretrained",
    roi_size: tuple[int, int, int] = (96, 96, 96),
    overlap: float = 0.5,
    stitching_mode: str = "gaussian",
    progress_bar: bool = True,
    batch_size: int = 1,
) -> np.ndarray:
    """Skeletonize a normalized distance field image.

    Parameters
    ----------
    image : np.ndarray
        The input image to skeletonize.
        This should be a normalized distance field image.
    model : Literal["pretrained"] | MultiscaleSkeletonizationNet = "pretrained",
        The model to use for prediction. This can either be an instance of
        MultiscaleSkeletonizationNet or the string "pretrained". If "pretrained",
        a pretrained model will be downloaded from the SkelePlex repository and used.
        Default value is "pretrained".
    roi_size : tuple[int, int, int]
        The size of each tile to predict on.
        The default value is (120, 120, 120).
    overlap : float
        The amount of overlap between tiles.
        Should be between 0 and 1.
        Default value is 0.5.
    stitching_mode : str
        The method to use to stitch overlapping tiles.
        Should be "gaussian" or "constant".
        "gaussian" uses a Gaussian kernel to weight the overlapping regions.
        "constant" uses equal weight across overlapping regions.
        "gaussian" is the default.
    progress_bar : bool
        Displays a progress bar during the prediction when set to True.
        Default is True.
    batch_size : int
        The number of tiles to predict at once.
        Default value is 1.
    """
    """ if "skel_pred" in locals():
        skel_pred = None
        del skel_pred
        gc.collect()
        torch.cuda.empty_cache()"""

    # start_time = time.time()
    worker_id = str(uuid.uuid4())
    # add dim -> NCZYX
    expanded_image = torch.from_numpy(make_image_5d(image))

    # get the skeletonziation model if requested
    if model == "pretrained":
        model = get_skeletonization_model()

    # put the model in eval mode
    model.eval()
    print("Model: ", model)
    # inferer = SimpleInferer()
    inferer = SlidingWindowInfererAdapt(
        roi_size=roi_size,
        sw_device=torch.device("cuda"),
        sw_batch_size=batch_size,
        overlap=overlap,
        mode=stitching_mode,
        progress=progress_bar,
    )

    # make the prediction
    with torch.no_grad():
        result = inferer(inputs=expanded_image, network=model)

    # squeeze dims -> ZYX
    result_cpu = result.cpu()
    del result

    # add the code to squeeze the extra dims and convert to numpy here
    # squeeze dims -> ZYX
    skel_pred_torch = torch.squeeze(torch.squeeze(result_cpu, dim=0), dim=0).numpy()
    skel_pred = copy.deepcopy(skel_pred_torch)
    # print(f"--- Skeleton Prediction for this chunk
    #  took {time.time() - start_time} seconds ---")
    print(
        f"--- Worker: {worker_id}, Allocated Memory: {torch.cuda.memory_allocated(0)}"
    )

    # clear memory
    # start_time = time.time()
    del model, expanded_image, skel_pred_torch
    gc.collect()
    torch.cuda.empty_cache()
    # print(f"-- Deleting result and model,
    # empty cache took {time.time() - start_time} seconds --" )
    print(
        f"--- Worker: {worker_id}, Allocated Memory: {torch.cuda.memory_allocated(0)}"
    )
    return skel_pred


##################################################################################################
#                                           RUNNING FUSION
##################################################################################################

################ Fusion Part 2.3 ################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--job-index", help="this is the index of the submitted job", type=int
)
parser.add_argument(
    "--job-index-offset",
    help="this number assists in getting the negative "
    "and positive scales required for the fusion algorithm (submit a positive integer)",
    type=int,
)
parser.add_argument("--workers", help="this sets the number of workers", type=int)
args = parser.parse_args()


print("Scale the image here:")
scale_number = args.job_index - args.job_index_offset
print("Job Index: ", args.job_index)
print("Job Index Offset: ", args.job_index_offset)
print("Scale number: ", scale_number)


# Load scaled distance images form zarr to Predict Skeleton on all scaled images
dist_image = da.from_zarr(
    f"{DISTANCE_FIELD_ZARR}/scale{scale_number}_maxball_2"
)
dist_image = dist_image.rechunk((192, 192, 192))


# Predict Skeleton on all scaled images
start_time = time.time()
model = SkeletonizationRegressionDynUNet.load_from_checkpoint(CHECKPOINT_PATH)

partial_skeltonize = partial(skeletonize_for_dask_simplified, model=model)


save_here = zarr.open(
    f"{SKELETON_PREDICTIONS_ZARR}/scale{scale_number}",
    mode="w",
    shape=dist_image.shape,
    chunks=dist_image.chunks,
    dtype=dist_image.dtype,
)

iteratively_process_chunks_3d(
    input_arrays=dist_image,
    output_zarr=save_here,
    function_to_apply=partial_skeltonize,
    chunk_shape=(192, 192, 192),
    extra_border=(60, 60, 60),
)


"""group = zarr.open_group(
    f"/data/{image_prefix}_skeleton_predictions_on_scales.zarr", mode="a"
)
scale_factor = 2**scale_number
group[f"scale{scale_number}"].attrs["scale"] = [
    scale_factor,
    scale_factor,
    scale_factor,
]"""

print(f"--- Skeleton Prediction took {time.time() - start_time} seconds ---")
