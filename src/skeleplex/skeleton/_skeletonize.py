import gc
import warnings
from pathlib import Path
from typing import Literal

import dask.array as da
import numpy as np
import torch
from tqdm import tqdm

from skeleplex.skeleton._utils import get_skeletonization_model, make_image_5d

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from monai.inferers import SlidingWindowInfererAdapt

_AnyModel = torch.nn.Module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_normal_field_model(checkpoint_path: Path | str) -> _AnyModel:
    """Load a SkeletonizationRegressionDynUNet from a checkpoint.

    Parameters
    ----------
    checkpoint_path : Path or str
        Path to a .ckpt file from a normal-field training run.
    """
    from morphospaces.networks.skeletonization import SkeletonizationRegressionDynUNet

    return SkeletonizationRegressionDynUNet.load_from_checkpoint(
        str(checkpoint_path),
        in_channels=3,
        out_channels=1,
    )


# ---------------------------------------------------------------------------
# Single-volume inference
# ---------------------------------------------------------------------------


def skeletonize(
    image: np.ndarray,
    model: Literal["pretrained"] | _AnyModel = "pretrained",
    roi_size: tuple[int, int, int] = (120, 120, 120),
    overlap: float = 0.5,
    stitching_mode: str = "gaussian",
    progress_bar: bool = True,
    batch_size: int = 1,
) -> np.ndarray:
    """Skeletonize an image using sliding-window inference.

    Accepts either a 3-D normalized distance field ``(Z, Y, X)`` or a 4-D
    normal field ``(C, Z, Y, X)``.  The channel dimension is inferred from the
    array shape and the correct batch tensor is built automatically, so the
    same function works with any model.

    Parameters
    ----------
    image : np.ndarray
        Input array, shape ``(Z, Y, X)`` or ``(C, Z, Y, X)``.
    model : "pretrained" or torch.nn.Module
        Model to use.  ``"pretrained"`` downloads the default weights.
    roi_size : tuple[int, int, int]
        Sliding-window tile size. Default (120, 120, 120).
    overlap : float
        Fractional overlap between tiles. Default 0.5.
    stitching_mode : str
        Tile blending: "gaussian" or "constant". Default "gaussian".
    progress_bar : bool
        Show inference progress. Default True.
    batch_size : int
        Tiles per forward pass. Default 1.

    Returns
    -------
    np.ndarray
        Skeleton prediction, shape (Z, Y, X).
    """
    if image.ndim == 3:
        tensor = torch.from_numpy(make_image_5d(image))   # (1, 1, Z, Y, X)
    elif image.ndim == 4:
        tensor = torch.from_numpy(image[np.newaxis])       # (1, C, Z, Y, X)
    else:
        raise ValueError(f"Expected 3-D or 4-D array, got shape {image.shape}")

    if model == "pretrained":
        model = get_skeletonization_model()

    model.eval()
    inferer = SlidingWindowInfererAdapt(
        roi_size=roi_size,
        sw_device=torch.device("cuda"),
        sw_batch_size=batch_size,
        overlap=overlap,
        mode=stitching_mode,
        progress=progress_bar,
    )

    with torch.no_grad():
        result = inferer(inputs=tensor, network=model)

    skel_pred = torch.squeeze(torch.squeeze(result.cpu(), dim=0), dim=0).numpy()

    del tensor, result
    gc.collect()
    torch.cuda.empty_cache()

    return skel_pred


# ---------------------------------------------------------------------------
# Chunkwise inference
# ---------------------------------------------------------------------------


def skeletonize_chunkwise(
    input_dask_array: da.Array,
    model: str | _AnyModel = "pretrained",
    chunk_size: tuple[int, int, int] = (512, 512, 512),
    roi_size: tuple[int, int, int] = (120, 120, 120),
    padding: tuple[int, int, int] = (60, 60, 60),
    overlap: float = 0.5,
    stitching_mode: str = "gaussian",
    batch_size: int = 1,
) -> da.Array:
    """Skeletonize a large volume chunk by chunk.

    Works with any model — pass the model directly or ``"pretrained"`` to load
    the default weights.  Accepts both 3-D ``(Z, Y, X)`` and 4-D
    ``(C, Z, Y, X)`` dask arrays; spatial chunking always operates on the last
    three dimensions.

    Parameters
    ----------
    input_dask_array : dask.array.Array
        Input array, shape (Z, Y, X) or (C, Z, Y, X).
    model : str or torch.nn.Module
        Model for skeletonization. ``"pretrained"`` loads the default weights.
    chunk_size : tuple
        Spatial size of each core chunk.
    roi_size : tuple
        ROI size for sliding-window inference.
    padding : tuple
        Spatial overlap margin added around each chunk.
    overlap : float
        Sliding-window overlap within each chunk.
    stitching_mode : str
        Stitching mode for overlapping patches.
    batch_size : int
        Sliding-window batch size.

    Returns
    -------
    dask.array.Array
        Skeleton prediction, shape (Z, Y, X).
    """
    if model == "pretrained":
        model = get_skeletonization_model()

    ndim = input_dask_array.ndim
    if ndim == 3:
        spatial_shape = input_dask_array.shape
    elif ndim == 4:
        spatial_shape = input_dask_array.shape[1:]
    else:
        raise ValueError(f"Expected 3-D or 4-D array, got {ndim}-D")

    start_indices = [
        range(0, s, cs) for s, cs in zip(spatial_shape, chunk_size, strict=True)
    ]
    depth_chunks = []

    for z_start in tqdm(start_indices[0], desc="Z"):
        height_chunks = []
        for y_start in start_indices[1]:
            width_chunks = []
            for x_start in start_indices[2]:
                z0 = max(z_start - padding[0], 0)
                y0 = max(y_start - padding[1], 0)
                x0 = max(x_start - padding[2], 0)
                z1 = min(z_start + chunk_size[0] + padding[0], spatial_shape[0])
                y1 = min(y_start + chunk_size[1] + padding[1], spatial_shape[1])
                x1 = min(x_start + chunk_size[2] + padding[2], spatial_shape[2])

                if ndim == 3:
                    padded_chunk = input_dask_array[z0:z1, y0:y1, x0:x1].compute()
                else:
                    padded_chunk = input_dask_array[:, z0:z1, y0:y1, x0:x1].compute()

                predicted = skeletonize(
                    padded_chunk,
                    model=model,
                    roi_size=roi_size,
                    overlap=overlap,
                    stitching_mode=stitching_mode,
                    progress_bar=False,
                    batch_size=batch_size,
                )

                cz0 = z_start - z0
                cy0 = y_start - y0
                cx0 = x_start - x0
                cz1 = cz0 + min(chunk_size[0], spatial_shape[0] - z_start)
                cy1 = cy0 + min(chunk_size[1], spatial_shape[1] - y_start)
                cx1 = cx0 + min(chunk_size[2], spatial_shape[2] - x_start)

                cropped = predicted[cz0:cz1, cy0:cy1, cx0:cx1]
                width_chunks.append(da.from_array(cropped, chunks=cropped.shape))

            height_chunks.append(da.concatenate(width_chunks, axis=2))
        depth_chunks.append(da.concatenate(height_chunks, axis=1))

    return da.concatenate(depth_chunks, axis=0)
