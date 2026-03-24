"""Fusion Part 1: Generate Radius Map and Scale Map."""

##################################################################################################
#                                           IMPORTS
##################################################################################################
import time
from collections.abc import Callable

import dask
import dask.array as da
import numpy as np
import zarr
from skimage.morphology import ball
from tqdm import tqdm


def iteratively_process_chunks_3d_multi(
    input_arrays: tuple[da.Array, da.Array, da.Array],
    output_zarr: zarr.Array,
    function_to_apply: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    chunk_shape: tuple[int, int, int],
    extra_border: tuple[int, int, int],
    *args,
    **kwargs,
):
    """
    Apply a function to each chunk of a Dask array with extra border handling.

    no
    ----------
    input_array : dask.array.Array
        The input Dask array to process. Must be 3D.
    output_zarr : zarr.Array
        The output Zarr array to write results to.
        Must have the same shape as input_array.
    function_to_apply : Callable[[np.ndarray], np.ndarray]
        The function to apply to each chunk.
    chunk_shape : tuple[int, int, int]
        The shape of each chunk to process.
    extra_border : tuple[int, int, int]
        The extra border to include around each chunk.
    *args
        Additional positional arguments to pass to function_to_apply.
    **kwargs
        Additional keyword arguments to pass to function_to_apply.
    """
    if len(input_arrays) != 3:
        raise ValueError(f"Expected 3 input arrays, got {len(input_arrays)}")

    # validate all arrays are 3D and same shape
    shape0 = input_arrays[0].shape
    for arr in input_arrays:
        if arr.ndim != 3:
            raise ValueError(f"All input arrays must be 3D, got {arr.ndim}D")
        if arr.shape != shape0:
            raise ValueError("All input arrays must have the same shape")

    array_shape = shape0
    n_chunks = tuple(int(np.ceil(array_shape[i] / chunk_shape[i])) for i in range(3))
    total_chunks = n_chunks[0] * n_chunks[1] * n_chunks[2]

    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        for i in range(n_chunks[0]):
            for j in range(n_chunks[1]):
                for k in range(n_chunks[2]):
                    pbar.update(1)

                    core_start = (
                        i * chunk_shape[0],
                        j * chunk_shape[1],
                        k * chunk_shape[2],
                    )
                    core_end = tuple(
                        min((idx + 1) * chunk_shape[dim], array_shape[dim])
                        for dim, idx in enumerate((i, j, k))
                    )
                    core_slice = tuple(
                        slice(core_start[dim], core_end[dim]) for dim in range(3)
                    )

                    expanded_start = tuple(
                        max(0, core_start[dim] - extra_border[dim]) for dim in range(3)
                    )
                    expanded_end = tuple(
                        min(array_shape[dim], core_end[dim] + extra_border[dim])
                        for dim in range(3)
                    )
                    expanded_slice = tuple(
                        slice(expanded_start[dim], expanded_end[dim])
                        for dim in range(3)
                    )
                    actual_border_before = tuple(
                        core_start[dim] - expanded_start[dim] for dim in range(3)
                    )

                    # compute chunks for all three arrays
                    chunks_with_border = [
                        arr[expanded_slice].compute() for arr in input_arrays
                    ]
                    print(
                        "Chunks with border shapes:",
                        [c.shape for c in chunks_with_border],
                    )

                    # apply function
                    processed = function_to_apply(*chunks_with_border, *args, **kwargs)
                    print("Output:", processed.shape)

                    core_in_result_slice = [
                        slice(
                            actual_border_before[dim],
                            actual_border_before[dim]
                            + (core_end[dim] - core_start[dim]),
                        )
                        for dim in range(3)
                    ]

                    # if the processed array has extra dims (e.g., channels/features),
                    # extend the slice with full slices for those dimensions
                    n_extra_dims = processed.ndim - 3
                    if n_extra_dims > 0:
                        extra_slices = [
                            slice(0, processed.shape[dim_idx])
                            for dim_idx in range(n_extra_dims)
                        ]
                        # this is used to slice the processed array
                        core_in_result_slice = extra_slices + core_in_result_slice
                        # this is used slice the output array into which we write
                        core_slice_extended = extra_slices + list(core_slice)
                    else:
                        # if no extra dims, just use the 3D slices
                        core_slice_extended = list(core_slice)

                    # convert back to tuple
                    core_in_result_slice = tuple(core_in_result_slice)
                    core_slice_extended = tuple(core_slice_extended)

                    if processed.ndim != len(core_in_result_slice):
                        raise ValueError(
                            "The output of function_to_apply has"
                            " incompatible number of dimensions."
                        )

                    core_result = processed[core_in_result_slice]
                    output_zarr[core_slice_extended] = core_result


def iteratively_process_chunks_3d(
    input_array: da.Array,
    output_zarr: zarr.Array,
    function_to_apply: Callable[[np.ndarray], np.ndarray],
    chunk_shape: tuple[int, int, int],
    extra_border: tuple[int, int, int],
    *args,
    **kwargs,
):
    """Apply a function to each chunk of a Dask array with extra border handling.

    no
    ----------
    input_array : dask.array.Array
        The input Dask array to process. Must be 3D.
    output_zarr : zarr.Array
        The output Zarr array to write results to.
        Must have the same shape as input_array.
    function_to_apply : Callable[[np.ndarray], np.ndarray]
        The function to apply to each chunk.
    chunk_shape : tuple[int, int, int]
        The shape of each chunk to process.
    extra_border : tuple[int, int, int]
        The extra border to include around each chunk.
    *args
        Additional positional arguments to pass to function_to_apply.
    **kwargs
        Additional keyword arguments to pass to function_to_apply.
    """
    # validate inputs before processing
    if input_array.ndim != 3:
        raise ValueError(f"Input array must be 3D, got {input_array.ndim}D")

    if len(chunk_shape) != 3:
        raise ValueError(
            f"chunk_shape must be a 3-tuple, got length {len(chunk_shape)}"
        )

    if len(extra_border) != 3:
        raise ValueError(
            f"extra_border must be a 3-tuple, got length {len(extra_border)}"
        )

    # calculate the chunk grid
    array_shape = input_array.shape
    n_chunks = tuple(int(np.ceil(array_shape[i] / chunk_shape[i])) for i in range(3))

    total_chunks = n_chunks[0] * n_chunks[1] * n_chunks[2]
    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        for i in range(n_chunks[0]):
            for j in range(n_chunks[1]):
                for k in range(n_chunks[2]):
                    pbar.update(1)

                    # calculate core chunk slice
                    core_start = (
                        i * chunk_shape[0],
                        j * chunk_shape[1],
                        k * chunk_shape[2],
                    )
                    core_end = (
                        min((i + 1) * chunk_shape[0], array_shape[0]),
                        min((j + 1) * chunk_shape[1], array_shape[1]),
                        min((k + 1) * chunk_shape[2], array_shape[2]),
                    )
                    core_slice = tuple(
                        slice(core_start[dim], core_end[dim]) for dim in range(3)
                    )

                    # calculate expanded slice (chunk + border)
                    # clipped to array boundaries
                    expanded_start = tuple(
                        max(0, core_start[dim] - extra_border[dim]) for dim in range(3)
                    )
                    expanded_end = tuple(
                        min(array_shape[dim], core_end[dim] + extra_border[dim])
                        for dim in range(3)
                    )
                    expanded_slice = tuple(
                        slice(expanded_start[dim], expanded_end[dim])
                        for dim in range(3)
                    )

                    # calculate actual border used (may be smaller at edges)
                    actual_border_before = tuple(
                        core_start[dim] - expanded_start[dim] for dim in range(3)
                    )

                    # extract chunk + border and compute
                    chunk_with_border = input_array[expanded_slice].compute()

                    # apply function
                    processed = function_to_apply(chunk_with_border, *args, **kwargs)

                    # extend slice to match output_array_shape array dimensions
                    core_in_result_slice = [
                        slice(
                            actual_border_before[dim],
                            actual_border_before[dim]
                            + (core_end[dim] - core_start[dim]),
                        )
                        for dim in range(3)
                    ]

                    # if the processed array has extra dims (e.g., channels/features),
                    # extend the slice with full slices for those dimensions
                    n_extra_dims = processed.ndim - 3
                    # dimensions beyond the first 3
                    if n_extra_dims > 0:
                        extra_slices = [
                            slice(0, processed.shape[dim_idx])
                            for dim_idx in range(n_extra_dims)
                        ]

                        # this is used to slice the processed array
                        core_in_result_slice = extra_slices + core_in_result_slice
                        # this is used slice the output array into which we write
                        core_slice_extended = extra_slices + list(core_slice)
                    else:
                        # if no extra dims, just use the 3D slices
                        core_slice_extended = list(core_slice)

                    # convert back to tuple
                    core_in_result_slice = tuple(core_in_result_slice)
                    core_slice_extended = tuple(core_slice_extended)

                    # check if end dimensions match input
                    if processed.ndim != len(core_in_result_slice):
                        raise ValueError(
                            "The output of function_to_apply has "
                            "incompatible number of dimensions."
                        )

                    core_result = processed[core_in_result_slice]

                    # write to Zarr
                    output_zarr[core_slice_extended] = core_result


##################################################################################################
#                                           FUNCTIONS
##################################################################################################
def radius_map_generator_gpu(
    image: np.ndarray,
    max_ball_radius: int = 60,
) -> np.ndarray:
    """
    Compute max radius map for a segmented image.

    This algorithm computes the maximal radius for each connected component
    of a segmented image in a given radius and labels the image according to
    radius size in voxels.

    This function is accelerated on the GPU using CuPy.

    Parameters
    ----------
    image : np.ndarray
        Binary array where non-zero values are interpreted as foreground.
    max_ball_radius : int
        Maximum radius for the structuring element used in the maximum filter.
        Default is 30.

    Returns
    -------
    np.ndarray
        Array of same shape as input image, containing distance values.
    """
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import (
            distance_transform_edt as distance_transform_edt_gpu,
        )
        from cupyx.scipy.ndimage import label
        from cupyx.scipy.ndimage import maximum_filter as maximum_filter_gpu

    except ImportError as err:
        raise ImportError(
            "local_normalized_distance_gpu requires CuPy. "
            "Please install it by following the CuPy "
            "installation instructions for your GPU."
        ) from err

    image = cp.asarray(image)  # move to GPU
    binary = image > 0
    labeled, num_labels = label(binary)
    local_max_distance = cp.zeros_like(image, dtype=cp.float32)

    for i in range(1, num_labels + 1):
        mask = labeled == i

        distance = distance_transform_edt_gpu(mask)

        local_max = cp.max(distance)
        radius = min(int(local_max / 2), max_ball_radius)

        # apply maximum filter to normalize distances locally
        footprint_ball = ball(radius * 2)
        local_max_distance[mask] = maximum_filter_gpu(
            distance, footprint=footprint_ball
        )[mask]

    del (
        image,
        binary,
        labeled,
        num_labels,
    )  # , distance, local_max, radius, footprint_ball

    return cp.asnumpy(local_max_distance)


def scale_map_generator_gpu(radius_map: np.ndarray, scale_ranges: dict) -> np.ndarray:
    """
    Generate the scales map for the fusion algorithm.

    Each value in the radius map is mapped to a certain scale based on the range
    that it falls into in the provided scale range dictionary.

    It is accelerated on the GPU using CuPy.

    Parameters
    ----------
    radius_map : np.ndarray
        Array with non-zero values documenting the radius of each tube.
    scale_ranges : dict
        This dictionary is used to map the scales to the radii in the radius_map.

    Returns
    -------
    np.ndarray
        Array of same shape as input image, containing scale mapped values.
    """
    try:
        import cupy as cp

    except ImportError as err:
        raise ImportError(
            "local_normalized_distance_gpu requires CuPy. "
            "Please install it by following the CuPy "
            "installation instructions for your GPU."
        ) from err

    radius_map = cp.asarray(radius_map)
    mask = radius_map > 0

    scale_map = cp.zeros_like(radius_map, dtype=np.float32)

    for key, (start, end) in scale_ranges.items():
        mask = (radius_map >= start) & (radius_map < end)
        scale_map[mask] = key

    return cp.asnumpy(scale_map)


def scale_map_processing_gpu(
    image: np.ndarray,
    scale_map: np.ndarray,
    radius_map: np.ndarray,
) -> np.ndarray:
    """
    Compute max radius map for a segmented image.

    This algorithm computes the maximal radius for each connected component
    of a segmented image in a given radius and labels the image according to
    radius size in voxels.

    This function is accelerated on the GPU using CuPy.

    Parameters
    ----------
    image : np.ndarray
        Binary array where non-zero values are interpreted as foreground.
    scale_map : np.ndarray
        Scale map to be processed
    radius_map : np.ndarray
        Radius map to be used for local border size estimation.

    Returns
    -------
    np.ndarray
        Array of same shape as input image, containing distance values.
    """
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import maximum_filter as maximum_filter_gpu

    except ImportError as err:
        raise ImportError(
            "local_normalized_distance_gpu requires CuPy. "
            "Please install it by following the CuPy "
            "installation instructions for your GPU."
        ) from err
    # Move data_chunk to GPU
    image_block = cp.asarray(image)
    scale_map_block = cp.asarray(scale_map)

    # Apply processing
    mask = image_block > 0

    # apply minimum filter to smoothen out scalemap
    footprint_ball = ball(10)

    # Replace background with -inf so it never wins the max filter
    masked_scale = cp.where(mask, scale_map_block, -cp.inf)

    # Apply maximum filter on FULL array (spatially correct)
    filtered = maximum_filter_gpu(masked_scale, footprint=footprint_ball)

    # Write back only foreground
    local_min_scale_block = cp.zeros_like(scale_map_block, dtype=cp.float32)
    local_min_scale_block[mask] = filtered[mask]

    del image_block, scale_map_block, footprint_ball, mask

    return cp.asnumpy(local_min_scale_block)


##################################################################################################
#                                           RUNNING FUSION
##################################################################################################

# Define the image prefix used to name the files
image_prefix = "LADAF-2021-17-left-v7_processed"  # ADAPT HERE

# Example: define scales and their valid ranges
scale_ranges_manual = {
    -1: (1, 10),
    -3: (10, 250),
}  # ADAPT HERE


# Load the initial image (here: label)
lung_image = da.from_zarr(f"/data/{image_prefix}.zarr")  # ADAPT HERE
lung_image = lung_image.rechunk((192, 192, 192))


################ Fusion Part 1 ################

# Load Radius Map
lung_image_radius_map = da.from_zarr(
    f"/data/{image_prefix}_radius_map_new.zarr/scale_original"
)
lung_image_radius_map = lung_image_radius_map.rechunk((192, 192, 192))


# Load Scale Map
lung_image_scale_map = da.from_zarr(
    f"/data/{image_prefix}_image_scale_map.zarr/scale_original"
)
lung_image_scale_map = lung_image_scale_map.rechunk((192, 192, 192))


# Process Scale Map
start_time = time.time()
with dask.config.set(num_workers=1):
    save_here = zarr.open(
        f"/data/{image_prefix}_image_scale_map_processed.zarr/scale_original",
        mode="w",
        shape=lung_image_scale_map.shape,
        chunks=lung_image_scale_map.chunks,
        dtype=lung_image_scale_map.dtype,
    )

    iteratively_process_chunks_3d_multi(
        input_arrays=(lung_image, lung_image_scale_map, lung_image_radius_map),
        output_zarr=save_here,
        function_to_apply=scale_map_processing_gpu,
        chunk_shape=lung_image_scale_map.chunksize,
        extra_border=(20, 20, 20),
        depth=30,
    )

print(f"--- Scale map processing took {time.time() - start_time} seconds ---")

print("End of Fusion part 1.")
