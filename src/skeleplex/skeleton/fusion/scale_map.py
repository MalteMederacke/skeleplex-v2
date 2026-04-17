import numpy as np
from skimage.morphology import ball



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

    del image, binary, labeled, num_labels

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
        from cupyx.scipy.ndimage import minimum_filter as minimum_filter_gpu

    except ImportError as err:
        raise ImportError(
            "local_normalized_distance_gpu requires CuPy. "
            "Please install it by following the CuPy "
            "installation instructions for your GPU."
        ) from err
    # Move data to GPU
    image_block = cp.asarray(image)
    scale_map_block = cp.asarray(scale_map)
    radius_map_block = cp.asarray(radius_map)

    # Apply processing
    local_min_scale_block = cp.zeros_like(image_block, dtype=cp.float32)
    mask = image_block > 0

    masked_radius_map = radius_map_block[mask]
    local_avg_radius = cp.mean(masked_radius_map)

    if local_avg_radius is None or cp.isnan(local_avg_radius):
        ball_radius = 60
    else:
        ball_radius = min(int(cp.abs(local_avg_radius) * 3), 90)

    # apply minimum filter to smoothen out scalemap
    footprint_ball = ball(ball_radius)
    local_min_scale_block[mask] = minimum_filter_gpu(
        scale_map_block, footprint=footprint_ball
    )[mask]

    del (
        image_block,
        scale_map_block,
        radius_map_block,
        masked_radius_map,
        local_avg_radius,
        footprint_ball,
        ball_radius,
        mask,
    )

    return cp.asnumpy(local_min_scale_block)


