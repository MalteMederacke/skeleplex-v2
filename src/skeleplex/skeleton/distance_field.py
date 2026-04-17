"""Functions for computing normalized distance transform."""

import numpy as np
from scipy.ndimage import distance_transform_edt, label, maximum_filter


def local_normalized_distance(
    image: np.ndarray,
    max_ball_radius: int = 30,
    return_distance: bool = False,
    use_local_max: bool = True,
) -> np.ndarray:
    """
    Compute normalized distance transform for a binary image.

    This algorithm computes the distance transform for each connected component
    of the binary image and normalizes the distances locally using a maximum filter.
    This ensures comparable distance measures across regions of varying sizes.

    Parameters
    ----------
    image : np.ndarray
        Binary array where non-zero values are interpreted as foreground.
    max_ball_radius : int
        Maximum radius of the ball used for maximum filtering.
        Default is 30.
    return_distance : bool
        If True, return both the distance and normalized distance as a stacked array.
        If False, return only the normalized distance. Default is False.
    use_local_max : bool
        If True, use the local maximum distance for each component.
        If False, use the max_ball_radius for all components.
        This can be useful for certain applications. Default is True.

    Returns
    -------
    np.ndarray
        Array of same shape as input image, containing normalized distance values.
    """
    image = np.asarray(image)
    binary = image > 0
    labeled, num_labels = label(binary)
    normalized_distance = np.zeros_like(image, dtype=np.float32)

    for i in range(1, num_labels + 1):
        mask = labeled == i

        distance = distance_transform_edt(mask)

        if use_local_max:
            local_max = np.max(distance)
        else:
            local_max = max_ball_radius
        radius = min(int(local_max / 2), max_ball_radius)
        # apply maximum filter to normalize distances locally
        local_max_distance = maximum_filter(distance, size=radius * 2 + 1)

        normalized_distance[mask] = distance[mask] / (local_max_distance[mask]) 

    if return_distance:
        return np.stack([distance, normalized_distance], axis=0)

    return normalized_distance


def local_normalized_distance_gpu(
    image: np.ndarray,
    max_ball_radius: int = 30,
    use_local_max: bool = True,
) -> np.ndarray:
    """
    Compute normalized distance transform for a binary image on GPU using CuPy.

    This algorithm computes the distance transform for each connected component
    of the binary image and normalizes the distances locally using a maximum filter.
    This ensures comparable distance measures across regions of varying sizes.
    It is accelerated on the GPU using CuPy.

    Parameters
    ----------
    image : np.ndarray
        Binary array where non-zero values are interpreted as foreground.
    max_ball_radius : int
        Maximum radius for the structuring element used in the maximum filter.
        Default is 30.
    use_local_max : bool
        If True, use the local maximum distance for each component.
        If False, use the max_ball_radius for all components.
        This can be useful for certain applications. Default is True.

    Returns
    -------
    np.ndarray
        Array of same shape as input image, containing normalized distance values.
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
    normalized_distance = cp.zeros_like(image, dtype=cp.float32)

    for i in range(1, num_labels + 1):
        mask = labeled == i

        distance = distance_transform_edt_gpu(mask)

        if use_local_max:
            local_max = cp.max(distance)
        else:
            local_max = max_ball_radius
        radius = min(int(local_max / 2), max_ball_radius)

        # apply maximum filter to normalize distances locally
        local_max_distance = maximum_filter_gpu(distance, size=radius * 2 + 1)

        normalized_distance[mask] = distance[mask] / (local_max_distance[mask])

    return cp.asnumpy(normalized_distance)
