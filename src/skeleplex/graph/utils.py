import logging  # noqa
import os
import h5py
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def write_slices_to_h5(
    file_path: str,
    file_base: str,
    image_slices: dict,
    segmentation_slices: dict | None = None,
    image_key: str = "image",
    segmentation_key: str = "segmentation",
    sample_grid_spacing_um: float | None = None,
):
    """
    Write image and segmentation slices to an h5 file.

    Parameters
    ----------
    file_path : str
        The path to save to.
    file_base : str
        The base name of the file.
    image_slices : dict
        A dictionary of image slices. Keys are the edge IDs
        of the edge the image slice belongs to.
    segmentation_slices : dict, optional
        A dictionary of segmentation slices. Keys are the edge IDs
        of the edge the segmentation slice belongs to.
    image_key : str
        The key to use for the image slices.
    segmentation_key : str
        The key to use for the segmentation slices.
    sample_grid_spacing_um : float, optional
        The pixel spacing in micrometers. Stored as a file attribute so that
        downstream measurement functions can convert pixel units to µm.
    """
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for edge in image_slices.keys():
        file_name = os.path.join(file_path, f"{file_base}_sn_{edge[0]}_en_{edge[1]}.h5")
        logger.info(f"Writing edge {edge} to {file_name}")

        with h5py.File(file_name, "w") as f:
            f.create_dataset(image_key, data=image_slices[edge])
            if segmentation_slices is not None:
                f.create_dataset(segmentation_key, data=segmentation_slices[edge])
            if sample_grid_spacing_um is not None:
                f.attrs["sample_grid_spacing_um"] = sample_grid_spacing_um


def select_points_in_bounding_box(
    points: np.ndarray,
    lower_left_corner: np.ndarray,
    upper_right_corner: np.ndarray,
) -> np.ndarray:
    """From an array of points, select all points inside a specified bounding box.

    Parameters
    ----------
    points : np.ndarray
        The n x d array containing the n, d-dimensional points to check.
    lower_left_corner : np.ndarray
        The point corresponding to the corner of the bounding box
        with lowest coordinate values.
    upper_right_corner : np.ndarray
        The point corresponding to the corner of the bounding box
        with the highest coordinate values.

    Returns
    -------
    points_in_box : np.ndarray
        The n x d array containing the n points inside of the
        specified bounding box.
    """
    in_box_mask = np.all(
        np.logical_and(lower_left_corner <= points, upper_right_corner >= points),
        axis=1,
    )
    return points[in_box_mask]


def draw_line_segment(
    start_point: np.ndarray,
    end_point: np.ndarray,
    skeleton_image: np.ndarray,
    fill_value: int = 1,
    return_line: bool = False,
):
    """Draw a line segment in-place.

    Note: line will be clipped if it extends beyond the
    bounding box of the skeleton_image.

    Parameters
    ----------
    start_point : np.ndarray
        (d,) array containing the starting point of the line segment.
        Must be an integer index.
    end_point : np.ndarray
        (d,) array containing the end point of the line segment.
        Most be an integer index
    skeleton_image : np.ndarray
        The image in which to draw the line segment.
        Must be the same dimensionality as start_point and end_point.
    fill_value : int
        The value to use for the line segment.
        Default value is 1.
    return_line : bool
        If True, return the coordinates of the line segment. Default is False.
    """
    branch_length = np.linalg.norm(end_point - start_point)
    n_skeleton_points = int(2 * branch_length)
    skeleton_points = np.linspace(start_point, end_point, n_skeleton_points)

    # filter for points within the image
    image_bounds = np.asarray(skeleton_image.shape) - 1
    skeleton_points = select_points_in_bounding_box(
        points=skeleton_points,
        lower_left_corner=np.array([0, 0, 0]),
        upper_right_corner=image_bounds,
    ).astype(int)
    skeleton_image[
        skeleton_points[:, 0], skeleton_points[:, 1], skeleton_points[:, 2]
    ] = fill_value

    if return_line:
        return skeleton_points
