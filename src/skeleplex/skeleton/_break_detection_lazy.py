"""Functions for lazy chunk-based skeleton break repair."""

from pathlib import Path
from typing import Literal

import numpy as np
import zarr
from tqdm import tqdm

from skeleplex.skeleton._break_detection import (
    BridgeRegistry,
    repair_breaks,
    repair_fusion_breaks,
)
from skeleplex.utils import calculate_expanded_slice


def repair_breaks_chunk(
    skeleton: zarr.Array,
    output_skeleton: zarr.Array,
    segmentation: zarr.Array,
    expanded_slice: tuple[slice, slice, slice],
    actual_border: tuple[int, int, int],
    repair_radius: float,
    label_map_zarr: zarr.Array | None = None,
    n_fit_voxels: int = 10,
    w_distance: float = 1.0,
    w_angle: float = 1.0,
    backend: Literal["cpu", "cupy"] = "cpu",
    bridging: Literal["all", "pair", "tree"] = "all",
    bridge_registry: BridgeRegistry | None = None,
    core_shape: tuple[int, int, int] | None = None,
) -> None:
    """Process a single chunk for skeleton break repair.

    Loads a chunk with boundary region, applies break repair only to
    endpoints in the core region, and writes the full result back to
    output. This allows repairs to extend into the boundary region
    while preventing duplicate processing of endpoints.

    Parameters
    ----------
    skeleton : zarr.Array
        The input zarr array containing the skeleton to repair.
    output_skeleton : zarr.Array
        The output zarr array to write the repaired skeleton to.
    segmentation : zarr.Array
        The zarr array containing the segmentation mask.
    expanded_slice : tuple[slice, slice, slice]
        Slice defining the chunk+boundary region to load from input.
    actual_border : tuple[int, int, int]
        The actual border size used (z, y, x). May be smaller than
        requested at volume edges.
    repair_radius : float
        The maximum Euclidean distance for connecting endpoints.
    label_map_zarr : zarr.Array or None, optional
        Pre-computed global connected component label map stored as a
        zarr array. When provided, the corresponding chunk is loaded
        and forwarded to ``repair_breaks`` to prevent false positives
        at chunk boundaries. Default is None.
    n_fit_voxels : int, optional
        Number of voxels to walk along the skeleton from each
        endpoint when estimating the local tangent direction.
        Default is 10.
    w_distance : float, optional
        Weight for the normalised distance term. Default is 1.0.
    w_angle : float, optional
        Weight for the normalised angle term. Default is 1.0.
    backend : Literal["cpu", "cupy"]
        The computation backend to use.
        Default is "cpu".
    bridging : {"all", "pair", "tree"}
        Which of the proposed bridges are drawn, see ``repair_breaks``.
        Default is "all".
    bridge_registry : BridgeRegistry or None
        Registry shared between chunks, see ``repair_breaks``.
    core_shape : tuple[int, int, int] or None
        Shape of the core region of the chunk (z, y, x). End points are
        searched in the core only. If None, the border after the core is
        assumed to equal ``actual_border``, which is wrong for chunks at
        the edge of the volume.

    Returns
    -------
    None
        Modifies output_zarr in-place.
    """
    # Load chunk+boundary data
    skeleton_chunk = np.array(skeleton[expanded_slice])
    segmentation_chunk = np.array(segmentation[expanded_slice])

    # Load label map chunk if provided
    label_map_chunk = None
    if label_map_zarr is not None:
        label_map_chunk = np.array(label_map_zarr[expanded_slice])

    # Calculate endpoint bounding box within the loaded chunk
    # This restricts endpoint search to the core region
    chunk_shape = skeleton_chunk.shape
    if core_shape is None:
        core_end = tuple(chunk_shape[dim] - actual_border[dim] for dim in range(3))
    else:
        core_end = tuple(actual_border[dim] + core_shape[dim] for dim in range(3))
    endpoint_bbox = (
        (actual_border[0], actual_border[1], actual_border[2]),
        core_end,
    )

    # Apply repair to full chunk but only search for endpoints in core
    repaired_chunk = repair_breaks(
        skeleton_image=skeleton_chunk,
        segmentation=segmentation_chunk,
        repair_radius=repair_radius,
        endpoint_bounding_box=endpoint_bbox,
        label_map=label_map_chunk,
        n_fit_voxels=n_fit_voxels,
        w_distance=w_distance,
        w_angle=w_angle,
        backend=backend,
        bridging=bridging,
        bridge_registry=bridge_registry,
    )

    # Write full result (core + boundary) to output
    # This ensures repairs extending into boundary are captured.
    # Combine with what other chunks have already written: every chunk starts
    # from the unrepaired input, so a plain assignment would erase the repairs
    # that earlier chunks drew in the overlap.
    output_skeleton[expanded_slice] = np.maximum(
        np.array(output_skeleton[expanded_slice]), repaired_chunk
    )


def repair_breaks_lazy(
    skeleton_path: str | Path,
    segmentation_path: str | Path,
    output_path: str | Path,
    repair_radius: float = 10.0,
    chunk_shape: tuple[int, int, int] = (256, 256, 256),
    label_map_path: str | Path | None = None,
    n_fit_voxels: int = 10,
    w_distance: float = 1.0,
    w_angle: float = 1.0,
    backend: Literal["cpu", "cupy"] = "cpu",
    bridging: Literal["all", "pair", "tree"] = "all",
) -> None:
    """Repair breaks in a skeleton using lazy chunk-based processing.

    Processes a skeleton image that is too large to fit in memory by
    dividing it into chunks with overlapping boundaries. Each chunk is
    processed serially to avoid write conflicts. Endpoints are only
    searched in the core region of each chunk, while repairs can extend
    into the boundary regions.

    Parameters
    ----------
    skeleton_path : str or Path
        Path to the input zarr array containing the skeleton.
    segmentation_path : str or Path
        Path to the zarr array containing the segmentation mask.
    output_path : str or Path
        Path where the output zarr array will be created.
    repair_radius : float, optional
        The maximum Euclidean distance for connecting endpoints. Also
        used as the boundary size around each chunk. The boundary is
        set to ``ceil(repair_radius) + 2`` voxels. Default is 10.0.
    chunk_shape : tuple[int, int, int], optional
        The shape of each core chunk to process (z, y, x).
        Independent of the zarr storage chunk size.
        Default is (256, 256, 256).
    label_map_path : str, Path, or None, optional
        Path to a zarr array containing a globally pre-computed
        connected component label map for the skeleton (e.g.
        produced by ``label_chunks_parallel``). When provided,
        chunk-local connected component labelling is skipped,
        preventing false positive repairs at chunk boundaries.
        Must have the same shape as the skeleton. Default is None.
    n_fit_voxels : int, optional
        Number of voxels to walk along the skeleton from each
        endpoint when estimating the local tangent direction.
        Default is 10.
    w_distance : float, optional
        Weight for the normalised distance term in the repair cost
        function. Default is 1.0.
    w_angle : float, optional
        Weight for the normalised angle deviation term in the repair
        cost function. Default is 1.0.
    backend : Literal["cpu", "cupy"], optional
        The backend to use for calculation. Default is "cpu".
    bridging : {"all", "pair", "tree"}, optional
        Which of the proposed bridges are drawn. "all" (default) draws one
        bridge per end point, which can connect the same two fragments
        several times and close loops. "pair" draws at most one bridge
        between any two fragments. "tree" draws a bridge only if the two
        fragments are not yet connected through other bridges. With
        ``label_map_path``, this holds across chunks; without it, only
        within each chunk.

    Raises
    ------
    ValueError
        If the skeleton and segmentation shapes do not match.
    ValueError
        If ``label_map_path`` is provided but points to an array
        whose shape does not match the skeleton.
    """
    # Open zarr arrays
    input_zarr = zarr.open(str(skeleton_path), mode="r")
    segmentation_zarr = zarr.open(str(segmentation_path), mode="r")

    # Validate shapes
    if input_zarr.shape != segmentation_zarr.shape:
        raise ValueError(
            f"Input and segmentation shapes must match. "
            f"Got {input_zarr.shape} and {segmentation_zarr.shape}"
        )

    # Open optional label map
    label_map_zarr = None
    if label_map_path is not None:
        label_map_zarr = zarr.open(str(label_map_path), mode="r")
        if label_map_zarr.shape != input_zarr.shape:
            raise ValueError(
                f"label_map shape {label_map_zarr.shape} does not "
                f"match skeleton shape {input_zarr.shape}"
            )

    # Get metadata
    input_shape = input_zarr.shape
    dtype = input_zarr.dtype
    zarr_chunks = input_zarr.chunks

    # Create output zarr array
    output_zarr = zarr.open(
        str(output_path),
        mode="w",
        shape=input_shape,
        chunks=zarr_chunks,
        dtype=dtype,
    )

    # Calculate chunk grid
    n_chunks = tuple(int(np.ceil(input_shape[i] / chunk_shape[i])) for i in range(3))
    total_chunks = n_chunks[0] * n_chunks[1] * n_chunks[2]

    # Use repair_radius as border size
    border_size = (
        int(np.ceil(repair_radius)) + 2,
        int(np.ceil(repair_radius)) + 2,
        int(np.ceil(repair_radius)) + 2,
    )

    print(
        f"Processing {total_chunks} chunks of size {chunk_shape} "
        f"with border size {border_size}"
    )

    # With a global label map, fragments keep their label in every chunk, so one
    # registry for all chunks prevents bridging the same fragments twice. Without
    # it, labels are chunk-local and each chunk uses its own registry.
    bridge_registry = None
    if bridging != "all" and label_map_zarr is not None:
        bridge_registry = BridgeRegistry(mode=bridging)

    # Process chunks serially
    with tqdm(total=total_chunks, desc="Repairing breaks") as pbar:
        for i in range(n_chunks[0]):
            for j in range(n_chunks[1]):
                for k in range(n_chunks[2]):
                    pbar.update(1)

                    # Calculate core chunk slice
                    core_start = (
                        i * chunk_shape[0],
                        j * chunk_shape[1],
                        k * chunk_shape[2],
                    )
                    core_end = (
                        min(
                            (i + 1) * chunk_shape[0],
                            input_shape[0],
                        ),
                        min(
                            (j + 1) * chunk_shape[1],
                            input_shape[1],
                        ),
                        min(
                            (k + 1) * chunk_shape[2],
                            input_shape[2],
                        ),
                    )
                    core_slice = tuple(
                        slice(core_start[dim], core_end[dim]) for dim in range(3)
                    )

                    # Calculate expanded slice with boundary
                    expanded_slice, actual_border = calculate_expanded_slice(
                        core_slice, border_size, input_shape
                    )

                    # Process this chunk
                    repair_breaks_chunk(
                        skeleton=input_zarr,
                        output_skeleton=output_zarr,
                        segmentation=segmentation_zarr,
                        expanded_slice=expanded_slice,
                        actual_border=actual_border,
                        repair_radius=repair_radius,
                        label_map_zarr=label_map_zarr,
                        n_fit_voxels=n_fit_voxels,
                        w_distance=w_distance,
                        w_angle=w_angle,
                        backend=backend,
                        bridging=bridging,
                        bridge_registry=bridge_registry,
                        core_shape=tuple(
                            core_end[dim] - core_start[dim] for dim in range(3)
                        ),
                    )


def repair_fusion_breaks_chunk(
    skeleton: zarr.Array,
    output_skeleton: zarr.Array,
    segmentation: zarr.Array,
    scale_map: zarr.Array,
    expanded_slice: tuple[slice, slice, slice],
    actual_border: tuple[int, int, int],
    repair_radius: float,
    label_map_zarr: zarr.Array | None = None,
    endpoint_mask_dilation: int = 0,
    backend: Literal["cpu", "cupy"] = "cpu",
    background_value: float = 0,
) -> None:
    """Process a single chunk for fusion boundary skeleton break repair.

    Loads a chunk with boundary region, applies fusion break repair
    only to endpoints in the core region, and writes the full result
    back to output. This allows repairs to extend into the boundary
    region while preventing duplicate processing of endpoints.

    Parameters
    ----------
    skeleton : zarr.Array
        The input zarr array containing the skeleton to repair.
    output_skeleton : zarr.Array
        The output zarr array to write the repaired skeleton to.
    segmentation : zarr.Array
        The zarr array containing the segmentation mask.
    scale_map : zarr.Array
        The zarr array containing the prediction tile ID map. Used to
        identify fusion boundaries between adjacent prediction tiles.
    expanded_slice : tuple[slice, slice, slice]
        Slice defining the chunk+boundary region to load from input.
    actual_border : tuple[int, int, int]
        The actual border size used (z, y, x). May be smaller than
        requested at volume edges.
    repair_radius : float
        The maximum Euclidean distance for connecting endpoints.
    label_map_zarr : zarr.Array or None, optional
        Pre-computed global connected component label map stored as a
        zarr array. When provided, the corresponding chunk is loaded
        and forwarded to ``repair_fusion_breaks`` to prevent false
        positives at chunk boundaries. Default is None.
    endpoint_mask_dilation : int, optional
        Number of binary dilation iterations to apply to the fusion
        boundary mask before filtering endpoints. Default is 0.
    backend : Literal["cpu", "cupy"]
        The computation backend to use. Default is "cpu".
    background_value : float, optional
        The value of background voxels in the scale map. Default is 0.
        Pass ``skeleplex.skeleton.fusion.scale_map.SCALE_MAP_BACKGROUND``
        for scale maps of the fusion pipeline, in which 0 is a scale number.

    Returns
    -------
    None
        Modifies output_skeleton in-place.
    """
    # Load chunk+boundary data
    skeleton_chunk = np.array(skeleton[expanded_slice])
    segmentation_chunk = np.array(segmentation[expanded_slice])
    scale_map_chunk = np.array(scale_map[expanded_slice])

    # Load label map chunk if provided
    label_map_chunk = None
    if label_map_zarr is not None:
        label_map_chunk = np.array(label_map_zarr[expanded_slice])

    # Calculate endpoint bounding box within the loaded chunk
    chunk_shape = skeleton_chunk.shape
    endpoint_bbox = ((0, 0, 0), chunk_shape)

    # Apply repair to full chunk but only search for endpoints in core
    repaired_chunk = repair_fusion_breaks(
        skeleton_image=skeleton_chunk,
        segmentation=segmentation_chunk,
        scale_map_image=scale_map_chunk,
        repair_radius=repair_radius,
        endpoint_bounding_box=endpoint_bbox,
        label_map=label_map_chunk,
        endpoint_mask_dilation=endpoint_mask_dilation,
        backend=backend,
        background_value=background_value,
    )

    # Write full result (core + boundary) to output
    # This ensures repairs extending into boundary are captured.
    # Combine with what other chunks have already written: every chunk starts
    # from the unrepaired input, so a plain assignment would erase the repairs
    # that earlier chunks drew in the overlap.
    output_skeleton[expanded_slice] = np.maximum(
        np.array(output_skeleton[expanded_slice]), repaired_chunk
    )


def repair_fusion_breaks_lazy(
    skeleton_path: str | Path,
    segmentation_path: str | Path,
    scale_map_path: str | Path,
    output_path: str | Path,
    repair_radius: float = 10.0,
    chunk_shape: tuple[int, int, int] = (256, 256, 256),
    label_map_path: str | Path | None = None,
    endpoint_mask_dilation: int = 0,
    backend: Literal["cpu", "cupy"] = "cpu",
    background_value: float = 0,
) -> None:
    """Repair fusion boundary breaks in a skeleton using lazy chunk-based processing.

    Processes a skeleton image that is too large to fit in memory by
    dividing it into chunks with overlapping boundaries. Each chunk is
    processed serially to avoid write conflicts. Only endpoints that
    lie on fusion boundaries (where adjacent voxels have different
    non-zero prediction IDs in the scale map) are considered for
    repair. Endpoints are only searched in the core region of each
    chunk, while repairs can extend into the boundary regions.

    Parameters
    ----------
    skeleton_path : str or Path
        Path to the input zarr array containing the skeleton.
    segmentation_path : str or Path
        Path to the zarr array containing the segmentation mask.
    scale_map_path : str or Path
        Path to the zarr array containing the prediction tile ID map.
        Used to identify fusion boundaries between adjacent prediction
        tiles. Must have the same shape as the skeleton.
    output_path : str or Path
        Path where the output zarr array will be created.
    repair_radius : float, optional
        The maximum Euclidean distance for connecting endpoints. Also
        used as the boundary size around each chunk. The boundary is
        set to ``ceil(repair_radius) + 2`` voxels. Default is 10.0.
    chunk_shape : tuple[int, int, int], optional
        The shape of each core chunk to process (z, y, x).
        Independent of the zarr storage chunk size.
        Default is (256, 256, 256).
    label_map_path : str, Path, or None, optional
        Path to a zarr array containing a globally pre-computed
        connected component label map for the skeleton (e.g.
        produced by ``label_chunks_parallel``). When provided,
        chunk-local connected component labelling is skipped,
        preventing false positive repairs at chunk boundaries.
        Must have the same shape as the skeleton. Default is None.
    endpoint_mask_dilation : int, optional
        Number of binary dilation iterations to apply to the fusion
        boundary mask before filtering endpoints. Uses 26-connectivity
        (3x3x3 structuring element). Useful for capturing endpoints
        that are near but not exactly on a tile boundary.
        Default is 0 (no dilation).
    backend : Literal["cpu", "cupy"], optional
        The backend to use for calculation. Default is "cpu".
    background_value : float, optional
        The value of background voxels in the scale map. Default is 0.
        Pass ``skeleplex.skeleton.fusion.scale_map.SCALE_MAP_BACKGROUND``
        for scale maps of the fusion pipeline, in which 0 is a scale number.

    Raises
    ------
    ValueError
        If the skeleton and segmentation shapes do not match.
    ValueError
        If the skeleton and scale map shapes do not match.
    ValueError
        If ``label_map_path`` is provided but points to an array
        whose shape does not match the skeleton.
    """
    # Open zarr arrays
    input_zarr = zarr.open(str(skeleton_path), mode="r")
    segmentation_zarr = zarr.open(str(segmentation_path), mode="r")
    scale_map_zarr = zarr.open(str(scale_map_path), mode="r")

    # Validate shapes
    if input_zarr.shape != segmentation_zarr.shape:
        raise ValueError(
            f"Skeleton and segmentation shapes must match. "
            f"Got {input_zarr.shape} and {segmentation_zarr.shape}"
        )
    if input_zarr.shape != scale_map_zarr.shape:
        raise ValueError(
            f"Skeleton and scale map shapes must match. "
            f"Got {input_zarr.shape} and {scale_map_zarr.shape}"
        )

    # Open optional label map
    label_map_zarr = None
    if label_map_path is not None:
        label_map_zarr = zarr.open(str(label_map_path), mode="r")
        if label_map_zarr.shape != input_zarr.shape:
            raise ValueError(
                f"label_map shape {label_map_zarr.shape} does not "
                f"match skeleton shape {input_zarr.shape}"
            )

    # Get metadata
    input_shape = input_zarr.shape
    dtype = input_zarr.dtype
    zarr_chunks = input_zarr.chunks

    # Create output zarr array
    output_zarr = zarr.open(
        str(output_path),
        mode="w",
        shape=input_shape,
        chunks=zarr_chunks,
        dtype=dtype,
    )

    # Calculate chunk grid
    n_chunks = tuple(int(np.ceil(input_shape[i] / chunk_shape[i])) for i in range(3))
    total_chunks = n_chunks[0] * n_chunks[1] * n_chunks[2]

    # Use repair_radius as border size
    border_size = (
        int(np.ceil(repair_radius)) + 2,
        int(np.ceil(repair_radius)) + 2,
        int(np.ceil(repair_radius)) + 2,
    )

    print(
        f"Processing {total_chunks} chunks of size {chunk_shape} "
        f"with border size {border_size}"
    )

    # Process chunks serially
    with tqdm(total=total_chunks, desc="Repairing fusion breaks") as pbar:
        for i in range(n_chunks[0]):
            for j in range(n_chunks[1]):
                for k in range(n_chunks[2]):
                    pbar.update(1)

                    # Calculate core chunk slice
                    core_start = (
                        i * chunk_shape[0],
                        j * chunk_shape[1],
                        k * chunk_shape[2],
                    )
                    core_end = (
                        min(
                            (i + 1) * chunk_shape[0],
                            input_shape[0],
                        ),
                        min(
                            (j + 1) * chunk_shape[1],
                            input_shape[1],
                        ),
                        min(
                            (k + 1) * chunk_shape[2],
                            input_shape[2],
                        ),
                    )
                    core_slice = tuple(
                        slice(core_start[dim], core_end[dim]) for dim in range(3)
                    )

                    # Calculate expanded slice with boundary
                    expanded_slice, actual_border = calculate_expanded_slice(
                        core_slice, border_size, input_shape
                    )

                    # Process this chunk
                    repair_fusion_breaks_chunk(
                        skeleton=input_zarr,
                        output_skeleton=output_zarr,
                        segmentation=segmentation_zarr,
                        scale_map=scale_map_zarr,
                        expanded_slice=expanded_slice,
                        actual_border=actual_border,
                        repair_radius=repair_radius,
                        label_map_zarr=label_map_zarr,
                        endpoint_mask_dilation=endpoint_mask_dilation,
                        backend=backend,
                        background_value=background_value,
                    )
