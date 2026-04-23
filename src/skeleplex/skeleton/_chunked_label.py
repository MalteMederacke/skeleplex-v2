import os
from functools import partial
from itertools import product
from multiprocessing import Lock, Value, get_context
from multiprocessing.pool import ThreadPool
from typing import Literal

import dask.array as da
import numpy as np
import zarr
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import label as scipy_label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from skeleplex.utils import get_boundary_slices

# Global variables to hold shared state in worker processes
_offset_counter = None
_counter_lock = None


def _init_worker(offset_counter, counter_lock):
    """Initialize worker process with shared state."""
    global _offset_counter, _counter_lock
    _offset_counter = offset_counter
    _counter_lock = counter_lock


def create_chunk_slices(
    array_shape: tuple[int, ...], chunk_shape: tuple[int, ...]
) -> list[tuple[slice, ...]]:
    """
    Create a list of slice tuples for iterating over an array in chunks.

    Parameters
    ----------
    array_shape : tuple of int
        Shape of the array to be chunked (e.g., (1024, 2048, 2048))
    chunk_shape : tuple of int
        Shape of each chunk (e.g., (256, 512, 512))

    Returns
    -------
    list of tuple of slice
        list where each element is a tuple of slices for one chunk.
        The tuple has the same length as array_shape.
    """
    if len(array_shape) != len(chunk_shape):
        raise ValueError("array_shape and chunk_shape must have same length")

    # Calculate number of chunks along each dimension
    n_chunks_per_dim = [
        (size + chunk_size - 1) // chunk_size  # Ceiling division
        for size, chunk_size in zip(array_shape, chunk_shape, strict=False)
    ]

    # Generate all chunk indices
    chunk_slices = []
    for chunk_indices in product(*[range(n) for n in n_chunks_per_dim]):
        slices = tuple(
            slice(idx * chunk_size, min((idx + 1) * chunk_size, array_size))
            for idx, chunk_size, array_size in zip(
                chunk_indices, chunk_shape, array_shape, strict=False
            )
        )
        chunk_slices.append(slices)

    return chunk_slices


def _label_chunk_with_offset(
    chunk_slices: tuple[slice, ...],
    input_path: str,
    output_path: str,
    structure: np.ndarray | None = None,
) -> tuple[tuple[slice, ...], int]:
    """
    Process a single chunk: label connected components and apply offset.

    Uses global variables _offset_counter and _counter_lock set by initializer.
    This should not be used independently; use label_chunks_parallel instead.

    Parameters
    ----------
    chunk_slices : tuple of slice
        Slice tuple defining the chunk location in the array
    input_path : str
        Path to input zarr array
    output_path : str
        Path to output zarr array
    structure : np.ndarray or None, optional
        Structuring element passed to scipy.ndimage.label. None uses full
        (26-connectivity for 3D) connectivity, matching the previous default.

    Returns
    -------
    tuple
        (chunk_slices, max_label) for logging/debugging
    """
    global _offset_counter, _counter_lock

    # Load input chunk
    input_zarr = zarr.open(input_path, mode="r")
    chunk_data = input_zarr[chunk_slices]

    if structure is None:
        structure = generate_binary_structure(chunk_data.ndim, chunk_data.ndim)
    labeled_chunk, max_label = scipy_label(chunk_data, structure=structure)

    # Get offset atomically and update counter
    with _counter_lock:
        my_offset = _offset_counter.value
        _offset_counter.value += max_label

    # Apply offset to non-background pixels
    if max_label > 0:
        mask = labeled_chunk > 0
        labeled_chunk[mask] += my_offset

    # Write to output zarr
    output_zarr = zarr.open(output_path, mode="r+")
    output_zarr[chunk_slices] = labeled_chunk

    return (chunk_slices, max_label)


def _label_chunk_with_offset_gpu(
    chunk_slices: tuple[slice, ...],
    input_path: str,
    output_path: str,
    structure: np.ndarray | None = None,
) -> tuple[tuple[slice, ...], int]:
    """
    Process a single chunk on GPU: label connected components and apply offset.

    Uses global variables _offset_counter and _counter_lock set by initializer.
    This should not be used independently; use label_chunks_parallel instead.

    Parameters
    ----------
    chunk_slices : tuple of slice
        Slice tuple defining the chunk location in the array
    input_path : str
        Path to input zarr array
    output_path : str
        Path to output zarr array
    structure : np.ndarray or None, optional
        Structuring element for connectivity. None uses full (26-connectivity
        for 3D) connectivity, matching the previous default. Numpy arrays are
        transferred to the GPU automatically.

    Returns
    -------
    tuple
        (chunk_slices, max_label) for logging/debugging
    """
    import cupy as cp
    from cupyx.scipy.ndimage import (
        generate_binary_structure as cupy_generate_binary_structure,
    )
    from cupyx.scipy.ndimage import label as cupy_label

    global _offset_counter, _counter_lock

    # Load input chunk
    input_zarr = zarr.open(input_path, mode="r")
    chunk_data = input_zarr[chunk_slices]

    # Transfer to GPU
    chunk_data_gpu = cp.asarray(chunk_data)

    if structure is None:
        structure_gpu = cupy_generate_binary_structure(
            rank=chunk_data.ndim, connectivity=chunk_data.ndim
        )
    else:
        structure_gpu = cp.asarray(structure)

    labeled_chunk_gpu, max_label = cupy_label(chunk_data_gpu, structure=structure_gpu)
    max_label = int(max_label)

    # Get offset atomically and update counter
    with _counter_lock:
        my_offset = _offset_counter.value
        _offset_counter.value += max_label

    # Apply offset to non-background pixels
    if max_label > 0:
        mask = labeled_chunk_gpu > 0
        labeled_chunk_gpu[mask] += my_offset

    # Transfer back to CPU and write to output zarr
    labeled_chunk = cp.asnumpy(labeled_chunk_gpu)
    output_zarr = zarr.open(output_path, mode="r+")
    output_zarr[chunk_slices] = labeled_chunk

    return (chunk_slices, max_label)


def label_chunks_parallel(
    input_path: str,
    output_path: str,
    chunk_shape: tuple[int, ...],
    n_processes: int = 4,
    pool_type: Literal["spawn", "fork", "forkserver", "thread"] = "fork",
    backend: Literal["cpu", "cupy"] = "cpu",
    structure: np.ndarray | None = None,
) -> int:
    """
    Label connected components in a large zarr image using parallel processing.

    Parameters
    ----------
    input_path : str
        Path to input zarr array
    output_path : str
        Path to output zarr array (will be created if doesn't exist)
    chunk_shape : tuple of int
        Shape of chunks to process in parallel. This will be the chunk shape
        of the output array.
    n_processes : int, default=4
        Number of parallel processes/threads
    pool_type : {'spawn', 'fork', 'forkserver', 'thread'}, default='spawn'
        Type of multiprocessing context to use.
        - 'spawn': Start fresh Python process (safest, works on all platforms)
        - 'fork': Copy parent process (faster but can have issues with threads)
        - 'forkserver': Hybrid approach (Unix only)
        - 'thread': Use threading instead of multiprocessing (good for I/O bound)
    backend : {'cpu', 'cupy'}, default='cpu'
        Backend to use for labeling. 'cpu' uses CPU-based labeling,
        'cupy' uses GPU-based labeling with CuPy. Default is 'cpu'.
    structure : np.ndarray or None, optional
        Structuring element passed to the labeling function. None uses full
        connectivity (26-connectivity for 3D). Pass a center-only array to
        treat each voxel as its own component (for skeleton unique-ID assignment).

    Returns
    -------
    int
        Total number of unique labels assigned

    Notes
    -----
    - Input zarr must already exist
    - Output zarr will be created with same shape/dtype as input if it doesn't exist
    - Components spanning chunk boundaries will receive different labels
    """
    # Open input zarr to get metadata
    input_zarr = zarr.open(input_path, mode="r")
    array_shape = input_zarr.shape

    # Create the output zarr
    _ = zarr.create_array(
        output_path, shape=array_shape, chunks=chunk_shape, dtype=np.uint64
    )

    # Create list of chunk slices
    chunk_slices_list = create_chunk_slices(array_shape, chunk_shape)

    print(
        f"Processing {len(chunk_slices_list)} chunks using {n_processes} "
        f"{pool_type} workers"
    )

    # Process chunks in parallel
    if pool_type == "thread":
        from multiprocessing.pool import ThreadPool

        offset_counter = Value("i", 0)
        counter_lock = Lock()
        pool = ThreadPool(
            n_processes,
            initializer=_init_worker,
            initargs=(offset_counter, counter_lock),
        )
    else:
        ctx = get_context(pool_type)
        offset_counter = ctx.Value("i", 0)
        counter_lock = ctx.Lock()
        pool = ctx.Pool(
            n_processes,
            initializer=_init_worker,
            initargs=(offset_counter, counter_lock),
        )

    # Create the processing function
    if backend == "cpu":
        process_func = partial(
            _label_chunk_with_offset,
            input_path=input_path,
            output_path=output_path,
            structure=structure,
        )
    elif backend == "cupy":
        process_func = partial(
            _label_chunk_with_offset_gpu,
            input_path=input_path,
            output_path=output_path,
            structure=structure,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    try:
        _ = pool.map(process_func, chunk_slices_list)
    finally:
        pool.close()
        pool.join()

    total_labels = offset_counter.value

    return total_labels


def _find_touching_labels(
    region_slice: tuple[slice, slice, slice], label_image_path: str
) -> np.ndarray:
    """
    Find pairs of labels that are touching within a region using 26-connectivity.

    Parameters
    ----------
    region_slice: tuple[slice, slice, slice]
        slice objects defining the region to check
    label_image_path: str
        path to zarr array containing labeled image

    Returns
    -------
    np.ndarray
        (n_pairs, 2) array where each row is [label_A, label_B]
        with label_A < label_B. Returns empty (0, 2) array if no touching pairs.
    """
    # Load the region from zarr
    label_image = zarr.open(str(label_image_path), mode="r")
    region = label_image[region_slice]

    # Get unique labels in the region (exclude 0 which is background)
    unique_labels = np.unique(region)
    unique_labels = unique_labels[unique_labels != 0]

    # Create 26-connectivity structure (3D, connectivity=3 includes all diagonals)
    connectivity_structure = generate_binary_structure(3, 3)

    # Set to store unique touching pairs
    touching_pairs = set()

    # For each label, find what it touches
    for label_A in unique_labels:
        # Create binary mask for this label
        binary_mask = region == label_A

        # Dilate by 1 voxel in all 26 directions
        dilated_mask = binary_dilation(binary_mask, structure=connectivity_structure)

        # Find labels in the dilated region
        touching_region = region[dilated_mask]
        touching_labels = np.unique(touching_region)

        # Exclude the label itself and background (0)
        touching_labels = touching_labels[
            (touching_labels != label_A) & (touching_labels != 0)
        ]

        # Add pairs (always store as min, max to avoid duplicates)
        for label_B in touching_labels:
            pair = (min(label_A, label_B), max(label_A, label_B))
            touching_pairs.add(pair)

    # Convert to numpy array
    if len(touching_pairs) == 0:
        return np.empty((0, 2), dtype=region.dtype)

    result = np.array(list(touching_pairs), dtype=region.dtype)
    return result


def _find_touching_labels_gpu(
    region_slice: tuple[slice, slice, slice], label_image_path: str
) -> np.ndarray:
    """Find pairs of labels that are touching within a region using 26-connectivity.

    This is a GPU-accelerated version using CuPy.

    Parameters
    ----------
    region_slice: tuple[slice, slice, slice]
        slice objects defining the region to check
    label_image_path: str
        path to zarr array containing labeled image

    Returns
    -------
    np.ndarray
        (n_pairs, 2) array where each row is [label_A, label_B]
        with label_A < label_B. Returns empty (0, 2) array if no touching pairs.
    """
    import cupy as cp
    from cupyx.scipy.ndimage import binary_dilation, generate_binary_structure

    # Load the region from zarr
    label_image = zarr.open(str(label_image_path), mode="r")
    region = label_image[region_slice]

    # Transfer to GPU
    region_gpu = cp.asarray(region)

    # Get unique labels in the region (exclude 0 which is background)
    unique_labels = cp.unique(region_gpu)
    unique_labels = unique_labels[unique_labels != 0]

    # Create 26-connectivity structure (3D, connectivity=3 includes all diagonals)
    connectivity_structure = generate_binary_structure(3, 3)

    # Set to store unique touching pairs
    touching_pairs = set()

    # For each label, find what it touches
    for label_A in unique_labels:
        # Create binary mask for this label
        binary_mask = region_gpu == label_A

        # Dilate by 1 voxel in all 26 directions
        dilated_mask = binary_dilation(binary_mask, structure=connectivity_structure)

        # Find labels in the dilated region
        touching_region = region_gpu[dilated_mask]
        touching_labels = cp.unique(touching_region)

        # Exclude the label itself and background (0)
        touching_labels = touching_labels[
            (touching_labels != label_A) & (touching_labels != 0)
        ]

        # Transfer touching_labels back to CPU for set operations
        touching_labels_cpu = cp.asnumpy(touching_labels)
        label_A_cpu = int(cp.asnumpy(label_A))

        # Add pairs (always store as min, max to avoid duplicates)
        for label_B in touching_labels_cpu:
            pair = (min(label_A_cpu, label_B), max(label_A_cpu, label_B))
            touching_pairs.add(pair)

    # Convert to numpy array
    if len(touching_pairs) == 0:
        return np.empty((0, 2), dtype=region.dtype)

    result = np.array(list(touching_pairs), dtype=region.dtype)
    return result


def _make_label_mapping(
    touching_pairs: np.ndarray, max_label_value: int
) -> dict[int, int]:
    """
    Create a label mapping based on connected components of touching labels.

    Labels in the same connected component are mapped to the maximum label
    value within that component.

    Parameters
    ----------
    touching_pairs : np.ndarray
        Array of shape (n_pairs, 2) with pairs of touching labels.
    max_label_value : int
        Maximum label value in the entire image.

    Returns
    -------
    dict[int, int]
        Dictionary mapping original labels to new labels. Only includes labels
        that need to change (excludes identity mappings like {5: 5}).
    """
    # Handle empty touching_pairs
    if len(touching_pairs) == 0:
        return {}

    # Build adjacency matrix
    # Matrix size is (max_label_value + 1) to accommodate labels
    # from 0 to max_label_value
    n_labels = max_label_value + 1

    # Extract row and column indices from touching_pairs
    rows = touching_pairs[:, 0]
    cols = touching_pairs[:, 1]

    # Create data for both directions (undirected graph)
    # Add both [a, b] and [b, a] edges
    row_indices = np.concatenate([rows, cols])
    col_indices = np.concatenate([cols, rows])
    data = np.ones(len(row_indices), dtype=np.uint8)

    # Build sparse adjacency matrix
    adjacency_matrix = csr_matrix(
        (data, (row_indices, col_indices)), shape=(n_labels, n_labels)
    )

    # Find connected components
    n_components, component_labels = connected_components(
        adjacency_matrix, directed=False, return_labels=True
    )

    # Get unique labels that appear in touching_pairs
    unique_labels = np.unique(touching_pairs)

    # Find max label in each component
    # component_max[component_id] = max_label in that component
    component_max = {}

    for label_value in unique_labels:
        component_id = component_labels[label_value]

        if component_id not in component_max:
            component_max[component_id] = label_value
        else:
            component_max[component_id] = max(component_max[component_id], label_value)

    # Create mapping: only include labels that change
    label_mapping = {}

    for label_value in unique_labels:
        component_id = component_labels[label_value]
        max_label_in_component = component_max[component_id]

        # Only add to mapping if label changes
        if label_value != max_label_in_component:
            label_mapping[label_value] = max_label_in_component

    return label_mapping


def relabel_parallel(
    label_image_path: str,
    output_array_path: str,
    chunk_shape: tuple[int, int, int],
    label_mapping: dict[int, int],
    n_processes: int,
    pool_type: Literal["spawn", "fork", "forkserver", "thread"],
) -> None:
    """
    Relabel a zarr array in parallel by applying a label mapping to chunks.

    Parameters
    ----------
    label_image_path : str
        Path to input zarr array.
    output_array_path : str
        Path to output zarr array (will be created if doesn't exist).
    chunk_shape : tuple[int, int, int]
        Shape of chunks to process in parallel.
    label_mapping : dict[int, int]
        Mapping from original labels to new labels.
    n_processes : int
        Number of parallel processes/threads.
    pool_type : {'spawn', 'fork', 'forkserver', 'thread'}
        Type of multiprocessing context to use.
    """
    # Open input zarr to get metadata
    input_zarr = zarr.open(label_image_path, mode="r")
    array_shape = input_zarr.shape
    dtype = input_zarr.dtype

    # Create the output zarr
    _ = zarr.create_array(
        output_array_path, shape=array_shape, chunks=chunk_shape, dtype=dtype
    )

    # Create list of chunk slices
    chunk_slices_list = create_chunk_slices(array_shape, chunk_shape)

    print(
        f"Processing {len(chunk_slices_list)} chunks using {n_processes} "
        f"{pool_type} workers"
    )

    # Process chunks in parallel
    if pool_type == "thread":
        pool = ThreadPool(n_processes)
    else:
        ctx = get_context(pool_type)
        pool = ctx.Pool(n_processes)

    # Create the processing function
    process_func = partial(
        _relabel_chunk,
        input_path=label_image_path,
        output_path=output_array_path,
        label_mapping=label_mapping,
    )

    try:
        pool.map(process_func, chunk_slices_list)
    finally:
        pool.close()
        pool.join()


def _relabel_chunk(
    chunk_slice: tuple[slice, ...],
    input_path: str,
    output_path: str,
    label_mapping: dict[int, int],
) -> None:
    """
    Relabel a single chunk by applying the label mapping.

    Parameters
    ----------
    chunk_slice : tuple of slice
        Slice objects defining the chunk region.
    input_path : str
        Path to input zarr array.
    output_path : str
        Path to output zarr array.
    label_mapping : dict[int, int]
        Mapping from original labels to new labels.
    """
    # Open zarr arrays
    input_zarr = zarr.open(input_path, mode="r")
    output_zarr = zarr.open(output_path, mode="r+")

    # Read chunk data
    chunk_data = input_zarr[chunk_slice]

    # Apply mapping
    output_data = _apply_label_mapping(chunk_data, label_mapping)

    # Write to output
    output_zarr[chunk_slice] = output_data


def _apply_label_mapping(
    chunk_data: np.ndarray, label_mapping: dict[int, int]
) -> np.ndarray:
    """
    Apply label mapping using lookup array (fancy indexing).

    Parameters
    ----------
    chunk_data : np.ndarray
        Array of label values.
    label_mapping : dict[int, int]
        Mapping from old labels to new labels.

    Returns
    -------
    np.ndarray
        Relabeled array.
    """
    if len(label_mapping) == 0:
        return chunk_data.copy()

    # Find which labels actually exist in this chunk
    labels_in_chunk = np.unique(chunk_data)

    # Filter mapping to only relevant labels
    relevant_mapping = {k: v for k, v in label_mapping.items() if k in labels_in_chunk}

    if len(relevant_mapping) == 0:
        return chunk_data.copy()

    # Create lookup table
    max_label = chunk_data.max()
    lookup = np.arange(max_label + 1, dtype=chunk_data.dtype)

    # Update lookup for labels that need remapping
    for old_label, new_label in relevant_mapping.items():
        lookup[old_label] = new_label

    # Apply mapping using fancy indexing
    output_data = lookup[chunk_data]

    return output_data


def merge_touching_labels(
    label_image_path: str,
    output_image_path: str,
    chunk_shape: tuple[int, int, int],
    max_label_value: int,
    n_processes: int,
    pool_type: Literal["spawn", "fork", "forkserver", "thread"],
    backend: Literal["cpu", "cupy"] = "cpu",
) -> None:
    """
    Merge touching labels across chunk boundaries.

    This function finds labels that touch at chunk boundaries, computes connected
    components, and relabels all labels in each connected component to the maximum
    label value in that component.

    Parameters
    ----------
    label_image_path : str
        Path to input zarr array.
    output_image_path : str
        Path to output zarr array (will be created if doesn't exist).
    chunk_shape : tuple[int, int, int]
        Shape of chunks to process in parallel.
    max_label_value : int
        Maximum label value in the entire image.
    n_processes : int
        Number of parallel processes/threads.
    pool_type : {'spawn', 'fork', 'forkserver', 'thread'}
        Type of multiprocessing context to use.
    backend : {'cpu', 'cupy'}, default='cpu'
        Backend to use for finding touching labels. 'cpu' uses CPU-based.
        'cupy' uses GPU-based with CuPy. Default is 'cpu'.
    """
    # Open input zarr to get array shape
    input_zarr = zarr.open(label_image_path, mode="r")
    array_shape = input_zarr.shape

    print(f"Merging touching labels in array with shape {array_shape}")

    # Get boundary slices
    boundaries = get_boundary_slices(array_shape, chunk_shape)

    print(f"Found {len(boundaries)} chunk boundaries")

    # Handle case with no boundaries (single chunk or small array)
    if len(boundaries) == 0:
        print("No chunk boundaries found. Copying input to output.")
        _copy_zarr_array(
            label_image_path, output_image_path, chunk_shape, n_processes, pool_type
        )
        return

    # Find touching labels at all boundaries in parallel
    print(f"Finding touching labels using {n_processes} {pool_type} workers")

    if pool_type == "thread":
        pool = ThreadPool(n_processes)
    else:
        ctx = get_context(pool_type)
        pool = ctx.Pool(n_processes)

    # Create processing function for finding touching labels
    if backend == "cpu":
        process_func = partial(_find_touching_labels, label_image_path=label_image_path)
    elif backend == "cupy":
        process_func = partial(
            _find_touching_labels_gpu, label_image_path=label_image_path
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    try:
        results = pool.map(process_func, boundaries)
    finally:
        pool.close()
        pool.join()

    # Combine all touching pairs
    all_touching_pairs = [arr for arr in results if len(arr) > 0]

    if len(all_touching_pairs) == 0:
        print("No touching labels found. Copying input to output.")
        _copy_zarr_array(
            label_image_path, output_image_path, chunk_shape, n_processes, pool_type
        )
        return

    # Stack and remove duplicates
    touching_pairs = np.vstack(all_touching_pairs)
    touching_pairs = np.unique(touching_pairs, axis=0)

    print(f"Found {len(touching_pairs)} unique touching label pairs")

    # Create label mapping based on connected components
    label_mapping = _make_label_mapping(touching_pairs, max_label_value)

    print(f"Created mapping for {len(label_mapping)} labels")

    if len(label_mapping) == 0:
        print("No labels need remapping. Copying input to output.")
        _copy_zarr_array(
            label_image_path, output_image_path, chunk_shape, n_processes, pool_type
        )
        return

    # Apply relabeling in parallel
    print("Applying relabeling to all chunks")
    relabel_parallel(
        label_image_path=label_image_path,
        output_array_path=output_image_path,
        chunk_shape=chunk_shape,
        label_mapping=label_mapping,
        n_processes=n_processes,
        pool_type=pool_type,
    )

    print("Merging complete")


def _copy_zarr_array(
    input_path: str,
    output_path: str,
    chunk_shape: tuple[int, int, int],
    n_processes: int,
    pool_type: Literal["spawn", "fork", "forkserver", "thread"],
) -> None:
    """
    Copy a zarr array from input to output chunk-by-chunk.

    Parameters
    ----------
    input_path : str
        Path to input zarr array.
    output_path : str
        Path to output zarr array.
    chunk_shape : tuple[int, int, int]
        Chunk shape for output array.
    n_processes : int
        Number of parallel processes/threads.
    pool_type : {'spawn', 'fork', 'forkserver', 'thread'}
        Type of multiprocessing context to use.
    """
    # Open input to get metadata
    input_zarr = zarr.open(input_path, mode="r")
    array_shape = input_zarr.shape
    dtype = input_zarr.dtype

    # Create output array
    _ = zarr.open(
        output_path,
        mode="w",
        shape=array_shape,
        chunks=chunk_shape,
        dtype=dtype,
    )

    # Get chunk slices
    chunk_slices_list = create_chunk_slices(array_shape, chunk_shape)

    print(
        f"Copying {len(chunk_slices_list)} chunks using"
        f"{n_processes} {pool_type} workers"
    )

    # Create pool
    if pool_type == "thread":
        pool = ThreadPool(n_processes)
    else:
        ctx = get_context(pool_type)
        pool = ctx.Pool(n_processes)

    # Create processing function
    process_func = partial(
        _copy_chunk,
        input_path=input_path,
        output_path=output_path,
    )

    try:
        pool.map(process_func, chunk_slices_list)
    finally:
        pool.close()
        pool.join()


def _copy_chunk(
    chunk_slice: tuple[slice, ...],
    input_path: str,
    output_path: str,
) -> None:
    """
    Copy a single chunk from input to output.

    Parameters
    ----------
    chunk_slice : tuple of slice
        Slice objects defining the chunk region.
    input_path : str
        Path to input zarr array.
    output_path : str
        Path to output zarr array.
    """
    # Open zarr arrays
    input_zarr = zarr.open(input_path, mode="r")
    output_zarr = zarr.open(output_path, mode="r+")

    # Copy chunk
    output_zarr[chunk_slice] = input_zarr[chunk_slice]


def label_and_merge(
    input_path: str,
    output_path: str,
    chunk_shape: tuple[int, ...],
    tmp_dir: str,
    n_label_processes: int = 4,
    n_merge_processes: int = 4,
    pool_type: Literal["spawn", "fork", "forkserver", "thread"] = "fork",
    backend: Literal["cpu", "cupy"] = "cpu",
    structure: np.ndarray | None = None,
) -> None:
    """
    Label connected components chunk-by-chunk, then merge labels across boundaries.

    Runs :func:`label_chunks_parallel` into a temporary intermediate zarr stored
    in *tmp_dir*, then calls :func:`merge_touching_labels` to stitch labels that
    span chunk boundaries into the final output.

    Parameters
    ----------
    input_path : str
        Path to the input binary segmentation zarr array.
    output_path : str
        Path where the merged labeled output zarr array will be written.
    chunk_shape : tuple of int
        Shape of the chunks used for both labeling and merging.
    tmp_dir : str
        Directory for the intermediate (unmerged) labeled zarr.
        Created automatically if it does not exist.
    n_label_processes : int, default=4
        Number of worker processes for the labeling step.
    n_merge_processes : int, default=4
        Number of worker processes for the merging step.
    pool_type : {'spawn', 'fork', 'forkserver', 'thread'}, default='fork'
        Multiprocessing context used for both steps.
    backend : {'cpu', 'cupy'}, default='cpu'
        Compute backend for both steps. Use 'cupy' for GPU acceleration.
    structure : np.ndarray or None, optional
        Structuring element for the labeling step. None uses full connectivity
        (26-connectivity for 3D).
    """
    os.makedirs(tmp_dir, exist_ok=True)
    labeled_tmp_path = os.path.join(tmp_dir, "labeled_unmerged.zarr")

    max_label_value = label_chunks_parallel(
        input_path=input_path,
        output_path=labeled_tmp_path,
        chunk_shape=chunk_shape,
        n_processes=n_label_processes,
        pool_type=pool_type,
        backend=backend,
        structure=structure,
    )

    merge_touching_labels(
        label_image_path=labeled_tmp_path,
        output_image_path=output_path,
        chunk_shape=chunk_shape,
        max_label_value=max_label_value,
        n_processes=n_merge_processes,
        pool_type=pool_type,
        backend=backend,
    )
