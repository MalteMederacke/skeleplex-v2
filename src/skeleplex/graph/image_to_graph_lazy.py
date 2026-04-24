"""
Functions for creating a graph from a skeleton image using lazy computation.

These functions are adapted from Genevieve Buckley's distributed-skeleton-analysis repo:
https://github.com/GenevieveBuckley/distributed-skeleton-analysis
"""

import functools
import operator
from itertools import product

import dask.array as da
import dask.dataframe as dd
import networkx as nx
import numpy as np
import pandas as pd
import scipy.ndimage
from dask import compute, delayed
from dask.array.slicing import cached_cumsum
from dask_image.ndfilters import convolve
from dask_image.ndmeasure import label
from scipy import sparse
from skan import Skeleton, summarize
from skan.csr import _build_skeleton_path_graph, _write_pixel_graph, csr_to_nbgraph
from skan.nputil import raveled_steps_to_neighbors


def compute_degrees(skeleton_image: da.Array) -> da.Array:
    """
    Compute the degrees image from a binary skeleton image.

    This function counts, for each voxel in the skeleton,
    how many of its neighbors also belong to the skeleton.
    It performs a convolution of the binary skeleton image with
    a 3x3x3 kernel to compute the neighbor count.

    Parameters
    ----------
    skeleton_image : dask.array.Array
        Binary skeleton image where non-zero values indicate skeleton voxels.

    Returns
    -------
    degrees_image : dask.array.Array
        An image of the same shape as skeleton_image. Each foreground voxel
        stores the count of neighboring skeleton voxels.
    """
    ndim = skeleton_image.ndim
    degree_kernel = np.ones((3,) * ndim, dtype=np.uint8)
    degree_kernel[(1,) * ndim] = 0

    degrees_image = convolve(
        skeleton_image.astype(np.uint8), degree_kernel, mode="constant"
    )

    degrees_image = degrees_image * skeleton_image

    return degrees_image


def remove_isolated_voxels(
    skeleton_image: da.Array,
    degrees_image: da.Array,
) -> da.Array:
    """
    Remove isolated voxels from the skeleton image using the degrees image.

    This function removes skeleton voxels whose degree is 0,
    i.e. voxels that do not have any neighboring skeleton voxels.

    Parameters
    ----------
    skeleton_image : dask.array.Array
        Binary skeleton image where non-zero values indicate skeleton voxels.
    degrees_image : dask.array.Array
        Degrees image where each voxel stores the number of neighboring skeleton voxels.

    Returns
    -------
    cleaned_skeleton : dask.array.Array
        Binary skeleton image with isolated voxels removed.
    """
    cleaned_skeleton = skeleton_image & (degrees_image > 0)

    return cleaned_skeleton


def assign_unique_ids(
    skeleton_image: da.Array,
) -> tuple[da.Array, int]:
    """
    Assign a unique integer label to each skeleton voxel in a binary skeleton image.

    Each foreground voxel in the skeleton is assigned a unique integer ID
    starting from 1 using a 3x3x3 kernel. Background voxels remain 0. This
    effectively enumerates all skeleton voxels.

    Parameters
    ----------
    skeleton_image : dask.array.Array
        Binary skeleton image where non-zero values indicate skeleton voxels.

    Returns
    -------
    labeled_skeleton : dask.array.Array
        Array of same shape as skeleton_image with unique integer labels
        assigned to each connected component.
    num_features : int
        The total number of skeleton voxels labeled.
    """
    ndim = skeleton_image.ndim
    structure_kernel = np.zeros((3,) * ndim)
    structure_kernel[(1,) * ndim] = 1

    labeled_skeleton, num_features = label(
        skeleton_image,
        structure=structure_kernel,
    )

    return labeled_skeleton, num_features


@delayed
def skeleton_graph_func(
    labeled_skeleton_chunk: np.ndarray, spacing: float = 1
) -> pd.DataFrame:
    """
    Create delayed computation to extract connections from a labeled skeleton chunk.

    This function analyzes a chunk of a labeled skeleton image to identify
    connections between neighboring skeleton voxels. It constructs a DataFrame
    listing each pair of connected voxels and the Euclidean distance
    between them. The computation is delayed and only executes when explicitly computed.

    Parameters
    ----------
    labeled_skeleton_chunk : np.ndarray
        A chunk of the labeled skeleton image. Each voxel contains either
        0 or a unique positive integer label assigned to a skeleton voxel.
    spacing : float, optional
        Spacing between voxels in the skeleton, by default 1.

    Returns
    -------
    dask.delayed.Delayed
        A delayed computation producing a pandas DataFrame with columns:
            - 'row' : int
                Label of the source voxel.
            - 'col' : int
                Label of the neighboring target voxel.
            - 'data' : float
                Euclidean distance between the connected voxels.
    """
    ndim = labeled_skeleton_chunk.ndim
    spacing = np.ones(ndim, dtype=float) * spacing
    num_edges = _num_edges(labeled_skeleton_chunk.astype(bool))
    padded_chunk = np.pad(labeled_skeleton_chunk, 1)
    steps, distances = raveled_steps_to_neighbors(
        padded_chunk.shape, ndim, spacing=spacing
    )

    row = np.empty(num_edges, dtype=int)
    col = np.empty(num_edges, dtype=int)
    data = np.empty(num_edges, dtype=float)
    _write_pixel_graph(padded_chunk, steps, distances, row, col, data)

    return pd.DataFrame({"row": row, "col": col, "data": data})


def _num_edges(skeleton_chunk: np.ndarray) -> int:
    """
    Compute the total number of connected voxels of a skeleton chunk.

    This function calculates how many neighbor connections exist
    in total within the given chunk.

    Parameters
    ----------
    skeleton_chunk : np.ndarray
        Binary array representing a skeleton chunk, where nonzero voxels
        indicate skeleton voxels.

    Returns
    -------
    int
        Total number of edges (neighbor connections) in the skeleton chunk.
    """
    ndim = skeleton_chunk.ndim
    degree_kernel = np.ones((3,) * ndim)
    degree_kernel[(1,) * ndim] = 0
    degree_image = (
        scipy.ndimage.convolve(
            skeleton_chunk.astype(int), degree_kernel, mode="constant"
        )
        * skeleton_chunk
    )
    num_edges = np.sum(degree_image)

    return int(num_edges)


def slices_from_chunks_overlap(
    chunks: tuple[tuple[int, ...], ...], array_shape: tuple[int, ...], depth: int = 1
) -> list[tuple[slice, ...]]:
    """
    Compute slices for extracting overlapping chunks from an array.

    Given the chunk structure of a Dask array, this function generates a list
    of slice tuples that define how to extract each chunk along with a border
    of overlapping voxels around it. The overlap is applied on all sides,
    except where the chunk is at the array edge.

    Parameters
    ----------
    chunks : tuple of tuples of int
        The chunk sizes in each dimension, as returned by Dask's `.chunks` attribute.
        For example: ((4,), (7, 7))
    array_shape : tuple of int
        Shape of the full array.
    depth : int, optional
        Number of voxels to include as overlap around each chunk.
        Defaults to 1.

    Returns
    -------
    list of tuple of slice
        A list of tuples of slice objects, one per chunk. Each tuple can be
        used to index into the array to extract that chunk plus its overlap.

    Example
    -------
    >>> slices_from_chunks_overlap(((4,), (7, 7)), (4, 14), depth=1)
    [(slice(0, 5, None), slice(0, 8, None)),
     (slice(0, 5, None), slice(6, 15, None))]
    """
    cumdims = [cached_cumsum(bds, initial_zero=True) for bds in chunks]

    slices = []
    for starts, shapes, maxshape in zip(cumdims, chunks, array_shape, strict=False):
        inner_slices = []
        for s, dim in zip(starts, shapes, strict=False):
            slice_start = s
            slice_stop = s + dim
            if slice_start > 0:
                slice_start -= depth
            if slice_stop >= maxshape:
                slice_stop += depth
            inner_slices.append(slice(slice_start, slice_stop))
        slices.append(inner_slices)

    return list(product(*slices))


def construct_dataframe(labeled_skeleton_image: da.Array) -> dd.DataFrame:
    """
    Construct a Dask DataFrame of connected voxels from a labeled skeleton image.

    This function processes a labeled skeleton image in overlapping chunks.
    For each chunk, it computes connections between neighboring labeled voxels and
    returns the combined result as a single Dask DataFrame.

    Parameters
    ----------
    labeled_skeleton_image : dask.array.Array
        Labeled skeleton image where each nonzero voxel has a unique integer label.
    overlap_depth : int, optional
        Number of voxels to overlap between neighboring chunks
        when extracting chunks from the array. Default is 1.

    Returns
    -------
    dask.dataframe.DataFrame
        Dask DataFrame with graph edges.
        Columns:
            - 'row': int
                Label of the source voxel.
            - 'col': int
                Label of the neighboring target voxel.
            - 'data': float
                Euclidean distance between the connected voxels.
    """
    chunk_iterator = zip(
        np.ndindex(*labeled_skeleton_image.numblocks),
        map(
            functools.partial(operator.getitem, labeled_skeleton_image),
            slices_from_chunks_overlap(
                labeled_skeleton_image.chunks, labeled_skeleton_image.shape, depth=1
            ),
        ),
        strict=False,
    )

    meta = dd.utils.make_meta(
        [("row", np.int64), ("col", np.int64), ("data", np.float64)]
    )
    chunk_graphs = [
        dd.from_delayed(skeleton_graph_func(chunk), meta=meta)
        for _, chunk in chunk_iterator
    ]

    graph_edges_ddf = dd.concat(chunk_graphs)
    graph_edges_ddf = graph_edges_ddf.drop_duplicates()

    return graph_edges_ddf


def build_pixel_indices(skeleton_image: da.Array) -> np.ndarray:
    """
    Compute global coordinates of all nonzero skeleton voxels.

    Parameters
    ----------
    skeleton_image : dask.array.Array
        Binary skeleton image.

    Returns
    -------
    np.ndarray
        Array of voxel coordinates with shape (N, ndim).
    """

    def extract_coords_block(block, key):
        coords = np.argwhere(block > 0)
        if coords.size == 0:
            return np.empty((0, block.ndim), dtype=np.int32)
        offset = np.array([k * s for k, s in zip(key, block.shape, strict=False)])
        return coords + offset

    delayed_blocks = skeleton_image.to_delayed().flatten()
    delayed_coords = [
        delayed(extract_coords_block)(block, block.key[1:]) for block in delayed_blocks
    ]

    coords_list = compute(*delayed_coords)
    all_coords = np.vstack(coords_list)

    pixel_indices = np.concatenate(
        ([np.zeros(skeleton_image.ndim)], all_coords), axis=0
    )

    return pixel_indices


def skeleton_image_to_graph(
    skeleton_image: da.Array,
    degrees_image: da.Array,
    graph_edges_df: pd.DataFrame,
    spacing: float = 1.0,
    image_voxel_size_um: float = 1.0,
) -> nx.Graph:
    """
    Build a NetworkX graph from a skeleton image and graph edges.

    This function constructs a NetworkX graph directly from a
    skeleton image, its degrees image, and the edge list DataFrame.
    It computes the voxel coordinates for each skeleton node and stores
    them as node attributes in the graph. Edge paths are attached as
    edge attributes.

    The node coordinates and edge path coordinates are scaled to microns
    using the provided image_voxel_size_um.

    Parameters
    ----------
    skeleton_image : dask.array.Array
        Binary skeleton image.
    degrees_image : dask.array.Array
        Degrees image indicating the number of neighbors for each voxel.
    graph_edges_df : pandas.DataFrame
        DataFrame of graph edges with columns: 'row', 'col', 'data'.
    spacing : float, optional
        Spacing between voxels. Default is 1.0.
        This does not scale the graph to um.
    image_voxel_size_um : float, optional
        Spacing of the voxels. Used to transform graph coordinates to um.
        Default is 1.0.

    Returns
    -------
    nx_graph : networkx.Graph
        NetworkX graph representing the skeleton.
        - Each edge has a 'path' attribute (Nx3 array of coordinates).
        - Each node has a 'node_coordinate' attribute (1x3 array).
    """
    row = graph_edges_df["row"].values
    col = graph_edges_df["col"].values
    data = graph_edges_df["data"].values
    adj = sparse.coo_matrix((data, (row, col))).tocsr()

    skel_obj = Skeleton(np.eye(3))
    skel_obj.skeleton_image = skeleton_image
    # this doesnt scale the coordinates
    skel_obj.spacing = [spacing] * skeleton_image.ndim
    skel_obj.graph = adj
    skel_obj.degrees_image = degrees_image

    skel_obj.coordinates = build_pixel_indices(skeleton_image)

    nonzero_degree_values = degrees_image[degrees_image > 0].compute()
    degrees = np.concatenate(([0], nonzero_degree_values))
    skel_obj.degrees = degrees

    nbgraph = csr_to_nbgraph(adj)
    skel_obj.nbgraph = nbgraph
    paths = _build_skeleton_path_graph(nbgraph)
    skel_obj.paths = paths
    skel_obj.n_paths = paths.shape[0]
    skel_obj._distances_initialized = False
    skel_obj.distances = np.empty(skel_obj.n_paths, dtype=float)
    skel_obj.path_lengths()

    summary_df = summarize(skel_obj, separator="_")

    nx_graph = nx.Graph()

    for row in summary_df.itertuples(name="Edge"):
        index = row.Index
        node_src = row.node_id_src
        node_dst = row.node_id_dst

        # path coordinates are not scaled to voxel size.
        # here we scale them to microns
        edge_coords = skel_obj.path_coordinates(index)
        edge_coords = edge_coords * image_voxel_size_um  # scale to um

        # Ensure enough points for a well-conditioned B3 spline fit.
        # Resample along the actual path rather than only using endpoints,
        # so curved short branches are not flattened into a straight line.
        min_points = 20
        if len(edge_coords) < min_points:
            # build a cumulative arc-length parameterisation of the raw path
            deltas = np.diff(edge_coords, axis=0)
            seg_lengths = np.linalg.norm(deltas, axis=1)
            # guard against degenerate zero-length edges
            if seg_lengths.sum() == 0:
                seg_lengths = np.ones(len(seg_lengths))
            cumlen = np.concatenate(([0], np.cumsum(seg_lengths)))
            cumlen /= cumlen[-1]
            t_out = np.linspace(0, 1, min_points)
            edge_coords = np.column_stack(
                [np.interp(t_out, cumlen, edge_coords[:, d]) for d in range(3)]
            )

        nx_graph.add_edge(
            node_src,
            node_dst,
            path=edge_coords,
        )

    # add node coordinates
    for node_index in nx_graph.nodes():
        node_coord = (
            np.asarray(skel_obj.coordinates[node_index]) * image_voxel_size_um
        )  # scale to um
        nx_graph.nodes[node_index]["node_coordinate"] = node_coord

    return nx_graph
