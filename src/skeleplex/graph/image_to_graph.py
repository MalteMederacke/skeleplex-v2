"""Utilities to convert a skeleton image to a graph."""

import networkx as nx
import numpy as np
from skan.csr import Skeleton as SkanSkeleton
from skan.csr import summarize

from skeleplex.graph.constants import (
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    NODE_COORDINATE_KEY,
)
from skeleplex.graph.spline import B3Spline


def image_to_graph_skan(
    skeleton_image: np.ndarray,
    max_spline_knots: int = 10,
    image_voxel_size_um: float = 1,
) -> nx.MultiGraph:
    """Convert a skeleton image to a graph using skan.

    Parameters
    ----------
    skeleton_image : np.ndarray
        The image to convert to a skeleton graph.
        The image should be a binary image and already skeletonized.
    max_spline_knots : int
        The maximum number of knots to use for the spline fit to the branch path.
        If the number of data points in the branch is less than this number,
        the spline will use n_data_points - 1 knots.
        See the splinebox Spline class docs for more information.
    image_voxel_size_um  : float or array of float
        Spacing of the voxels. Used to transform graph coordinates to um.
    """
    # make the skeleton
    skeleton = SkanSkeleton(skeleton_image=skeleton_image, spacing=image_voxel_size_um)
    summary_table = summarize(skeleton, separator="_")

    # get all of the nodes
    # this might be slow - may need to speed up
    # source_nodes = set(summary_table["node_id_src"])
    # destination_nodes = set(summary_table["node_id_dst"])
    # all_nodes = source_nodes.union(destination_nodes)

    skeleton_graph = nx.MultiGraph()
    for row in summary_table.itertuples(name="Edge"):
        # Iterate over the rows in the table.
        # Each row is an edge in the graph
        index = row.Index
        i = row.node_id_src
        j = row.node_id_dst

        # fit a spline to the path
        # todo: factor our to spline module
        # todo: reconsider how the number of knots is set
        spline_path = skeleton.path_coordinates(index)
        spline_path = spline_path * image_voxel_size_um  # scale to um
        n_points = len(spline_path)
        if n_points <= max_spline_knots:
            n_spline_knots = n_points - 1
            if n_spline_knots <= 3:  # min for b3 spline
                # resample along the actual path so curved short branches
                # are not flattened to a straight line
                deltas = np.diff(spline_path, axis=0)
                seg_lengths = np.linalg.norm(deltas, axis=1)
                if seg_lengths.sum() == 0:
                    seg_lengths = np.ones(len(seg_lengths))
                cumlen = np.concatenate(([0], np.cumsum(seg_lengths)))
                cumlen /= cumlen[-1]
                t_out = np.linspace(0, 1, max_spline_knots)
                spline_path = np.column_stack(
                    [np.interp(t_out, cumlen, spline_path[:, d]) for d in range(3)]
                )
                n_spline_knots = max_spline_knots - 1
        else:
            n_spline_knots = max_spline_knots
        spline = B3Spline.from_points(
            points=spline_path,
            n_knots=n_spline_knots,
        )
        # Nodes are added if they don't exist so only need to add edges
        skeleton_graph.add_edge(
            i,
            j,
            **{EDGE_COORDINATES_KEY: spline_path, EDGE_SPLINE_KEY: spline},
        )

    # add the node coordinates
    new_node_data = {}
    for node_index, node_data in skeleton_graph.nodes(data=True):
        node_coordinate = np.asarray(skeleton.coordinates[node_index])
        node_coordinate = node_coordinate * image_voxel_size_um  # scale to um
        node_data[NODE_COORDINATE_KEY] = node_coordinate
        new_node_data[node_index] = node_data

    nx.set_node_attributes(skeleton_graph, new_node_data)

    return skeleton_graph
