import logging  # noqa: D100

import networkx as nx
import numpy as np

from skeleplex.graph.constants import (
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    END_NODE_KEY,
    GENERATION_KEY,
    LENGTH_KEY,
    NODE_COORDINATE_KEY,
    START_NODE_KEY,
    VALIDATION_KEY,
)
from skeleplex.graph.skeleton_graph import SkeletonGraph, get_next_node_key
from skeleplex.graph.spline import B3Spline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def merge_edge(skeleton_graph: SkeletonGraph, n1: int, v1: int, n2: int):
    """Merge edges in graph and add edge attributes.

    n1 is merged with n2. v1 is removed.

                "n1 = 1, v1 = 2, n2 = 3"
    1               1
    |               |
    |               |
    2   -->         |
    |               |
    |               |
    3               3

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        The graph to merge the edges in.
    n1 : int
        The start node of the first edge.
    v1 : int
        The end node of the first edge and the start node of the second edge.
        This node is removed.
    n2 : int
        The end node of the second edge.


    """
    graph = skeleton_graph.graph.copy()

    start_node = graph.nodes(data=True)[n1][NODE_COORDINATE_KEY]
    end_node = graph.nodes(data=True)[n2][NODE_COORDINATE_KEY]
    middle_node = graph.nodes(data=True)[v1][NODE_COORDINATE_KEY]

    if graph.degree(v1) != 2:
        raise ValueError(
            f"Node {v1} has degree {graph.degree(middle_node)}. "
            "Only nodes with degree 2 can be merged."
        )

    # adapt for directed graphs
    if graph.has_edge(n1, v1) and graph.has_edge(v1, n2):
        edge1 = (n1, v1)
        edge2 = (v1, n2)
        merge_edge = (n1, n2)
    # first flipped
    elif graph.has_edge(v1, n2) and graph.has_edge(n1, v1):
        edge1 = (v1, n2)
        edge2 = (n1, v1)
        merge_edge = (n2, n1)
    # second flipped
    elif graph.has_edge(n1, v1) and graph.has_edge(n2, v1):
        edge1 = (n1, v1)
        edge2 = (n2, v1)
        merge_edge = (n1, n2)
    # both flipped
    elif graph.has_edge(v1, n2) and graph.has_edge(n2, v1):
        edge1 = (v1, n2)
        edge2 = (n2, v1)
        merge_edge = (n1, n2)
    else:
        raise ValueError(
            f"Edges {n1, v1} and {v1, n2} not found in graph. "
            "Are you trying to merge nodes that are not connected?"
        )

    # edge1 = (start_node, middle_node)
    # edge2 = (middle_node, end_node)

    edge_attributes1 = graph.get_edge_data(edge1[0], edge1[1])
    edge_attributes2 = graph.get_edge_data(edge2[0], edge2[1])
    graph.remove_edge(edge1[0], edge1[1])
    graph.remove_edge(edge2[0], edge2[1])
    graph.remove_node(edge1[1])
    # merge_edge = (edge1[0], edge2[1])
    merge_attributes = {}
    for key in edge_attributes1:
        if key == VALIDATION_KEY:
            if edge_attributes1[key] and edge_attributes2[key]:
                merge_attributes[key] = True
            else:
                merge_attributes[key] = False

        if key == EDGE_SPLINE_KEY:
            points1 = edge_attributes1[EDGE_COORDINATES_KEY]
            points2 = edge_attributes2[EDGE_COORDINATES_KEY]

            # start and end node coordinates
            # this checking is probably not necessary as we use directed graphs
            # but just to be sure
            if np.allclose(points1[0], start_node) & np.allclose(
                points2[0], middle_node
            ):
                logger.info("None of the edges need to be flipped")
                spline_points = np.vstack((points1, points2))

            elif np.allclose(points1[-1], start_node) & np.allclose(
                points2[0], middle_node
            ):
                logger.info(f"flip edge {edge1}")
                spline_points = np.vstack((np.flip(points1, axis=0), points2))
            elif np.allclose(points1[0], start_node) & np.allclose(
                points2[-1], middle_node
            ):
                logger.info(f"flip {edge2}")
                spline_points = np.vstack((points1, np.flip(points2, axis=0)))
            elif np.allclose(points1[-1], start_node) & np.allclose(
                points2[-1], middle_node
            ):
                logger.info(f"flip {edge1} and {edge2}")
                spline_points = np.vstack(
                    (np.flip(points1, axis=0), np.flip(points2, axis=0))
                )
            else:
                logger.warning("Warning: Edge splines not connected.")
                spline_points = np.vstack((points1, points2))
            # sanity check
            if np.allclose(spline_points[-1], end_node):
                logger.info("sanity check passed")

            _, idx = np.unique(spline_points, axis=0, return_index=True)
            spline_points = spline_points[np.sort(idx)]
            spline = B3Spline.from_points(spline_points)
            merge_attributes[key] = spline
            merge_attributes[EDGE_COORDINATES_KEY] = spline_points
        if key == START_NODE_KEY:
            merge_attributes[key] = n1
        if key == END_NODE_KEY:
            merge_attributes[key] = n2
        if key == GENERATION_KEY:
            merge_attributes[key] = edge_attributes1[key]

        if key == LENGTH_KEY:
            merge_attributes[key] = merge_attributes[EDGE_SPLINE_KEY].arc_length

        if key not in [
            VALIDATION_KEY,
            EDGE_COORDINATES_KEY,
            EDGE_SPLINE_KEY,
            START_NODE_KEY,
            END_NODE_KEY,
            GENERATION_KEY,
            LENGTH_KEY,
        ]:
            logger.warning(
                (f"Warning: Attribute {key} not merged. ", "Consider recomputing.")
            )

    graph.add_edge(*merge_edge, **merge_attributes)
    skeleton_graph.graph = graph


def delete_edge(skeleton_graph: SkeletonGraph, edge: tuple[int, int]):
    """Delete edge in skeleton graph.

    To maintain a dichotomous structure, the edge is deleted and the
    resulting degree 2 node is merged with its neighbors.

            "Delete (2,4)"
    1               1
    |               |
    |               |
    2------4   -->  |
    |               |
    |               |
    3               3

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        The graph to delete the edge from.
    edge : Tuple[int, int]
        The edge to delete.
    """
    # check if directed
    # copy graph
    graph = skeleton_graph.graph.copy()
    graph.remove_edge(*edge)

    # detect all changes
    changed_edges = set(skeleton_graph.graph.edges) - set(graph.edges)
    skeleton_graph.graph = graph
    for edge in changed_edges:
        for node in edge:
            if skeleton_graph.graph.degree(node) == 0:
                skeleton_graph.graph.remove_node(node)
            # merge edges if node has degree 2
            elif skeleton_graph.graph.degree(node) == 2:
                if skeleton_graph.graph.is_directed():

                    # merge
                    in_edge = list(skeleton_graph.graph.in_edges(node))
                    out_edge = list(skeleton_graph.graph.out_edges(node))
                    if len(in_edge) == 0:
                        raise ValueError(
                            ("Deleting the edge would break the graph"),
                            "Are you trying to delete the origin?",
                        )

                    merge_edge(skeleton_graph, in_edge[0][0], node, out_edge[0][1])
                    logger.info("merge")
                else:
                    neighbors = list(skeleton_graph.graph.neighbors(node))
                    if len(neighbors) != 2:
                        raise ValueError(
                            "Node should have exactly 2 neighbors for merging"
                        )
                    # For undirected graphs, merge the two neighbors through this node
                    merge_edge(skeleton_graph, neighbors[0], node, neighbors[1])
                    logger.info("merge")

    # check if graph is still connected, if not remove orphaned nodes
    skeleton_graph.graph.remove_nodes_from(list(nx.isolates(skeleton_graph.graph)))


def length_pruning(skeleton_graph: SkeletonGraph, length_threshold: int):
    """Prune all edges with length below threshold.

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        The graph to prune.
    length_threshold : int
        The threshold for the length of the edges.


    """
    graph = skeleton_graph.graph
    g_unmodified = graph.copy()

    # check if length is already computed
    if len(nx.get_edge_attributes(graph, LENGTH_KEY)) == 0:
        len_dict = skeleton_graph.compute_branch_lengths()
        nx.set_edge_attributes(graph, len_dict, LENGTH_KEY)

    for node, degree in g_unmodified.degree():
        if (degree == 1) and (node != skeleton_graph.origin):
            edge = next(iter(graph.in_edges(node)))
            path_length = graph.edges[edge[0], edge[1]].get(LENGTH_KEY)
            # start_node = edge[0]
            if path_length < length_threshold:
                # check if edge still exists in original graph
                if edge not in skeleton_graph.graph.edges:
                    continue
                try:
                    delete_edge(skeleton_graph, edge)
                    logger.info(f"Deleted {edge}")
                except KeyError:
                    logger.warning(f"Edge {edge} not found in graph, could not delete.")
                except Exception as e:
                    logger.error(f"Unexpected error while deleting {edge}: {e}")


def prune_degree_2_nodes(skeleton_graph: SkeletonGraph):
    """Remove all degree-2 nodes by merging their incoming and outgoing edges.

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        The graph to prune.

    """
    graph = skeleton_graph.graph.copy()
    origin = skeleton_graph.origin

    nodes_to_merge = []
    for node in graph.nodes:
        if node == origin:
            continue
        if graph.degree(node) == 2:
            in_edges = list(graph.in_edges(node))
            out_edges = list(graph.out_edges(node))
            if len(in_edges) == 1 and len(out_edges) == 1:
                u = in_edges[0][0]
                w = out_edges[0][1]
                nodes_to_merge.append((u, node, w))

    logger.info(f"Found {len(nodes_to_merge)} degree-2 nodes to merge.")

    for u, v, w in nodes_to_merge:
        try:
            merge_edge(skeleton_graph, u, v, w)  # just calling, not assigning
            logger.info(f"Merged node {v} between {u} and {w}")
        except Exception as e:
            logger.warning(f"Could not merge node {v} between {u} and {w}: {e}")

    # reload graph because merge_edge modifies skeleton_graph.graph
    graph = skeleton_graph.graph

    # remove any isolates
    graph.remove_nodes_from(list(nx.isolates(graph)))

    # save updated graph back
    skeleton_graph.graph = graph


def split_edge(
    skeleton_graph: SkeletonGraph, edge_to_split_ID: tuple, split_pos: float
):
    """Split an edge at a given position.

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        The skeleton graph object.
    edge_to_split_ID : Tuple
        The edge to split.
    split_pos : float
        The position to split the edge at. Normalized between 0 and 1.
    """
    # test if edge is in tree
    if edge_to_split_ID not in skeleton_graph.graph.edges:
        ValueError(f"Edge {edge_to_split_ID} not in graph.")
    graph = skeleton_graph.graph.copy()
    spline = skeleton_graph.graph.edges[edge_to_split_ID][EDGE_SPLINE_KEY]
    edge_coordinates = graph.edges[edge_to_split_ID][EDGE_COORDINATES_KEY]
    coordinate_to_split = spline.eval(split_pos)
    split_index = np.argmin(
        np.linalg.norm(edge_coordinates - coordinate_to_split, axis=1)
    )
    new_node_number = get_next_node_key(graph)
    graph.add_node(new_node_number, node_coordinate=coordinate_to_split)
    new_edge_coords1 = edge_coordinates[: split_index + 1]
    if len(new_edge_coords1) < 4:
        logger.warning(
            f"Edge {edge_to_split_ID} is to short to split at {split_pos}."
            " Approximate edge as line."
        )
        new_edge_coords1 = np.linspace(new_edge_coords1[0], new_edge_coords1[-1], 5)
    new_edge_dict1 = {
        EDGE_COORDINATES_KEY: new_edge_coords1,
        START_NODE_KEY: new_node_number,
        END_NODE_KEY: edge_to_split_ID[1],
        EDGE_SPLINE_KEY: B3Spline.from_points(
            new_edge_coords1,
        ),
    }
    graph.add_edge(edge_to_split_ID[0], new_node_number, **new_edge_dict1)

    # add second piece
    new_edge_coords2 = edge_coordinates[split_index:]
    if len(new_edge_coords2) < 4:
        logger.warning(
            f"Edge {edge_to_split_ID} is to short to split at {split_pos}."
            " Approximate edge as line."
        )
        new_edge_coords2 = np.linspace(new_edge_coords2[0], new_edge_coords2[-1], 5)

    new_edge_dict2 = {
        EDGE_COORDINATES_KEY: new_edge_coords2,
        START_NODE_KEY: edge_to_split_ID[0],
        END_NODE_KEY: new_node_number,
        EDGE_SPLINE_KEY: B3Spline.from_points(
            new_edge_coords2,
        ),
    }

    graph.add_edge(new_node_number, edge_to_split_ID[1], **new_edge_dict2)

    graph.remove_edge(*edge_to_split_ID)

    skeleton_graph.graph = graph


def move_branch_point_along_edge(
    skeleton_graph: SkeletonGraph,
    node: int,
    edge_to_shorten: tuple,
    edge_to_elongate: tuple,
    edge_to_remodel: tuple,
    distance,
):
    """
    Move Branch point along edge.

    Moves the branch point along the edge_to_shorten by distance and
    splits the edge_to_shorten at the new position. The second daughter edge is
    remoddled as a straight line.

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        The skeleton graph object.
    node : int
        The node to move.
    edge_to_shorten : Tuple
        The edge to shorten.
    edge_to_elongate : Tuple
        The edge to elongate.
    edge_to_remodel : Tuple
        The edge to remodel.
    distance : float
        The distance to move the node along the edge_to_shorten.
        Normalized between 0 and 1.
    """
    graph = skeleton_graph.graph
    spline_to_shorten = graph.edges[edge_to_shorten][EDGE_SPLINE_KEY]
    # move branch point
    point_on_edge_to_shorten = spline_to_shorten.eval(distance)

    # split edge_coordinates into two parts at the point_on_in_edge
    edge_to_shoorten_coords = graph.edges[edge_to_shorten][EDGE_COORDINATES_KEY]
    # rather get the closest point
    idx = np.argmin(
        np.linalg.norm(edge_to_shoorten_coords - point_on_edge_to_shorten, axis=1)
    )
    # split
    edge_coordinates_1 = edge_to_shoorten_coords[: idx + 1]
    edge_coordinates_2 = edge_to_shoorten_coords[idx:]

    # check if edge is long enough
    if len(edge_coordinates_1) < 4:
        logger.warning(
            f"Edge {edge_to_shorten} is to short to split at {distance}."
            " Approximate edge as line."
        )
        edge_coordinates_1 = np.linspace(
            edge_coordinates_1[0], edge_coordinates_1[-1], 5
        )
    out_edge_coordinates = graph.edges[edge_to_elongate][EDGE_COORDINATES_KEY]

    # maybe it makes sense to take the sampled point of the spline,
    # instead it takes the underlying coordinates. Could be more robust
    new_node_pos = edge_to_shoorten_coords[idx]

    new_out_edge_coordinates = np.concatenate(
        (edge_coordinates_2, out_edge_coordinates)
    )

    # filter duplicates
    new_out_edge_coordinates, indices = np.unique(
        new_out_edge_coordinates, axis=0, return_index=True
    )
    new_out_edge_coordinates = new_out_edge_coordinates[np.argsort(indices)]

    new_in_edge_coordinates = edge_coordinates_1

    # update edges
    graph.edges[edge_to_shorten][EDGE_COORDINATES_KEY] = new_in_edge_coordinates
    graph.edges[edge_to_elongate][EDGE_COORDINATES_KEY] = new_out_edge_coordinates
    graph.nodes[node][NODE_COORDINATE_KEY] = new_node_pos

    # update splines
    graph.edges[edge_to_shorten][EDGE_SPLINE_KEY] = B3Spline.from_points(
        new_in_edge_coordinates
    )
    graph.edges[edge_to_elongate][EDGE_SPLINE_KEY] = B3Spline.from_points(
        new_out_edge_coordinates
    )

    # draw new edge for the other out edge
    out_edge_node2 = edge_to_remodel[1]
    out_edge_node2_pos = graph.nodes[out_edge_node2][NODE_COORDINATE_KEY]
    new_out_edge_coordinates = np.linspace(new_node_pos, out_edge_node2_pos, 10)

    # update edge
    graph.edges[edge_to_remodel][EDGE_COORDINATES_KEY] = new_out_edge_coordinates
    graph.edges[edge_to_remodel][EDGE_SPLINE_KEY] = B3Spline.from_points(
        new_out_edge_coordinates
    )

    # update graph
    skeleton_graph.graph = graph


def connect_without_merging(
    skeleton_graph: SkeletonGraph, node1: int, node2: int, return_graph: bool = False
):
    """Connect two nodes without merging them.

    A line is just drawn between the two nodes.

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        The skeleton graph object.
    node1 : int
        The first node to connect.
    node2 : int
        The second node to connect.
    return_graph : bool, optional
        If True, return the modified graph.
        Default is False.

    Returns
    -------
    nx.DiGraph
        The modified skeleton graph with the new edge added.
    """
    # check if directed
    if not skeleton_graph.graph.is_directed():
        ValueError("Graph is not directed. Convert to directed graph.")
    # copy graph
    graph = skeleton_graph.graph.copy()
    # check if nodes are already connected
    if graph.has_edge(node1, node2):
        logger.warning(f"Nodes {node1} and {node2} are already connected.")

    u_coordinates = skeleton_graph.graph.nodes[node1][NODE_COORDINATE_KEY]
    v_coordinates = skeleton_graph.graph.nodes[node2][NODE_COORDINATE_KEY]
    path = np.linspace(u_coordinates, v_coordinates, num=20)
    spline = B3Spline.from_points(path)

    g = skeleton_graph.graph.copy()
    g.add_edge(
        node1,
        node2,
        **{
            EDGE_COORDINATES_KEY: path,
            EDGE_SPLINE_KEY: spline,
            START_NODE_KEY: node1,
            END_NODE_KEY: node2,
        },
    )

    skeleton_graph.graph = g
    if return_graph:
        return skeleton_graph.graph


def merge_nodes(
    skeleton_graph: SkeletonGraph,
    node_to_keep: int,
    node_to_merge: int,
):
    """Merge two nodes in a graph.

    Parameters
    ----------
    skeleton_graph : nx.Graph
        The skeleton in which to perform the merging.
    node_to_keep : int
        The index of the merged nodes to keep in the graph.
    node_to_merge : int
        The index of the merged nodes that should be removed.


    """
    degree_of_merge = skeleton_graph.graph.degree(node_to_merge)
    if degree_of_merge != 1:
        raise ValueError(
            f"Node {node_to_merge} has degree {degree_of_merge}. "
            "Only nodes with degree 1 can be merged."
        )
    edge_to_merge = skeleton_graph.graph.in_edges(node_to_merge)
    second_node_to_keep = next(iter(edge_to_merge))[0]
    skeleton_graph.graph = connect_without_merging(
        skeleton_graph=skeleton_graph,
        node1=node_to_keep,
        node2=node_to_merge,
        return_graph=True,
    )

    merge_edge(
        skeleton_graph=skeleton_graph,
        n1=node_to_keep,
        v1=node_to_merge,
        n2=second_node_to_keep,
    )
