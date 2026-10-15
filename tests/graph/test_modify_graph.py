from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from skeleplex.graph.constants import (
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    NODE_COORDINATE_KEY,
)
from skeleplex.graph.modify_graph import (
    delete_edge,
    length_pruning,
    merge_edge,
    merge_nodes,
    move_branch_point_along_edge,
    split_edge,
)
from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.graph.spline import B3Spline


def test_merge_edge(straight_edge_graph):
    """Test merging two edges."""
    merged_graph = deepcopy(straight_edge_graph)
    merge_edge(merged_graph, 0, 1, 2)
    assert merged_graph.graph.number_of_edges() == 1
    assert list(merged_graph.graph.nodes) == [0, 2]

    # assert if all the edge attributes are copied
    original_attributes = set()
    for _, _, attr in straight_edge_graph.graph.edges(data=True):
        original_attributes.update(attr.keys())
    merged_attributes = set()
    for _, _, attr in merged_graph.graph.edges(data=True):
        merged_attributes.update(attr.keys())
    assert original_attributes == merged_attributes


def test_delete_edge(simple_t_skeleton_graph):
    """Test deleting an edge."""

    delete_edge(simple_t_skeleton_graph, (1, 3))

    assert simple_t_skeleton_graph.graph.number_of_edges() == 1
    assert list(simple_t_skeleton_graph.graph.edges) == [(0, 2)]


def test_length_pruning(simple_t_skeleton_graph):
    """Test length pruning of an edge."""
    simple_t_skeleton_graph.to_directed(origin=0)
    length_pruning(simple_t_skeleton_graph, 10)
    assert simple_t_skeleton_graph == simple_t_skeleton_graph

    length_pruning(simple_t_skeleton_graph, 20)
    assert list(simple_t_skeleton_graph.graph.edges) == [(0, 3)]


def test_split_edge(simple_t_skeleton_graph):
    """Test splitting an edge."""
    assert simple_t_skeleton_graph.graph.number_of_edges() == 3
    assert list(simple_t_skeleton_graph.graph.in_edges(1)) == [(0, 1)]
    split_edge(simple_t_skeleton_graph, (0, 1), 0.5)
    assert simple_t_skeleton_graph.graph.number_of_edges() == 4
    assert list(simple_t_skeleton_graph.graph.in_edges(1)) == [(4, 1)]


def test_move_branch_point_along_edge(simple_t_skeleton_graph):
    """Test moving a branch point along an edge."""
    obj = simple_t_skeleton_graph
    np.testing.assert_allclose(
        obj.graph.nodes[1][NODE_COORDINATE_KEY], np.array([10, 10, 0])
    )
    move_branch_point_along_edge(obj, 1, (0, 1), (1, 2), (1, 3), 0.5)
    np.testing.assert_allclose(
        obj.graph.nodes[1][NODE_COORDINATE_KEY], np.array([10, 6.666, 0]), atol=0.001
    )


def _two_fragment_graph(directed: bool) -> SkeletonGraph:
    """Two collinear edges 0-1 and 2-3 separated by a break between 1 and 2."""
    graph = nx.DiGraph() if directed else nx.Graph()
    for node in range(4):
        graph.add_node(node, **{NODE_COORDINATE_KEY: np.array([0, 0, 10.0 * node])})
    for u, v in ((0, 1), (2, 3)):
        points = np.linspace([0, 0, 10.0 * u], [0, 0, 10.0 * v], 6)
        graph.add_edge(
            u,
            v,
            **{
                EDGE_COORDINATES_KEY: points,
                EDGE_SPLINE_KEY: B3Spline.from_points(points),
            },
        )
    return SkeletonGraph(graph=graph)


@pytest.mark.parametrize("directed", [False, True])
def test_merge_nodes(directed):
    """Merging an end point bridges the break and removes the merged node.

    On undirected graphs the neighbour of the merged node used to be taken as
    the node itself, so the merge always failed.
    """
    skeleton_graph = _two_fragment_graph(directed)

    merge_nodes(skeleton_graph, node_to_keep=1, node_to_merge=2)

    graph = skeleton_graph.graph
    assert 2 not in graph
    assert set(graph.nodes) == {0, 1, 3}
    assert graph.number_of_edges() == 2
    assert graph.has_edge(1, 3) or graph.has_edge(3, 1)
    assert nx.number_connected_components(graph.to_undirected()) == 1


@pytest.mark.parametrize("first_node", [0, 5])
def test_delete_edge_multigraph(first_node):
    """Deleting an edge works on a MultiGraph, as built from a skeleton image.

    The edges of a MultiGraph are (u, v, key) triples. The key used to be
    treated as a node, so the deletion failed when node 0 did not exist or
    was removed with the edge.
    """
    a, b, c, d = (first_node + i for i in range(4))
    graph = nx.MultiGraph()
    for node in (a, b, c, d):
        graph.add_node(node, **{NODE_COORDINATE_KEY: np.array([0.0, 0.0, float(node)])})
    for u, v in ((a, b), (b, c), (b, d)):
        points = np.linspace(
            graph.nodes[u][NODE_COORDINATE_KEY], graph.nodes[v][NODE_COORDINATE_KEY], 6
        )
        graph.add_edge(
            u,
            v,
            **{
                EDGE_COORDINATES_KEY: points,
                EDGE_SPLINE_KEY: B3Spline.from_points(points),
            },
        )
    skeleton_graph = SkeletonGraph(graph=graph)

    # the terminal edge at the first node: the node is removed, b is merged away
    delete_edge(skeleton_graph, (a, b))

    assert set(skeleton_graph.graph.nodes) == {c, d}
    assert skeleton_graph.graph.number_of_edges() == 1
