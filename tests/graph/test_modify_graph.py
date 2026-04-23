from copy import deepcopy

import numpy as np

from skeleplex.graph.constants import (
    NODE_COORDINATE_KEY,
)
from skeleplex.graph.modify_graph import (
    delete_edge,
    length_pruning,
    merge_edge,
    move_branch_point_along_edge,
    split_edge,
)


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


def test_delete_edge_directed(simple_t_skeleton_graph):
    """Test deleting an edge."""

    delete_edge(simple_t_skeleton_graph, (1, 3))

    assert simple_t_skeleton_graph.graph.number_of_edges() == 1
    assert list(simple_t_skeleton_graph.graph.edges) == [(0, 2)]

def test_delete_edge_undirected(simple_t_skeleton_graph):
    """Test deleting an edge."""
    simple_t_skeleton_graph.graph = simple_t_skeleton_graph.graph.to_undirected()
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
