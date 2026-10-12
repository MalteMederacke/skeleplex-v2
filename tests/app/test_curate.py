"""Tests for the curation manager."""

import networkx as nx
import numpy as np
import pytest

from skeleplex.app._curate import CurationManager
from skeleplex.graph.constants import EDGE_COORDINATES_KEY, NODE_COORDINATE_KEY
from skeleplex.graph.skeleton_graph import SkeletonGraph


class _Data:
    """Minimal stand-in for the data manager."""

    def __init__(self, skeleton_graph: SkeletonGraph) -> None:
        self._skeleton_graph = skeleton_graph

    @property
    def skeleton_graph(self) -> SkeletonGraph:
        return self._skeleton_graph


def _manager(graph: nx.Graph) -> CurationManager:
    """Return a curation manager for a graph with the minimal attributes."""
    for node in graph.nodes:
        graph.nodes[node][NODE_COORDINATE_KEY] = np.array([0.0, 0.0, node])
    for u, v in graph.edges:
        graph.edges[u, v][EDGE_COORDINATES_KEY] = np.array(
            [[0.0, 0.0, u], [0.0, 0.0, v]]
        )
    return CurationManager(_Data(SkeletonGraph(graph=graph)))


def test_set_origin_directs_and_keeps_origin_component():
    """The origin component is directed away from the origin, the rest removed."""
    graph = nx.Graph([(0, 1), (1, 2), (1, 3), (10, 11)])
    manager = _manager(graph)

    summary = manager.set_origin(3, redraw=False)

    skeleton_graph = manager._data.skeleton_graph
    assert summary == {"components": 2, "removed_nodes": 2, "loops": 0}
    assert skeleton_graph.origin == 3
    assert skeleton_graph.graph.is_directed()
    assert set(skeleton_graph.graph.edges) == {(3, 1), (1, 0), (1, 2)}
    assert skeleton_graph.graph.in_degree(3) == 0


def test_set_origin_keeps_fragments_and_counts_loops():
    """Fragments can be kept; loops are counted before one edge each is dropped."""
    graph = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (10, 11)])
    manager = _manager(graph)

    summary = manager.set_origin(
        3, keep_only_origin_component=False, redraw=False
    )

    directed = manager._data.skeleton_graph.graph
    assert summary == {"components": 2, "removed_nodes": 0, "loops": 1}
    assert {10, 11} <= set(directed.nodes)
    # the loop 0-1-2 lost one edge
    assert directed.subgraph([0, 1, 2, 3]).number_of_edges() == 3


def test_set_origin_changes_origin_of_directed_graph():
    """A graph that is already directed can be given a new origin."""
    manager = _manager(nx.Graph([(0, 1), (1, 2)]))
    manager.set_origin(0, redraw=False)
    assert set(manager._data.skeleton_graph.graph.edges) == {(0, 1), (1, 2)}

    manager.set_origin(2, redraw=False)

    assert manager._data.skeleton_graph.origin == 2
    assert set(manager._data.skeleton_graph.graph.edges) == {(2, 1), (1, 0)}


def test_set_origin_unknown_node_and_undo():
    """An unknown node raises without touching the graph; undo restores it."""
    manager = _manager(nx.Graph([(0, 1), (1, 2), (10, 11)]))
    with pytest.raises(ValueError):
        manager.set_origin(99, redraw=False)
    assert len(manager._undo_buffer) == 0

    manager.set_origin(0, redraw=False)
    manager.undo(redraw=False)

    restored = manager._data.skeleton_graph.graph
    assert not restored.is_directed()
    assert restored.number_of_nodes() == 5
