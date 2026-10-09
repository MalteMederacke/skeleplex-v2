"""Learned loop breaking for skeleton graphs.

A branching tree has no loops, but touching branches fuse in the segmentation
and every fusion closes one in the skeleton graph. Breaking them by hand is most
of the work in review_graphs.py, so this module learns the choice from graphs
that are already curated:

  1. ``edge_features``  describes every edge of the raw, undirected graph
  2. ``kept_fraction``  recovers what was done to it by comparing the raw graph
                        with its curated copy (there is no edit log)
  3. ``fit``            trains two classifiers on the loop edges: is this edge
                        cut, and if so is it deleted whole or split in three
                        with the middle removed
  4. ``plan_cuts``      scores a new graph and keeps the spanning tree of the
                        edges least likely to be cut -- everything left over is
                        cut, which is exactly one edge per loop
  5. ``apply_cuts``     edits the graph in place

Only graph geometry is used -- no segmentation -- so planning takes seconds.
The features depend on the origin node but not on the graph being directed:
directing needs the loops gone first, which is the point of doing this before.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingClassifier

from skeleplex.graph.constants import EDGE_COORDINATES_KEY, EDGE_SPLINE_KEY
from skeleplex.graph.constants import NODE_COORDINATE_KEY
from skeleplex.graph.modify_graph import merge_edge
from skeleplex.graph.spline import B3Spline

FEATURES = (
    "length",           # arc length of the edge
    "tort",             # length / straight-line distance between its ends
    "chord",            # straight-line distance between its ends
    "detour",           # shortest other route between its ends
    "detour_hops",      # ... counted in edges
    "len_over_detour",  # length / detour
    "depth",            # distance of the nearer end from the origin
    "ddepth",           # difference in distance from the origin between ends
    "ddepth_rel",       # ddepth / length; ~0 where two routes from the origin meet
)

# Cuts scored below this are worth a look. On held-out graphs, cuts above 0.7
# matched the manual choice about 80% of the time, cuts below it about 50%.
UNCERTAIN_BELOW = 0.7

# A raw edge counts as untouched when at least this much of it is still there.
_KEPT_FRACTION = 0.97
# ... and as "split in three, middle removed" (rather than deleted) above this.
_STUB_FRACTION = 0.5
# Shorter paths are deleted whole: there is nothing left to keep as stubs.
_MIN_POINTS_FOR_STUBS = 6


@dataclass
class EdgeTable:
    """Per-edge description of one undirected graph. Rows follow ``ids``."""

    ids: list             # (u, v, key) for multigraphs, (u, v) otherwise
    features: np.ndarray  # (n_edges, len(FEATURES)); missing values are -1
    in_cycle: np.ndarray  # edge lies on at least one loop
    self_loop: np.ndarray

    @property
    def trainable(self):
        """Edges the classifiers are trained on and applied to."""
        return self.in_cycle & ~self.self_loop


@dataclass
class Cut:
    """One planned edit."""

    edge: tuple
    confidence: float     # classifier score of the edge that is cut
    leave_stubs: bool     # split in three and remove the middle, vs delete whole
    position: np.ndarray  # midpoint of the edge, for marking it in the viewer


def _edge_ids(graph):
    if graph.is_multigraph():
        return list(graph.edges(keys=True))
    return list(graph.edges())


def edge_features(graph, origin):
    """Describe every edge of an undirected graph.

    Parameters
    ----------
    graph : nx.Graph or nx.MultiGraph
        Raw skeleton graph, loops and all.
    origin : int
        Node the tree is rooted at.

    Returns
    -------
    EdgeTable
    """
    if graph.is_directed():
        raise ValueError("loop breaking needs the undirected graph")

    ids = _edge_ids(graph)
    lengths = np.empty(len(ids))
    chords = np.empty(len(ids))
    for i, edge in enumerate(ids):
        path = graph.edges[edge][EDGE_COORDINATES_KEY]
        lengths[i] = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        chords[i] = np.linalg.norm(path[-1] - path[0])

    # same edges keyed by row, so one can be taken out and put back
    by_row = nx.MultiGraph()
    by_row.add_nodes_from(graph.nodes)
    for i, edge in enumerate(ids):
        by_row.add_edge(edge[0], edge[1], key=i, length=lengths[i])

    bridges = {frozenset(b) for b in nx.bridges(nx.Graph(by_row))}
    self_loop = np.array([e[0] == e[1] for e in ids], dtype=bool)
    in_cycle = np.array(
        [
            e[0] == e[1]
            or by_row.number_of_edges(e[0], e[1]) > 1
            or frozenset(e[:2]) not in bridges
            for e in ids
        ],
        dtype=bool,
    )

    from_origin = nx.single_source_dijkstra_path_length(
        by_row, origin, weight="length"
    )

    features = np.full((len(ids), len(FEATURES)), np.nan)
    for i, edge in enumerate(ids):
        u, v = edge[0], edge[1]
        detour = detour_hops = np.nan
        if in_cycle[i] and not self_loop[i]:
            by_row.remove_edge(u, v, i)
            try:
                detour = nx.dijkstra_path_length(by_row, u, v, weight="length")
                detour_hops = nx.shortest_path_length(by_row, u, v)
            except nx.NetworkXNoPath:
                pass
            by_row.add_edge(u, v, key=i, length=lengths[i])

        depth_u = from_origin.get(u, np.nan)
        depth_v = from_origin.get(v, np.nan)
        ddepth = abs(depth_u - depth_v)
        features[i] = (
            lengths[i],
            lengths[i] / max(chords[i], 1e-6),
            chords[i],
            detour,
            detour_hops,
            lengths[i] / detour,
            min(depth_u, depth_v),
            ddepth,
            ddepth / max(lengths[i], 1.0),
        )

    return EdgeTable(
        ids=ids,
        features=np.nan_to_num(features, nan=-1.0),
        in_cycle=in_cycle,
        self_loop=self_loop,
    )


def kept_fraction(raw_graph, fixed_graph):
    """Fraction of each raw edge's path that survives in the curated graph.

    Curation leaves the path points of untouched edges where they were, so an
    edge is located in the curated graph by position rather than by node ids,
    which change whenever neighbouring edges are merged.

    Returns
    -------
    np.ndarray
        One value per raw edge, in ``_edge_ids`` order: ~1 untouched, ~0.67
        split in three with the middle removed, ~0 deleted.
    """
    fixed_points = np.concatenate(
        [d[EDGE_COORDINATES_KEY] for _, _, d in fixed_graph.edges(data=True)]
    )
    tree = cKDTree(fixed_points)

    paths = [raw_graph.edges[e][EDGE_COORDINATES_KEY] for e in _edge_ids(raw_graph)]
    steps = np.concatenate([np.linalg.norm(np.diff(p, axis=0), axis=1) for p in paths])
    tolerance = 0.5 * np.median(steps)

    kept = np.empty(len(paths))
    for i, path in enumerate(paths):
        distance, _ = tree.query(path)
        kept[i] = (distance <= tolerance).mean()
    return kept


def fit(examples):
    """Train the loop breaker.

    Parameters
    ----------
    examples : list of (EdgeTable, np.ndarray)
        One ``edge_features`` table and its ``kept_fraction`` per curated graph.

    Returns
    -------
    dict
        ``cut``: classifier for "this loop edge gets cut"; ``stubs``: classifier
        for "... by removing its middle third" among the cut ones.
    """
    X = np.concatenate([t.features[t.trainable] for t, _ in examples])
    kept = np.concatenate([k[t.trainable] for t, k in examples])
    was_cut = kept <= _KEPT_FRACTION
    stubs_left = kept >= _STUB_FRACTION

    cut_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
        random_state=0,
    ).fit(X, was_cut)
    stub_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0,
    ).fit(X[was_cut], stubs_left[was_cut])

    return {"features": FEATURES, "cut": cut_model, "stubs": stub_model}


def plan_cuts(graph, origin, model, table=None):
    """Choose one edge per loop to cut, in the component holding ``origin``.

    The classifier scores are turned into cuts with a spanning tree rather than
    a threshold: the tree keeps the edges least likely to be cut while holding
    the component together, so what is left out breaks every loop exactly once.
    Other components are left alone -- directing the graph drops them anyway.

    Parameters
    ----------
    graph : nx.Graph or nx.MultiGraph
        Undirected skeleton graph.
    origin : int
        Node the tree is rooted at.
    model : dict
        As returned by ``fit``.
    table : EdgeTable, optional
        ``edge_features(graph, origin)``, if already computed.

    Returns
    -------
    list of Cut
    """
    if tuple(model["features"]) != FEATURES:
        raise ValueError(
            "loop-breaker model was trained on different features; "
            "re-run train_loop_breaker.py"
        )
    if table is None:
        table = edge_features(graph, origin)

    score = np.zeros(len(table.ids))  # bridges: never cut
    stub_score = np.zeros(len(table.ids))
    rows = table.trainable
    if rows.any():
        score[rows] = model["cut"].predict_proba(table.features[rows])[:, 1]
        stub_score[rows] = model["stubs"].predict_proba(table.features[rows])[:, 1]
    score[table.self_loop] = 2.0  # never part of a tree

    component = nx.node_connected_component(graph, origin)
    candidates = nx.MultiGraph()
    for i, edge in enumerate(table.ids):
        if edge[0] in component:
            candidates.add_edge(edge[0], edge[1], key=i, score=score[i])
    kept = {
        key
        for _, _, key in nx.minimum_spanning_edges(
            candidates, weight="score", keys=True, data=False
        )
    }

    cuts = []
    for _, _, i in candidates.edges(keys=True):
        if i in kept:
            continue
        path = graph.edges[table.ids[i]][EDGE_COORDINATES_KEY]
        cuts.append(
            Cut(
                edge=table.ids[i],
                confidence=float(min(score[i], 1.0)),
                leave_stubs=bool(
                    stub_score[i] >= 0.5 and len(path) >= _MIN_POINTS_FOR_STUBS
                ),
                position=np.asarray(path[len(path) // 2], dtype=float),
            )
        )
    return cuts


def _add_stub(graph, node, new_node, points):
    """Attach ``points`` to ``node`` as a new terminal edge ending in ``new_node``."""
    if len(points) < 4:
        # too short for a spline; approximate as a line, as split_edge does
        points = np.linspace(points[0], points[-1], 5)
    graph.add_node(new_node, **{NODE_COORDINATE_KEY: points[-1]})
    graph.add_edge(
        node,
        new_node,
        **{
            EDGE_COORDINATES_KEY: points,
            EDGE_SPLINE_KEY: B3Spline.from_points(points),
        },
    )


def apply_cuts(skeleton_graph, cuts, origin):
    """Apply planned cuts to ``skeleton_graph`` in place.

    Mirrors the manual tools: a stub cut does what "Split in 3, delete middle"
    does, a whole-edge cut what "delete edge" does, including merging the
    degree-2 nodes it leaves behind. Merging is deferred to the end, because it
    renames neighbouring edges that other planned cuts still refer to.

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        Graph to edit; must still be undirected.
    cuts : list of Cut
        As returned by ``plan_cuts`` for this graph.
    origin : int
        Origin node; never merged away.
    """
    graph = skeleton_graph.graph
    next_node = max(graph.nodes) + 1
    to_merge = set()

    for cut in cuts:
        u, v = cut.edge[0], cut.edge[1]
        path = np.asarray(graph.edges[cut.edge][EDGE_COORDINATES_KEY])
        graph.remove_edge(*cut.edge)

        if not cut.leave_stubs:
            to_merge.update((u, v))
            continue

        # run the path from u to v, whichever way it was stored
        u_position = graph.nodes[u][NODE_COORDINATE_KEY]
        if np.linalg.norm(path[0] - u_position) > np.linalg.norm(path[-1] - u_position):
            path = path[::-1]
        arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        )
        first = int(np.clip(np.searchsorted(arc, arc[-1] / 3), 1, len(path) - 3))
        second = int(
            np.clip(np.searchsorted(arc, 2 * arc[-1] / 3), first + 1, len(path) - 2)
        )
        _add_stub(graph, u, next_node, path[: first + 1])
        _add_stub(graph, v, next_node + 1, path[second:][::-1])
        next_node += 2

    graph.remove_nodes_from(list(nx.isolates(graph)))

    for node in to_merge:
        graph = skeleton_graph.graph  # merge_edge swaps in a new graph object
        if node == origin or node not in graph or graph.degree(node) != 2:
            continue
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 2:
            merge_edge(skeleton_graph, neighbors[0], node, neighbors[1])
