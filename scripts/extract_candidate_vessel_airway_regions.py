"""Find candidate vessel/airway edge pairs for QC.

Outputs a JSON file consumable by the EdgeColoringNavigatorWidget:
  [ [[u1, v1], [u2, v2]], ... ]
"""

import json
import random
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from skeleplex.graph.constants import EDGE_COORDINATES_KEY, EDGE_SPLINE_KEY
from skeleplex.graph.skeleton_graph import SkeletonGraph

# ── Paths ─────────────────────────────────────────────────────────────────────
GRAPH_PATH = Path(
    "/local1/dluca/lung/20260429_graph_seg/"
    "LADAF-2021-17-left-v9_graph_break_detection_on_scales_curated.json"
)
OUTPUT_PATH = Path("candidate_vessel_airway_pairs.json")

# ── Parameters ────────────────────────────────────────────────────────────────
SUBSET_SIZE = None  # None = use all edges
N_PAIRS = None  # None = no limit
MAX_DIST_UM = 2000  # maximum distance between parallel edges in µm
MAX_ANGLE = 30  # maximum angle of parallel edges in degrees
MIN_LENGTH = 3500  # minimum length of parallel edges in µm
RANDOM_SEED = 42  # for subsampling
# ──────────────────────────────────────────────────────────────────────────────


def _get_edge_data(g, u, v):
    """Return the attribute dict for edge (u, v), handling multigraph keys."""
    return g[u][v][next(iter(g[u][v]))]


def _get_edge_direction(g, u, v):
    """Return the unit direction vector of edge (u, v) from start to end."""
    data = _get_edge_data(g, u, v)
    if EDGE_COORDINATES_KEY in data:
        coords = data[EDGE_COORDINATES_KEY]
        direction = coords[-1] - coords[0]
    elif EDGE_SPLINE_KEY in data:
        pts = np.array(data[EDGE_SPLINE_KEY].eval(np.array([0.01, 0.99])))
        direction = pts[-1] - pts[0]
    else:
        raise KeyError(f"Edge ({u},{v}) has neither path nor spline.")
    return direction / np.linalg.norm(direction)


def _get_edge_midpoint(g, u, v):
    """Return the 3-D midpoint coordinate of edge (u, v)."""
    data = _get_edge_data(g, u, v)
    if EDGE_COORDINATES_KEY in data:
        coords = data[EDGE_COORDINATES_KEY]
        return coords[len(coords) // 2]
    elif EDGE_SPLINE_KEY in data:
        return np.array(data[EDGE_SPLINE_KEY].eval(np.array([0.5])))[0]
    else:
        raise KeyError(f"Edge ({u},{v}) has neither path nor spline.")


def _get_edge_length(g, u, v):
    """Return the arc length of edge (u, v) in the same units as the coordinates."""
    data = _get_edge_data(g, u, v)
    if EDGE_COORDINATES_KEY in data:
        coords = data[EDGE_COORDINATES_KEY]
    elif EDGE_SPLINE_KEY in data:
        coords = np.array(data[EDGE_SPLINE_KEY].eval(np.linspace(0.01, 0.99, 20)))
    else:
        raise KeyError(f"Edge ({u},{v}) has neither path nor spline.")
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))


def _angle_between_edges(g, edge1, edge2):
    """Return the acute angle in degrees between the directions of two edges."""
    d1 = _get_edge_direction(g, *edge1)
    d2 = _get_edge_direction(g, *edge2)
    cos_angle = np.clip(np.dot(d1, d2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return min(angle, 180 - angle)


def find_candidate_pairs(
    g,
    *,
    subset_size=None,
    n_pairs=1000,
    max_dist_um=2000,
    max_angle=30,
    min_length=3500,
    random_seed=42,
):
    """Return edge pairs that are spatially close and nearly parallel.

    Returns
    -------
    list of (edge1, edge2, angle_deg, dist_um)
    """
    all_edges = list(g.edges())
    print(f"Total edges: {len(all_edges)}")

    random.seed(random_seed)
    pool = (
        all_edges
        if subset_size is None
        else random.sample(all_edges, min(subset_size, len(all_edges)))
    )

    midpoints = {}
    for u, v in pool:
        try:
            if _get_edge_length(g, u, v) >= min_length:
                midpoints[(u, v)] = _get_edge_midpoint(g, u, v)
        except Exception:
            pass

    valid_edges = list(midpoints.keys())
    coords_arr = np.array([midpoints[e] for e in valid_edges])
    print(f"Edges passing length filter ({min_length} µm): {len(valid_edges)}")

    tree = KDTree(coords_arr)
    seen = set()
    pairs = []
    for i, edge1 in enumerate(valid_edges):
        dists, nbrs = tree.query(coords_arr[i], k=2)
        if dists[1] >= max_dist_um:
            continue
        edge2 = valid_edges[nbrs[1]]
        key = frozenset([edge1, edge2])
        if key in seen:
            continue
        seen.add(key)
        try:
            angle = _angle_between_edges(g, edge1, edge2)
            if angle <= max_angle:
                pairs.append((edge1, edge2, angle, dists[1]))
        except Exception:
            pass
        if n_pairs is not None and len(pairs) >= n_pairs:
            break

    print(
        f"Found {len(pairs)} pairs within {max_dist_um} µm " f"and angle ≤ {max_angle}°"
    )
    return pairs


def main():
    """Load skeleton graph, find candidate edge pairs, and save results as JSON."""
    print(f"Loading graph from {GRAPH_PATH} ...")
    skeleton = SkeletonGraph.from_json_file(GRAPH_PATH)
    g = skeleton.graph

    pairs = find_candidate_pairs(
        g,
        subset_size=SUBSET_SIZE,
        n_pairs=N_PAIRS,
        max_dist_um=MAX_DIST_UM,
        max_angle=MAX_ANGLE,
        min_length=MIN_LENGTH,
        random_seed=RANDOM_SEED,
    )

    print(f"\n{'Edge 1':<28} {'Edge 2':<28} {'Angle':>8} {'Distance':>12}")
    print("-" * 82)
    for edge1, edge2, angle, dist in pairs:
        print(f"  {edge1!s:<26}   {edge2!s:<26}  {angle:>7.2f}°  {dist:>10.1f} µm")

    output = [[list(edge1), list(edge2)] for edge1, edge2, *_ in pairs]
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f)
    print(f"\nSaved {len(output)} pairs → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
