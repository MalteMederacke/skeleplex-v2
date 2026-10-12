"""Train the loop breaker used by review_graphs.py from the curated graphs.

Every graph in graphs_fixed/ that is finished -- origin set, no loops left --
is paired with its raw graph, and the edits that turned one into the other
become the training examples (see ``_loop_breaking.py``). The model is written
to LOOP_BREAKER_MODEL, where "Auto-break loops" in review_graphs.py picks it up.

With two or more curated graphs the script first reports how well a model
trained on the others reproduces each one, so you can see whether it is worth
using. Re-run it whenever more graphs have been curated.

Usage:
    python train_loop_breaker.py
"""

import sys
from pathlib import Path

import joblib
import networkx as nx
import numpy as np

from skeleplex.graph.constants import NODE_COORDINATE_KEY
from skeleplex.graph.skeleton_graph import SkeletonGraph

sys.path.insert(0, str(Path(__file__).parent))
from _constants import LOOP_BREAKER_MODEL  # noqa: E402
from _layout import find_graphs, graph_path, rel  # noqa: E402
from _loop_breaking import (  # noqa: E402
    UNCERTAIN_BELOW,
    _KEPT_FRACTION,
    edge_features,
    fit,
    kept_fraction,
    plan_cuts,
)


def n_loops(graph):
    undirected = graph.to_undirected()
    return (
        undirected.number_of_edges()
        - undirected.number_of_nodes()
        + nx.number_connected_components(undirected)
    )


def origin_in_raw(raw_graph, fixed):
    """The curated graph's origin as a node of the raw graph.

    Usually the same id; if the origin was created during curation (by a
    split), fall back to the raw node closest to it.
    """
    if fixed.origin in raw_graph:
        return fixed.origin
    target = fixed.graph.nodes[fixed.origin][NODE_COORDINATE_KEY]
    nodes = list(raw_graph.nodes)
    coordinates = np.array([raw_graph.nodes[n][NODE_COORDINATE_KEY] for n in nodes])
    return nodes[int(np.argmin(np.linalg.norm(coordinates - target, axis=1)))]


def load_example(ref):
    """(raw graph, origin, EdgeTable, kept fraction) for one curated graph, or None."""
    fixed = SkeletonGraph.from_json_file(str(ref.path))
    if getattr(fixed, "origin", None) is None:
        print("  [skip] origin not set")
        return None
    if n_loops(fixed.graph) > 0:
        print("  [skip] still has loops -- curation not finished")
        return None

    raw = SkeletonGraph.from_json_file(str(graph_path(ref.sample)))
    origin = origin_in_raw(raw.graph, fixed)
    table = edge_features(raw.graph, origin)
    kept = kept_fraction(raw.graph, fixed.graph)
    n_cut = int((kept[table.trainable] <= _KEPT_FRACTION).sum())
    print(f"  {n_loops(raw.graph)} loops, {n_cut} loop edges cut by hand")
    return raw.graph, origin, table, kept


def report_held_out(examples):
    """Train on all graphs but one, and compare the cuts on that one."""
    print("\nHeld-out check (model trained on the other graphs):")
    total = np.zeros(4, dtype=int)
    for i, (name, (graph, origin, table, kept)) in enumerate(examples):
        others = [(e[2], e[3]) for j, (_, e) in enumerate(examples) if j != i]
        cuts = plan_cuts(graph, origin, fit(others), table=table)
        row = {edge: r for r, edge in enumerate(table.ids)}
        same = np.array([kept[row[c.edge]] <= _KEPT_FRACTION for c in cuts])
        sure = np.array([c.confidence >= UNCERTAIN_BELOW for c in cuts])
        counts = np.array([len(cuts), same.sum(), sure.sum(), (same & sure).sum()])
        total += counts
        print(
            f"  {name}: {counts[1]}/{counts[0]} cuts as by hand, "
            f"{counts[3]}/{counts[2]} of the confident ones"
        )
    print(
        f"  overall: {total[1] / max(total[0], 1):.0%} of cuts as by hand, "
        f"{total[3] / max(total[2], 1):.0%} of the confident ones "
        f"(score >= {UNCERTAIN_BELOW}), which are {total[2] / max(total[0], 1):.0%} "
        "of all cuts"
    )


def main():
    refs = [r for r in find_graphs() if r.curated]
    if not refs:
        print("No curated graphs found. Curate some in review_graphs.py first.")
        sys.exit(1)

    examples = []
    for i, ref in enumerate(refs, 1):
        print(f"[{i}/{len(refs)}] {rel(ref.path)}")
        example = load_example(ref)
        if example is not None:
            examples.append((ref.sample.name, example))
    if not examples:
        print("No finished curated graphs to learn from.")
        sys.exit(1)

    if len(examples) >= 2:
        report_held_out(examples)

    model = fit([(e[2], e[3]) for _, e in examples])
    model["trained_on"] = [name for name, _ in examples]
    out_path = Path(LOOP_BREAKER_MODEL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"\nTrained on {len(examples)} graphs -> {out_path}")


if __name__ == "__main__":
    main()
