"""Skeleplex-based browser for reviewing and curating skeleton graphs.

Widget (added to the skeleplex viewer):
  Prev / Next            : cycle through every graph found by the layout module
                           (loads the graph plus the segmentation zarr it was
                           built on, so the two are always aligned)
  Origin node ID         : node id to use as origin — accepts a bare int (322)
                           or a single-element set as typed ({322})
  Set origin & direct    : keep only the connected component that contains the
                           origin, call graph.to_directed(origin), and update
                           the viewer in place
  Save to graphs_fixed/  : write the (possibly directed) graph beside its
                           sample, where every later stage picks it up in
                           preference to the raw graph

A graph already curated is reloaded from graphs_fixed/ rather than from the raw
graph, so re-opening the browser shows your previous work.

Usage:
    python review_graphs.py
"""

import functools
import sys
import traceback
from pathlib import Path

import networkx as nx
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import skeleplex
from skeleplex.app import view_skeleton
from skeleplex.app._data import ImageFile, SkeletonGraphFile  # noqa: F401
from skeleplex.graph.skeleton_graph import SkeletonGraph

sys.path.insert(0, str(Path(__file__).parent))
from _constants import DEFAULT_VOXEL_SIZE_UM  # noqa: E402
from _layout import find_graphs, graphs_fixed_dir, rel  # noqa: E402


def seg_array_key(store) -> str:
    """Segmentation array matching the graph coordinate space (cropped if present)."""
    return "segmentation_cropped" if "segmentation_cropped" in store else "segmentation"


def guard(method):
    """Wrap a widget callback so an exception is reported in the status line
    (and printed) instead of escaping into Qt and aborting the viewer."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Qt's clicked signal passes a `checked` bool; the guarded callbacks
        # take no arguments, so drop anything Qt (or an internal call) supplies.
        try:
            return method(self)
        except Exception as e:
            traceback.print_exc()
            try:
                self.status.setText(f"ERROR: {e}")
            except Exception:
                pass
    return wrapper


def parse_origin_node(text):
    """Parse an origin node id from the widget text. Accepts a bare int
    ('322') or a single-element set as typed ('{322}')."""
    s = str(text).strip().strip("{}").strip().rstrip(",").strip()
    return int(s)


def update_graph_in_viewer(graph: SkeletonGraph, app) -> None:
    app.data._skeleton_graph = graph
    app.data._update_node_coordinates()
    app.data._update_edge_coordinates()
    app.data._update_edge_colors()
    app.data.events.data.emit()


def update_segmentation_in_viewer(
    zarr_path: Path, seg_key: str, voxel_size_um: list, app
) -> None:
    seg_array_path = zarr_path / seg_key
    seg_file = ImageFile(
        path=seg_array_path,
        voxel_size_um=tuple(voxel_size_um),
    )
    app.data._update_segmentation_file_load_data(seg_file)
    app.load_main_viewer_segmentation()


class GraphBrowser(QWidget):
    def __init__(self, app, refs: list):
        super().__init__()
        self.app = app
        self.refs = refs
        self.index = 0
        self._graph: SkeletonGraph | None = None

        layout = QVBoxLayout()

        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # Prev / Next
        nav = QHBoxLayout()
        btn_prev = QPushButton("← Prev")
        btn_prev.clicked.connect(self.prev)
        nav.addWidget(btn_prev)
        btn_next = QPushButton("Next →")
        btn_next.clicked.connect(self.next)
        nav.addWidget(btn_next)
        layout.addLayout(nav)

        # Origin input
        layout.addWidget(QLabel("Origin node ID:"))
        self.origin_edit = QLineEdit("0")
        self.origin_edit.setPlaceholderText("e.g. 456 or {456}")
        layout.addWidget(self.origin_edit)

        btn_direct = QPushButton("Set origin & make directed")
        btn_direct.clicked.connect(self.set_origin_and_direct)
        layout.addWidget(btn_direct)

        self.btn_save = QPushButton("Save → graphs_fixed/")
        self.btn_save.clicked.connect(self.save_graph)
        layout.addWidget(self.btn_save)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        layout.addStretch()
        self.setLayout(layout)
        self._load_current()

    @property
    def ref(self):
        return self.refs[self.index]

    def _update_title(self):
        ref = self.ref
        zarr_path = ref.sample.zarr_path
        zarr_ok = "✓ seg" if zarr_path.exists() else "✗ no seg"
        fixed = " [fixed]" if ref.curated else ""
        self.title_label.setText(
            f"[{self.index + 1}/{len(self.refs)}] {zarr_ok}{fixed}\n"
            f"{rel(ref.path)}"
        )

    @guard
    def _load_current(self):
        self.status.setText("")
        graph_path = self.ref.path
        zarr_path = self.ref.sample.zarr_path

        # Load graph
        try:
            self._graph = SkeletonGraph.from_json_file(str(graph_path))
        except Exception as e:
            self.status.setText(f"ERROR loading graph: {e}")
            self._update_title()
            return

        # Update origin edit if graph already has origin stored
        if hasattr(self._graph, "origin") and self._graph.origin is not None:
            self.origin_edit.setText(str(self._graph.origin))

        n_nodes = self._graph.graph.number_of_nodes()
        n_edges = self._graph.graph.number_of_edges()
        self.info_label.setText(f"nodes: {n_nodes}  edges: {n_edges}")

        update_graph_in_viewer(self._graph, self.app)
        self.app.look_at_skeleton()

        # Load matching segmentation (aligned with the graph)
        if zarr_path.exists():
            import zarr as _zarr
            store = _zarr.open(str(zarr_path))
            voxel_size = list(
                store.attrs.get("voxel_size_um", list(DEFAULT_VOXEL_SIZE_UM))
            )
            seg_key = seg_array_key(store)
            try:
                update_segmentation_in_viewer(zarr_path, seg_key, voxel_size, self.app)
            except Exception as e:
                self.status.setText(f"Warning: segmentation not loaded ({e})")

        self._update_title()

    @guard
    def prev(self):
        if self.index > 0:
            self.index -= 1
            self._load_current()

    @guard
    def next(self):
        if self.index < len(self.refs) - 1:
            self.index += 1
            self._load_current()

    @guard
    def set_origin_and_direct(self):
        if self._graph is None:
            self.status.setText("No graph loaded.")
            return
        try:
            origin = parse_origin_node(self.origin_edit.text())
        except ValueError:
            self.status.setText("Invalid origin node ID (use e.g. 322 or {322}).")
            return

        undirected = self._graph.graph.to_undirected()
        if origin not in undirected:
            self.status.setText(f"Node {origin} not in graph.")
            return

        # Keep only the connected component containing origin
        components = list(nx.connected_components(undirected))
        main_comp = next(c for c in components if origin in c)
        if len(main_comp) < undirected.number_of_nodes():
            removed = undirected.number_of_nodes() - len(main_comp)
            self._graph.graph = nx.Graph(undirected.subgraph(main_comp).copy())
            self.status.setText(
                f"Kept component with {len(main_comp)} nodes (removed {removed})."
            )
        else:
            self.status.setText(f"Origin {origin} set.")

        self._graph.to_directed(origin)
        update_graph_in_viewer(self._graph, self.app)

        n_nodes = self._graph.graph.number_of_nodes()
        n_edges = self._graph.graph.number_of_edges()
        self.info_label.setText(
            f"nodes: {n_nodes}  edges: {n_edges}  origin: {origin}  [directed]"
        )

    @guard
    def save_graph(self):
        if self._graph is None:
            self.status.setText("No graph loaded.")
            return
        ref = self.ref
        out_dir = graphs_fixed_dir(ref.sample)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ref.path.name
        self._graph.to_json_file(str(out_path))
        self.status.setText(f"Saved → {rel(out_path)}")


def main():
    refs = find_graphs()
    if not refs:
        print("No *_graph.json files found. Run segmentation_to_graph_batch.py first.")
        sys.exit(1)
    n_curated = sum(1 for r in refs if r.curated)
    print(f"Found {len(refs)} graphs ({n_curated} already curated).")

    # Bootstrap viewer with the first graph (and its segmentation if available)
    first = refs[0]
    first_zarr = first.sample.zarr_path
    seg_path = None
    seg_voxel = tuple(DEFAULT_VOXEL_SIZE_UM)
    if first_zarr.exists():
        import zarr as _zarr
        store = _zarr.open(str(first_zarr))
        voxel_size = list(
            store.attrs.get("voxel_size_um", list(DEFAULT_VOXEL_SIZE_UM))
        )
        seg_path = str(first_zarr / seg_array_key(store))
        seg_voxel = tuple(voxel_size)

    app = view_skeleton(
        graph_path=str(first.path),
        segmentation_path=seg_path,
        segmentation_voxel_size_um=seg_voxel,
    )

    widget = GraphBrowser(app, refs)
    app.add_auxiliary_widget(widget, name="Graph Browser")

    skeleplex.app.run()


if __name__ == "__main__":
    main()
