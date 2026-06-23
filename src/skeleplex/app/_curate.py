import numbers
from collections import deque
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Annotated, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from magicgui import magicgui
from qtpy.QtCore import QByteArray
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from skeleplex.graph.constants import (
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    GENERATION_KEY,
    LENGTH_KEY,
    NODE_COORDINATE_KEY,
)
from skeleplex.graph.modify_graph import (
    connect_without_merging,
    delete_edge,
    merge_nodes,
    split_edge,
)
from skeleplex.graph.utils import draw_line_segment
from skeleplex.visualize import EdgeColormap

if TYPE_CHECKING:
    # prevent circular import
    from skeleplex.app._data import DataManager

import ast

from skan import Skeleton
from skimage.morphology import label as sk_label

from skeleplex.graph.break_detection import find_breaks_in_skeleton


def edge_string_to_key(edge_string: str) -> set[tuple[int, ...]]:
    """Parse a string representation of a set of tuples back into a Python set.

    This function safely converts string representations of sets containing tuples
    back into their original Python data structure. It handles the case where
    the string was created using str() on a set of tuples.

    Parameters
    ----------
    edge_string : str
        String representation of a set of tuples, typically created by
        calling str() on a set object containing tuples.

    Returns
    -------
    set[tuple[int, ...]]
        A set containing tuples parsed from the input string.

    Raises
    ------
    ValueError
        If the string cannot be safely parsed as a set of tuples.
    SyntaxError
        If the string contains invalid Python syntax.
    """
    try:
        # parse the string to convert it back to a set of tuples
        parsed_result = ast.literal_eval(edge_string)

        # Verify that the result is a set
        if not isinstance(parsed_result, set):
            raise ValueError(f"Expected a set, but got {type(parsed_result).__name__}")

        # Verify that all elements are tuples
        for element in parsed_result:
            if not isinstance(element, tuple):
                raise ValueError("Expected all elements must be tuples")

        return parsed_result

    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Could not parse edge string '{edge_string}': {e}") from e


def node_string_to_node_keys(node_string: str) -> set[int]:
    """Parse a string representation of a set of integers back into a Python set.

    This function safely converts string representations of sets containing integers
    back into their original Python data structure. It handles the case where
    the string was created using str() on a set of integers.

    Parameters
    ----------
    node_string : str
        String representation of a set of integers, typically created by
        calling str() on a set object containing integers.

    Returns
    -------
    set[int]
        A set containing integers parsed from the input string.

    Raises
    ------
    ValueError
        If the string cannot be safely parsed as a set of integers,
        or if the string exceeds the maximum allowed length.
    SyntaxError
        If the string contains invalid Python syntax.
    """
    try:
        parsed_result = ast.literal_eval(node_string)

        # Verify that the result is a set
        if not isinstance(parsed_result, set):
            raise ValueError(f"Expected a set, but got {type(parsed_result).__name__}")

        # Verify that all elements are integers
        for element in parsed_result:
            if not isinstance(element, int):
                raise ValueError("All elements must be integers")

        return parsed_result

    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Could not parse node string '{node_string}': {e}") from e


class LIFOBuffer:
    """A last-in-first-out buffer with a maximum size.

    This buffer automatically removes the oldest items when the maximum
    size is exceeded, maintaining LIFO ordering for retrieval.

    Parameters
    ----------
    max_size : int
        The maximum number of items the buffer can hold.
        Must be greater than 0.

    Raises
    ------
    ValueError
        If max_size is not a positive integer.
    """

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be greater than 0")

        self._buffer = deque(maxlen=max_size)
        self._max_size = max_size

    def push(self, item: Any) -> None:
        """Add an item to the buffer.

        If the buffer is at maximum capacity, the oldest item
        will be automatically removed.

        Parameters
        ----------
        item : Any
            The item to add to the buffer.
        """
        self._buffer.append(item)

    def pop(self) -> Any:
        """Remove and return the most recently added item.

        Returns
        -------
        Any
            The most recently added item.

        Raises
        ------
        IndexError
            If the buffer is empty.
        """
        if not self._buffer:
            raise IndexError("pop from empty buffer")
        return self._buffer.pop()

    @property
    def max_size(self) -> int:
        """Get the maximum capacity of the buffer.

        Returns
        -------
        int
            The maximum number of items the buffer can hold.
        """
        return self._max_size

    def clear(self) -> None:
        """Remove all items from the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Return the number of items in the buffer."""
        return len(self._buffer)


class CurationManager:
    def __init__(
        self,
        data_manager: "DataManager",
    ):
        self._data = data_manager

        # buffers for undo and redo operations
        self._undo_buffer = LIFOBuffer(max_size=10)
        self._redo_buffer = LIFOBuffer(max_size=10)

    def delete_edge(
        self,
        edge: Annotated[set[tuple[int, int]] | str, {"widget_type": "LineEdit"}],
        force: bool = False,
        redraw: bool = True,
    ) -> None:
        """Delete an edge from the skeleton graph.

        Parameters
        ----------
        edge : tuple[int, int] | None
            The edge to delete, represented as a tuple of node IDs.
            If None, no action is taken.
        force : bool
            If True, remove the edge without merging degree-2 nodes.
            Default is False.
        redraw : bool
            Flag set to True to redraw the graph after deletion.
            Defaults value is True.
        """
        if len(edge) == 0:
            # if no edge is selected, do nothing
            return

        # parse the edge if it is a string
        edges = edge_string_to_key(edge) if isinstance(edge, str) else edge

        # store the previous state in the undo buffer
        self._undo_buffer.push(deepcopy(self._data.skeleton_graph))

        # delete the edge from the skeleton graph
        for edge in edges:
            delete_edge(
                skeleton_graph=self._data.skeleton_graph, edge=edge, force=force
            )

        if redraw:
            # redraw the graph
            self._update_and_request_redraw()

    def connect_without_merging(
        self,
        start_node: Annotated[int, {"widget_type": "LineEdit"}],
        end_node: Annotated[int, {"widget_type": "LineEdit"}],
        redraw: bool = True,
    ) -> None:
        """Connect two nodes in the skeleton graph without merging them.

        This method connects two nodes in the skeleton graph by creating an edge
        between them. If the nodes are already connected, no action is taken.
        The connection does not merge the nodes, preserving their individual identities.

        Parameters
        ----------
        start_node : int
            The ID of the first node to connect.
        end_node : int
            The ID of the second node to connect.
        redraw : bool
            Flag set to True to redraw the graph after connecting.
            Defaults value is True.
        """
        start_node = node_string_to_node_keys(start_node)
        end_node = node_string_to_node_keys(end_node)
        start_node = next(iter(start_node), None)
        end_node = next(iter(end_node), None)
        if start_node is None or end_node is None:
            # if either node is None, do nothing
            return
        if start_node == end_node:
            # if both nodes are the same, do nothing
            return

        # store the previous state in the undo buffer
        self._undo_buffer.push(deepcopy(self._data.skeleton_graph))

        # connect the nodes without merging
        connect_without_merging(
            skeleton_graph=self._data.skeleton_graph, node1=start_node, node2=end_node
        )

        if redraw:
            # redraw the graph
            self._update_and_request_redraw()

    def connect_with_merging(
        self,
        node_to_keep: Annotated[int, {"widget_type": "LineEdit"}],
        node_to_merge: Annotated[int, {"widget_type": "LineEdit"}],
        redraw: bool = True,
    ) -> None:
        """Connect two nodes in the skeleton graph by merging them.

        This method connects two nodes in the skeleton graph by merging one node
        into another. The node to keep will retain its identity, while the other
        node will be merged into it, effectively removing it from the graph.

        Parameters
        ----------
        node_to_keep : int
            The ID of the nod e to keep after merging.
        node_to_merge : int
            The ID of the node to merge into the first node.
        redraw : bool
            Flag set to True to redraw the graph after connecting.
            Defaults value is True.
        """
        node_to_keep = node_string_to_node_keys(node_to_keep)
        node_to_merge = node_string_to_node_keys(node_to_merge)
        node_to_keep = next(iter(node_to_keep), None)
        node_to_merge = next(iter(node_to_merge), None)

        if node_to_keep == node_to_merge:
            # if both nodes are the same, do nothing
            return

        # store the previous state in the undo buffer
        self._undo_buffer.push(deepcopy(self._data.skeleton_graph))

        # merge the nodes in the skeleton graph
        merge_nodes(
            skeleton_graph=self._data.skeleton_graph,
            node_to_keep=node_to_keep,
            node_to_merge=node_to_merge,
        )

        # connect the nodes with merging
        if redraw:
            # redraw the graph
            self._update_and_request_redraw()

    def undo(self, redraw: bool = True) -> None:
        """Undo the last action performed on the skeleton graph.

        This method restores the skeleton graph to its previous state
        using the undo buffer. If there are no actions to undo, it does nothing.

        Parameters
        ----------
        redraw : bool
            Flag set to True to redraw the graph after undoing.
            Defaults value is True.
        """
        if len(self._undo_buffer) == 0:
            # if there are no actions to undo, do nothing
            return

        # store the current state in the redo buffer
        self._redo_buffer.push(deepcopy(self._data.skeleton_graph))

        # restore the previous state from the undo buffer
        previous_state = self._undo_buffer.pop()
        self._data._skeleton_graph = previous_state

        if redraw:
            # redraw the graph
            self._update_and_request_redraw()

    def redo(self, redraw: bool = True) -> None:
        """Redo the last undone action on the skeleton graph.

        This method restores the skeleton graph to the next state in the undo buffer.
        If there are no actions to redo, it does nothing.

        Parameters
        ----------
        redraw : bool
            Flag set to True to redraw the graph after redoing.
            Defaults value is True.
        """
        if len(self._redo_buffer) == 0:
            # if there are no actions to redo, do nothing
            return

        # store the current state in the undo buffer
        self._undo_buffer.push(deepcopy(self._data.skeleton_graph))

        # restore the next state from the redo buffer
        next_state = self._redo_buffer.pop()
        self._data._skeleton_graph = next_state

        if redraw:
            # redraw the graph
            self._update_and_request_redraw()

    def render_around_node(
        self,
        node_id=int,
        bounding_box_width: int = 100,
        render_segmentation: bool = False,
    ):
        """Render a bounding box around the specified node.

        Parameters
        ----------
        node_id : int
            The ID of the node to render around.
        bounding_box_width : int
            The width of the bounding box to render around the node.
            Default is 100.
        render_segmentation : bool
            Whether to also render the segmentation in the same bounding box.
            Default is False.
        """
        # get the coordinate of the node
        node_id = node_string_to_node_keys(node_id)
        node_id = next(iter(node_id), None)
        graph_object = self._data.skeleton_graph.graph
        node_coordinate = graph_object.nodes[node_id][NODE_COORDINATE_KEY]

        # get the minimum and maximum coordinates for the bounding box
        half_width = bounding_box_width / 2
        min_coordinate = node_coordinate - half_width
        max_coordinate = node_coordinate + half_width

        # set the bounding box in the viewer
        self._data.skeleton_view.bounding_box._min_coordinate = min_coordinate
        self._data.skeleton_view.bounding_box._max_coordinate = max_coordinate

        # set the render mode to bounding box
        self._data.skeleton_view.mode = "bounding_box"

        if render_segmentation and self._data._segmentation is not None:
            self._data.segmentation_view.bounding_box._min_coordinate = min_coordinate
            self._data.segmentation_view.bounding_box._max_coordinate = max_coordinate
            self._data.segmentation_view.mode = "bounding_box"
        else:
            self._data.segmentation_view.mode = "none"

    def _update_and_request_redraw(
        self, clear_edge_selection: bool = True, clear_node_selection: bool = True
    ) -> None:
        """Update the rendered graph data and request a redraw."""
        # Clear the selection if specified
        if clear_edge_selection:
            self._data.selection.edge.values = set()
        if clear_node_selection:
            self._data.selection.node.values = set()

        # Update the skeleton graph data
        self._data._update_edge_coordinates()
        self._data._update_edge_colors()
        self._data._update_node_coordinates()

        # Update the data view
        self._data.skeleton_view.update()


class RenderAroundNodeWidget(QWidget):
    """Widget for rendering around a node and pruning edges in the bounding box."""

    def __init__(self, viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._bbox_min: np.ndarray | None = None
        self._bbox_max: np.ndarray | None = None

        node_label = QLabel("Node ID:")
        self._node_input = QLineEdit()
        self._node_input.setPlaceholderText("e.g. {1}")

        bbox_label = QLabel("Bounding box width:")
        self._bbox_spinbox = QDoubleSpinBox()
        self._bbox_spinbox.setMinimum(0)
        self._bbox_spinbox.setMaximum(1e9)
        self._bbox_spinbox.setValue(100)

        self._segmentation_checkbox = QCheckBox("Render segmentation")

        render_button = QPushButton("Render around node")
        render_button.clicked.connect(self._on_render_clicked)

        threshold_label = QLabel("Min length threshold:")
        self._threshold_spinbox = QDoubleSpinBox()
        self._threshold_spinbox.setMinimum(0)
        self._threshold_spinbox.setMaximum(1e9)
        self._threshold_spinbox.setValue(10.0)

        delete_button = QPushButton("Delete short edges in bbox (force)")
        delete_button.clicked.connect(self._on_delete_clicked)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(node_label)
        layout.addWidget(self._node_input)
        layout.addWidget(bbox_label)
        layout.addWidget(self._bbox_spinbox)
        layout.addWidget(self._segmentation_checkbox)
        layout.addWidget(render_button)
        layout.addWidget(threshold_label)
        layout.addWidget(self._threshold_spinbox)
        layout.addWidget(delete_button)
        layout.addWidget(self._status_label)
        self.setLayout(layout)

        viewer.add_auxiliary_widget(self, name="Render around node")

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    def _on_render_clicked(self) -> None:
        self._set_status("")
        try:
            node_id = node_string_to_node_keys(self._node_input.text())
            node_id = next(iter(node_id), None)
            if node_id is None:
                self._set_status("No node parsed.")
                return
            width = self._bbox_spinbox.value()
            render_seg = self._segmentation_checkbox.isChecked()
            self.viewer.curate.render_around_node(
                node_id=str({node_id}),
                bounding_box_width=width,
                render_segmentation=render_seg,
            )
            half = width / 2
            graph = self.viewer.data.skeleton_graph.graph
            coord = graph.nodes[node_id][NODE_COORDINATE_KEY]
            self._bbox_min = coord - half
            self._bbox_max = coord + half
            self._set_status(f"Rendering around node {node_id}.")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _on_delete_clicked(self) -> None:
        self._set_status("")
        if self._bbox_min is None or self._bbox_max is None:
            self._set_status("Render around a node first.")
            return
        try:
            threshold = self._threshold_spinbox.value()
            graph = self.viewer.data.skeleton_graph.graph
            to_delete = []
            for u, v, data in graph.edges(data=True):
                u_coord = graph.nodes[u][NODE_COORDINATE_KEY]
                v_coord = graph.nodes[v][NODE_COORDINATE_KEY]
                u_in = np.all(u_coord >= self._bbox_min) and np.all(
                    u_coord <= self._bbox_max
                )
                v_in = np.all(v_coord >= self._bbox_min) and np.all(
                    v_coord <= self._bbox_max
                )
                if not (u_in or v_in):
                    continue
                length = data.get(LENGTH_KEY)
                if length is None:
                    coords = data.get(EDGE_COORDINATES_KEY)
                    if coords is not None:
                        length = float(
                            np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
                        )
                if length is not None and length < threshold:
                    to_delete.append((u, v))
            if not to_delete:
                self._set_status("No short edges found in bounding box.")
                return
            self.viewer.curate._undo_buffer.push(
                deepcopy(self.viewer.data.skeleton_graph)
            )
            for edge in to_delete:
                if graph.has_edge(*edge):
                    delete_edge(
                        skeleton_graph=self.viewer.data.skeleton_graph,
                        edge=edge,
                        force=True,
                    )
            self.viewer.curate._update_and_request_redraw()
            self._set_status(f"Deleted {len(to_delete)} short edge(s).")
        except Exception as e:
            self._set_status(f"Error: {e}")


class BreakDetectionWidget(QWidget):
    """Widget for detecting breaks at a selected skeleton edge.

    Loads a segmentation chunk around the edge, rasterizes the local graph
    edges into a binary skeleton image, then runs skan-based break detection
    to find endpoints that can be connected through the segmentation.

    Source endpoints are shown as one point cloud, suggested destinations
    as another. The first run may be slow due to numba JIT compilation.

    Parameters
    ----------
    viewer : SkelePlexApp
        The SkelePlex application instance.
    """

    def __init__(self, viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._source_visual = None
        self._source_store = None
        self._dest_visual = None
        self._dest_store = None

        edge_label = QLabel("Edge key:")
        self._edge_input = QLineEdit()
        self._edge_input.setPlaceholderText("e.g. {(1, 2)}")

        margin_label = QLabel("BBox margin (µm):")
        self._margin_spinbox = QDoubleSpinBox()
        self._margin_spinbox.setMinimum(0)
        self._margin_spinbox.setMaximum(1e9)
        self._margin_spinbox.setValue(200.0)

        radius_label = QLabel("Search radius (voxels):")
        self._radius_spinbox = QDoubleSpinBox()
        self._radius_spinbox.setMinimum(1)
        self._radius_spinbox.setMaximum(1e6)
        self._radius_spinbox.setValue(50.0)

        self._use_angles_checkbox = QCheckBox("Use angle filtering")
        self._use_angles_checkbox.setChecked(True)

        angle_label = QLabel("Angle threshold (°):")
        self._angle_spinbox = QDoubleSpinBox()
        self._angle_spinbox.setMinimum(0)
        self._angle_spinbox.setMaximum(180)
        self._angle_spinbox.setValue(90.0)

        run_button = QPushButton("Run break detection")
        run_button.clicked.connect(self._on_run_clicked)

        clear_button = QPushButton("Clear results")
        clear_button.clicked.connect(self._on_clear_clicked)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(edge_label)
        layout.addWidget(self._edge_input)
        layout.addWidget(margin_label)
        layout.addWidget(self._margin_spinbox)
        layout.addWidget(radius_label)
        layout.addWidget(self._radius_spinbox)
        layout.addWidget(self._use_angles_checkbox)
        layout.addWidget(angle_label)
        layout.addWidget(self._angle_spinbox)
        layout.addWidget(run_button)
        layout.addWidget(clear_button)
        layout.addWidget(self._status_label)
        self.setLayout(layout)

        viewer.add_auxiliary_widget(self, name="Break Detection")

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    def _on_run_clicked(self) -> None:
        self._set_status("Running… (first run may be slow due to JIT compilation)")
        try:
            # --- 1. Parse edge ---
            edges = edge_string_to_key(self._edge_input.text())
            if not edges:
                self._set_status("No edge parsed.")
                return
            u, v = next(iter(edges))[:2]

            graph = self.viewer.data.skeleton_graph.graph
            if not graph.has_edge(u, v):
                self._set_status(f"Edge ({u}, {v}) not in graph.")
                return

            edata = graph[u][v]
            if graph.is_multigraph():
                edata = edata[min(edata.keys())]
            edge_coords = edata[EDGE_COORDINATES_KEY]  # (N, 3) in (z,y,x) µm

            # --- 2. Bounding box around the edge ---
            margin = self._margin_spinbox.value()
            bbox_min_world = edge_coords.min(axis=0) - margin
            bbox_max_world = edge_coords.max(axis=0) + margin

            # --- 3. Load segmentation chunk ---
            segmentation = self.viewer.data.segmentation
            if segmentation is None:
                self._set_status("No segmentation loaded.")
                return

            scale = np.array(self.viewer.data.segmentation_scale)  # (z, y, x) µm/vx
            seg_shape = np.array(segmentation.shape)

            bbox_min_vx = np.maximum(np.round(bbox_min_world / scale), 0).astype(
                np.int64
            )
            bbox_max_vx = np.minimum(
                np.round(bbox_max_world / scale), seg_shape
            ).astype(np.int64)

            if np.any(bbox_min_vx >= bbox_max_vx):
                self._set_status("Bounding box is empty after clamping.")
                return

            self._set_status("Loading segmentation chunk…")
            seg_chunk = np.asarray(
                segmentation[
                    bbox_min_vx[0] : bbox_max_vx[0],
                    bbox_min_vx[1] : bbox_max_vx[1],
                    bbox_min_vx[2] : bbox_max_vx[2],
                ]
            )
            chunk_shape = np.array(seg_chunk.shape)

            # --- 4. Rasterize all graph edges in the chunk ---
            self._set_status("Rasterizing skeleton…")
            skeleton_image = np.zeros(chunk_shape, dtype=np.uint8)

            if graph.is_multigraph():
                edge_iter = [
                    attrs for _, _, attrs in graph.edges(data=True, keys=False)
                ]
            else:
                edge_iter = [attrs for _, _, attrs in graph.edges(data=True)]

            for attrs in edge_iter:
                pts = attrs.get(EDGE_COORDINATES_KEY)
                if pts is None:
                    continue
                local_vx = np.round(pts / scale).astype(np.int64) - bbox_min_vx
                # skip edges entirely outside the chunk
                in_chunk = np.all(local_vx >= 0, axis=1) & np.all(
                    local_vx < chunk_shape, axis=1
                )
                if not np.any(in_chunk):
                    continue
                for i in range(len(local_vx) - 1):
                    draw_line_segment(
                        local_vx[i].astype(float),
                        local_vx[i + 1].astype(float),
                        skeleton_image,
                    )

            if skeleton_image.sum() == 0:
                self._set_status("No skeleton voxels found in region.")
                return

            # --- 5. Build skan Skeleton and labeled skeleton image ---
            self._set_status("Building skeleton graph…")

            skeleton_label_image = sk_label(skeleton_image)
            skeleton_obj = Skeleton(skeleton_image.astype(bool))

            # --- 6. Run break detection ---
            self._set_status("Detecting breaks…")
            node_ids, source_vx, dest_vx = find_breaks_in_skeleton(
                skeleton_obj=skeleton_obj,
                end_point_radius=self._radius_spinbox.value(),
                segmentation_label_image=seg_chunk,
                skeleton_label_image=skeleton_label_image,
                include_angles=self._use_angles_checkbox.isChecked(),
                angle_threshold=self._angle_spinbox.value(),
            )

            if len(node_ids) == 0:
                self._set_status("No breaks detected in this region.")
                return

            # --- 7. Convert skan local voxel coords → viewer display coords ---
            # skan coords: local chunk indices (z, y, x)
            # world: (z, y, x) µm = (local_vx + bbox_min_vx) * scale
            # raw zyx; cellier reverses to (x, y, z) internally
            source_world = (source_vx + bbox_min_vx) * scale
            dest_world = (dest_vx + bbox_min_vx) * scale
            display_source = source_world.astype(np.float32)
            display_dest = dest_world.astype(np.float32)

            # --- 8. Show in viewer ---
            point_size = max(np.max(self.viewer.data.node_coordinates) * 0.01, 50)
            if self._source_visual is None:
                self._source_visual, self._source_store = self.viewer.add_points(
                    point_size=point_size
                )
            if self._dest_visual is None:
                self._dest_visual, self._dest_store = self.viewer.add_points(
                    point_size=point_size
                )

            self._source_store.positions = display_source
            self._dest_store.positions = display_dest
            self._source_visual.appearance.visible = True
            self._dest_visual.appearance.visible = True

            self._set_status(
                f"Found {len(node_ids)} potential break(s). "
                "First point cloud = endpoints, second = suggested targets."
            )

        except Exception as e:
            import traceback

            self._set_status(f"Error: {e}\n{traceback.format_exc()[:300]}")

    def _on_clear_clicked(self) -> None:
        if self._source_visual is not None:
            self._source_visual.appearance.visible = False
        if self._dest_visual is not None:
            self._dest_visual.appearance.visible = False
        self._set_status("")


class ConnectedComponentsWidget(QWidget):
    """Widget for exploring connected components of the skeleton graph.

    Colors the largest component blue, a selected smaller component orange,
    and makes everything else transparent. Buttons cycle through the top-N
    smaller components and allow resetting or recomputing.

    Parameters
    ----------
    viewer : SkelePlexApp
        The SkelePlex application instance.
    """

    _MAIN_COLOR = np.array([0.0, 0.4, 1.0, 1.0], dtype=np.float32)
    _OTHER_COLOR = np.array([1.0, 0.5, 0.0, 1.0], dtype=np.float32)
    _HIDDEN_COLOR = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _DEFAULT_COLOR = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    def __init__(self, viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._components: list[frozenset[int]] = []
        self._current_idx: int = 0  # index into self._components[1:]

        n_label = QLabel("Max other components to track:")
        self._n_spinbox = QSpinBox()
        self._n_spinbox.setMinimum(1)
        self._n_spinbox.setMaximum(100)
        self._n_spinbox.setValue(10)

        self._recompute_button = QPushButton("Recompute components")
        self._recompute_button.clicked.connect(self._on_recompute_clicked)

        self._status_label = QLabel("Not computed yet.")
        self._status_label.setWordWrap(True)

        nav_layout = QHBoxLayout()
        self._prev_button = QPushButton("← Previous")
        self._prev_button.clicked.connect(self._on_prev_clicked)
        self._next_button = QPushButton("Next →")
        self._next_button.clicked.connect(self._on_next_clicked)
        nav_layout.addWidget(self._prev_button)
        nav_layout.addWidget(self._next_button)

        self._reset_button = QPushButton("Reset colors")
        self._reset_button.clicked.connect(self._on_reset_clicked)

        layout = QVBoxLayout()
        layout.addWidget(n_label)
        layout.addWidget(self._n_spinbox)
        layout.addWidget(self._recompute_button)
        layout.addWidget(self._status_label)
        layout.addLayout(nav_layout)
        layout.addWidget(self._reset_button)
        self.setLayout(layout)

        viewer.add_auxiliary_widget(self, name="Connected Components")

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    def _on_recompute_clicked(self) -> None:
        self._set_status("Computing...")
        try:
            graph = self.viewer.data.skeleton_graph.graph
            ug = graph.to_undirected() if graph.is_directed() else graph
            n_max = self._n_spinbox.value()
            all_components = sorted(nx.connected_components(ug), key=len, reverse=True)
            # keep largest + top-N others
            self._components = [frozenset(c) for c in all_components[: n_max + 1]]
            self._current_idx = 0
            n_others = len(self._components) - 1
            self._set_status(
                f"Found {len(all_components)} components. "
                f"Largest: {len(self._components[0])} nodes. "
                f"Tracking {n_others} smaller ones."
            )
            self._apply_colors()
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _on_prev_clicked(self) -> None:
        if len(self._components) < 2:
            self._set_status("Recompute first.")
            return
        n_others = len(self._components) - 1
        self._current_idx = (self._current_idx - 1) % n_others
        self._apply_colors()

    def _on_next_clicked(self) -> None:
        if len(self._components) < 2:
            self._set_status("Recompute first.")
            return
        n_others = len(self._components) - 1
        self._current_idx = (self._current_idx + 1) % n_others
        self._apply_colors()

    def _apply_colors(self) -> None:
        if not self._components:
            return
        try:
            graph = self.viewer.data.skeleton_graph.graph
            main_nodes = self._components[0]
            other_idx = self._current_idx + 1
            other_nodes = (
                self._components[other_idx]
                if other_idx < len(self._components)
                else frozenset()
            )

            color_dict: dict[tuple[int, int], np.ndarray] = {}
            for u, v in graph.edges():
                if u in main_nodes and v in main_nodes:
                    color_dict[(u, v)] = self._MAIN_COLOR.copy()
                elif u in other_nodes and v in other_nodes:
                    color_dict[(u, v)] = self._OTHER_COLOR.copy()
                else:
                    color_dict[(u, v)] = self._HIDDEN_COLOR.copy()

            self.viewer.data.edge_colormap = EdgeColormap.from_arrays(
                colormap=color_dict,
                default_color=self._HIDDEN_COLOR,
            )
            n_others = len(self._components) - 1
            self._set_status(
                f"Component {self._current_idx + 1}/{n_others} "
                f"({len(other_nodes)} nodes). "
                f"Blue = main ({len(main_nodes)} nodes)."
            )
        except Exception as e:
            self._set_status(f"Color error: {e}")

    def _on_reset_clicked(self) -> None:
        try:
            self.viewer.data.edge_colormap = EdgeColormap.from_arrays(
                colormap={},
                default_color=self._DEFAULT_COLOR,
            )
            self._set_status("")
        except Exception as e:
            self._set_status(f"Reset error: {e}")


def make_split_edge_widget(viewer):
    """Create a widget for splitting edges in the skeleton graph."""

    @magicgui(
        edge_to_split_ID={"widget_type": "LineEdit"},
        split_pos={
            "widget_type": "FloatSlider",
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "value": 0.5,
        },
    )
    def split_edge_widget(edge_to_split_ID: int, split_pos: float = 0.5):
        """Widget to split an edge in the skeleton graph.

        Parameters
        ----------
        edge_to_split_ID : str
            The ID of the edge to split, represented as a string.
            This should be a string representation of a set of tuples,
            e.g. "{(1, 2), (2, 3)}".
        split_pos : float
            The position to split the edge at, between 0 and 1.
            Default value is 0.5, which means the edge will be split in the middle
            of its length.
        """
        edge_key = next(iter(edge_string_to_key(edge_to_split_ID)))
        viewer.curate._undo_buffer.push(deepcopy(viewer.curate._data.skeleton_graph))
        split_edge(viewer.curate._data.skeleton_graph, edge_key, split_pos)
        viewer.curate._update_and_request_redraw()

    def preview_split():
        """Preview the split edge operation.

        This function is connected to the split_pos widget to update the preview
        of the split edge in the viewer.
        It calculates the position of the split point based on the current
        split_pos value and the spline of the edge to be split.
        The calculated point position is then set in the point_store and made visible.
        """
        edge_key = next(
            iter(edge_string_to_key(split_edge_widget.edge_to_split_ID.value))
        )
        spline = viewer.data.skeleton_graph.graph.edges[edge_key][EDGE_SPLINE_KEY]
        point_pos = spline.eval(split_edge_widget.split_pos.value)

        split_edge_widget.point_store.positions = np.array(
            [point_pos], dtype=np.float32
        )
        split_edge_widget.point_visual.appearance.visible = True
        viewer._viewer._backend.reslice_all()

    split_edge_widget.split_pos.changed.connect(preview_split)
    point_size = np.max((np.max(viewer.data.node_coordinates) * 0.01, 50))
    split_edge_widget.point_visual, split_edge_widget.point_store = viewer.add_points(
        point_size=point_size
    )
    split_edge_widget.point_visual.appearance.visible = False

    return split_edge_widget


class ChangeBranchColorWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer

        # Create magicgui widget
        self.widget = magicgui(
            self.change_branch_color,
            edge_attribute={
                "choices": self.get_edge_attributes(),
                "widget_type": "ComboBox",
            },
            cmap={"widget_type": "ComboBox", "choices": plt.colormaps()},
            vmin={
                "widget_type": "FloatSlider",
                "min": 0,
                "max": 1,
                "step": 1,
                "value": 0,
            },
            vmax={
                "widget_type": "FloatSlider",
                "min": 0,
                "max": 1,
                "step": 1,
                "value": 1,
            },
        )

        self.widget.call_button.visible = False

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run_clicked)

        self.filter_button = QPushButton("Filter Edges")
        self.filter_button.clicked.connect(self._on_filter_clicked)

        # QLabel for colorbar
        self.colorbar_label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.widget.native)
        layout.addWidget(self.run_button)
        layout.addWidget(self.filter_button)
        layout.addWidget(self.colorbar_label)
        self.setLayout(layout)

        self.widget.edge_attribute.changed.connect(self._on_attribute_change)
        self._on_attribute_change(self.widget.edge_attribute.value)

        self.widget.vmin.changed.connect(self._on_slider_value_change)
        self.widget.vmax.changed.connect(self._on_slider_value_change)

        self.viewer.add_auxiliary_widget(self, name="Change Branch Color", area="left")

    def change_branch_color(
        self, edge_attribute: str, cmap: str, vmin: float, vmax: float
    ):
        """Change the color of edges based on a specific edge attribute."""
        change_color_attr(
            self.viewer,
            edge_attribute=edge_attribute,
            cmap=plt.get_cmap(cmap),
            vmin=vmin,
            vmax=vmax,
        )

    def filter_edges(self):
        """Filter edges based on the selected edge attribute and its value range."""
        edge_attribute = self.widget.edge_attribute.value
        cmap = plt.get_cmap(self.widget.cmap.value)
        vmin = self.widget.vmin.value
        vmax = self.widget.vmax.value

        filter_edges_by_attribute(
            self.viewer,
            edge_attribute=edge_attribute,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    def get_edge_attributes(self):
        """Get the edge attributes from the skeleton graph."""
        if not self.viewer.data.skeleton_graph.graph.edges:
            return []
        attribute_set = set()
        for _, _, edge_data in self.viewer.data.skeleton_graph.graph.edges(data=True):
            attribute_set.update(edge_data.keys())
        return list(attribute_set)

    def get_min_max_values(self, edge_attribute: str):
        values = [
            value
            for _, value in nx.get_edge_attributes(
                self.viewer.data.skeleton_graph.graph, edge_attribute
            ).items()
            if isinstance(value, numbers.Number) and not np.isnan(value)
        ]
        if not values:
            return 0, 0
        return min(values), max(values)

    def _on_run_clicked(self):
        """Apply coloring."""
        self.change_branch_color(
            self.widget.edge_attribute.value,
            self.widget.cmap.value,
            self.widget.vmin.value,
            self.widget.vmax.value,
        )

    def _on_filter_clicked(self):
        """Run a filter function on edges."""
        self.filter_edges()

    def _update_vmin_vmax(self, edge_attribute: str):
        """Update the vmin and vmax sliders based on the edge attribute."""
        vmin, vmax = self.get_min_max_values(edge_attribute)
        self.widget.vmin.min = vmin - 1
        self.widget.vmin.max = vmax + 1
        self.widget.vmax.min = vmin - 1
        self.widget.vmax.max = vmax + 1
        self.widget.vmin.value = vmin
        self.widget.vmax.value = vmax

    def _on_attribute_change(self, value):
        """Callback when edge attribute is changed."""
        self._update_vmin_vmax(value)
        self._update_colorbar(value, self.widget.vmin.value, self.widget.vmax.value)

    def _on_slider_value_change(self):
        """Callback when vmin or vmax slider values are changed."""
        edge_attr = self.widget.edge_attribute.value
        vmin = self.widget.vmin.value
        vmax = self.widget.vmax.value
        self._update_colorbar(edge_attr, vmin, vmax)

    def _update_colorbar(self, attribute_name, vmin, vmax):
        """Update the colorbar based on the selected edge attribute and its value range.

        This method generates a colorbar image and updates the QLabel to display it.

        Parameters
        ----------
        attribute_name : str
            The name of the edge attribute to use for the colorbar.
        vmin : float
            The minimum value for the colorbar normalization.
        vmax : float
            The maximum value for the colorbar normalization.
        """
        cmap = plt.get_cmap(self.widget.cmap.value)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(4, 0.4))
        fig.subplots_adjust(bottom=0.5)

        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation="horizontal",
        )
        cbar.ax.set_xlabel(f"{attribute_name} (Min: {vmin:.2f}, Max: {vmax:.2f})")

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        pixmap = QPixmap()
        pixmap.loadFromData(QByteArray(buf.getvalue()))
        self.colorbar_label.setPixmap(pixmap)


def get_reachable_edges(
    graph: nx.Graph, edge: tuple[int, ...], n_steps: int | None = None
) -> set[tuple[int, int]]:
    """Return all edges reachable from the given edge.

    For directed graphs (DiGraph, MultiDiGraph), returns all edges reachable
    downstream by following outgoing edges from the edge's target node.

    For undirected graphs (Graph, MultiGraph), returns all edges in the
    connected component containing the edge.

    Parameters
    ----------
    graph : nx.Graph
        The networkx graph to search.
    edge : tuple[int, ...]
        The reference edge as a (u, v) or (u, v, key) tuple.
    n_steps : int | None
        Maximum number of edge-hops from the input edge to include.
        None means no limit (entire reachable subgraph).

    Returns
    -------
    set[tuple[int, int]]
        Set of (u, v) edge keys for all reachable edges, including the input edge.
    """
    u, v = edge[0], edge[1]

    if isinstance(graph, nx.DiGraph):
        reachable: set[tuple[int, int]] = {(u, v)}
        for a, b in nx.bfs_edges(graph, v, depth_limit=n_steps):
            reachable.add((a, b))
            reachable.add((b, a))
    else:
        if n_steps is None:
            component = nx.node_connected_component(graph, u)
            nearby_nodes = component
        else:
            nodes_u = {n for n, _ in nx.bfs_edges(graph, u, depth_limit=n_steps)}
            nodes_u.add(u)
            nodes_v = {n for n, _ in nx.bfs_edges(graph, v, depth_limit=n_steps)}
            nodes_v.add(v)
            nearby_nodes = nodes_u | nodes_v
        subgraph = graph.subgraph(nearby_nodes)
        reachable = set(subgraph.edges()) | {(b, a) for a, b in subgraph.edges()}

    return reachable


def _get_terminal_edges(
    graph: nx.Graph, reachable: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    """Return the subset of reachable edges that have no outgoing reachable edges.

    For directed graphs an edge (u, v) is terminal when v has no successors
    that are reached via a reachable edge.  For undirected graphs an edge is
    terminal when one of its endpoints connects only to nodes reached via the
    same edge (i.e. it is a leaf in the reachable subgraph).
    """
    reachable_nodes = {u for u, v in reachable} | {v for u, v in reachable}
    terminal: set[tuple[int, int]] = set()

    if isinstance(graph, nx.DiGraph):
        for u, v in reachable:
            successors_in_reachable = any(
                (v, w) in reachable for w in graph.successors(v)
            )
            if not successors_in_reachable:
                terminal.add((u, v))
    else:
        reachable_subgraph = graph.subgraph(reachable_nodes)
        for u, v in reachable:
            if reachable_subgraph.degree(u) == 1 or reachable_subgraph.degree(v) == 1:
                terminal.add((u, v))
                terminal.add((v, u))

    return terminal


class RenderReachableEdgesWidget(QWidget):
    """Widget that highlights edges reachable from a given input edge.

    For directed graphs, highlights all downstream edges following outgoing
    directions from the edge's target node. For undirected graphs, highlights
    all edges in the same connected component.

    The input edge is shown in red, terminal reachable edges (no further
    branches) in orange, other reachable edges in blue, and all other edges
    are made transparent.

    Parameters
    ----------
    viewer : SkelePlexApp
        The SkelePlex application instance.
    """

    _INPUT_COLOR = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    _REACHABLE_COLOR = np.array([0.3, 0.3, 0.8, 1.0], dtype=np.float32)
    _TERMINAL_COLOR = np.array([1.0, 0.65, 0.0, 1.0], dtype=np.float32)
    _OTHER_COLOR = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _DEFAULT_BLUE = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    def __init__(self, viewer) -> None:
        super().__init__()
        self.viewer = viewer

        label = QLabel("Edge key:")
        self._edge_input = QLineEdit()
        self._edge_input.setPlaceholderText("e.g. {(1, 2)}")

        steps_label = QLabel("Max steps (0 = all):")
        self._steps_spinbox = QSpinBox()
        self._steps_spinbox.setMinimum(0)
        self._steps_spinbox.setMaximum(9999)
        self._steps_spinbox.setValue(0)

        run_button = QPushButton("Render reachable edges")
        run_button.clicked.connect(self._on_run_clicked)

        reset_button = QPushButton("Reset colors")
        reset_button.clicked.connect(self._on_reset_clicked)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self._edge_input)
        layout.addWidget(steps_label)
        layout.addWidget(self._steps_spinbox)
        layout.addWidget(run_button)
        layout.addWidget(reset_button)
        layout.addWidget(self._status_label)
        self.setLayout(layout)

        viewer.add_auxiliary_widget(self, name="Render Reachable Edges")

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    def _on_run_clicked(self) -> None:
        """Parse the input edge, find reachable edges, and update edge colors."""
        self._set_status("")
        try:
            edges = edge_string_to_key(self._edge_input.text())
        except ValueError as e:
            self._set_status(f"Parse error: {e}")
            return
        if not edges:
            self._set_status("No edges parsed.")
            return

        try:
            edge = next(iter(edges))
            skeleton_graph = self.viewer.data.skeleton_graph
            if isinstance(skeleton_graph.graph, nx.DiGraph):
                skeleton_graph.to_directed(skeleton_graph.origin)
            graph = skeleton_graph.graph
            n_steps = self._steps_spinbox.value() or None
            reachable = get_reachable_edges(graph, edge, n_steps=n_steps)
            input_key = (edge[0], edge[1])

            terminal = _get_terminal_edges(graph, reachable)

            color_dict: dict[tuple[int, int], np.ndarray] = {}
            for graph_edge in graph.edges():
                key = (graph_edge[0], graph_edge[1])
                if key == input_key:
                    color_dict[key] = self._INPUT_COLOR.copy()
                elif key in terminal:
                    color_dict[key] = self._TERMINAL_COLOR.copy()
                elif key in reachable:
                    color_dict[key] = self._REACHABLE_COLOR.copy()
                else:
                    color_dict[key] = self._OTHER_COLOR.copy()

            self.viewer.data.edge_colormap = EdgeColormap.from_arrays(
                colormap=color_dict,
                default_color=self._OTHER_COLOR,
            )
            self._set_status(
                f"Highlighted {len(reachable)} reachable edges "
                f"({len(terminal)} terminal, input: {input_key})."
            )
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _on_reset_clicked(self) -> None:
        """Restore the default uniform blue colormap."""
        try:
            self.viewer.data.edge_colormap = EdgeColormap.from_arrays(
                colormap={},
                default_color=self._DEFAULT_BLUE,
            )
            self._set_status("")
        except Exception as e:
            self._set_status(f"Reset error: {e}")


class HighLevelPathWidget(QWidget):
    """Widget that finds and highlights the path to the highest-generation edge.

    Useful for detecting unintended fusions with vasculature, which tend to
    create spuriously deep branches with high generation numbers (>30).

    A "Recompute" button re-runs graph_to_directed and compute_level so the
    generation attribute stays current after edits. "Find deepest path" jumps
    to rank #1; "Next →" steps down to the 2nd, 3rd, … deepest edge in turn.
    The path from origin is shown in blue; the shortest edge on it (the likely
    fusion point) is shown in red.

    Parameters
    ----------
    viewer : SkelePlexApp
        The SkelePlex application instance.
    """

    _PATH_COLOR = np.array([0.3, 0.3, 0.8, 1.0], dtype=np.float32)
    _SHORTEST_COLOR = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    _OTHER_COLOR = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _DEFAULT_BLUE = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    def __init__(self, viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self._sorted_edges: list[tuple[int, int]] = []
        self._current_idx: int = 0

        origin_label = QLabel("Origin node:")
        self._origin_input = QLineEdit()
        self._origin_input.setPlaceholderText("e.g. {0}")

        recompute_button = QPushButton("Recompute directed graph + level")
        recompute_button.clicked.connect(self._on_recompute_clicked)

        highlight_button = QPushButton("Find deepest path")
        highlight_button.clicked.connect(self._on_highlight_clicked)

        next_button = QPushButton("Next →")
        next_button.clicked.connect(self._on_next_clicked)

        reset_button = QPushButton("Reset colors")
        reset_button.clicked.connect(self._on_reset_clicked)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(origin_label)
        layout.addWidget(self._origin_input)
        layout.addWidget(recompute_button)
        layout.addWidget(highlight_button)
        layout.addWidget(next_button)
        layout.addWidget(reset_button)
        layout.addWidget(self._status_label)
        self.setLayout(layout)

        viewer.add_auxiliary_widget(self, name="Deep Path Finder")

    def _set_status(self, msg: str) -> None:
        self._status_label.setText(msg)

    def _get_origin(self) -> int | None:
        text = self._origin_input.text().strip()
        if not text:
            return self.viewer.data.skeleton_graph.origin
        try:
            node_set = node_string_to_node_keys(text)
            return next(iter(node_set), None)
        except ValueError:
            return None

    def _on_recompute_clicked(self) -> None:
        """Recompute the directed graph and generation levels."""
        from skeleplex.measurements.graph_properties import compute_level

        try:
            origin = self._get_origin()
            if origin is None:
                self._set_status("Could not parse origin node.")
                return

            skeleton_graph = self.viewer.data.skeleton_graph
            self._set_status("Computing directed graph…")
            skeleton_graph.to_directed(origin)

            self._set_status("Computing generation levels…")
            skeleton_graph.graph = compute_level(skeleton_graph.graph, origin)

            gen_vals = nx.get_edge_attributes(skeleton_graph.graph, GENERATION_KEY)
            finite_vals = [v for v in gen_vals.values() if not np.isnan(v)]
            max_gen = int(max(finite_vals)) if finite_vals else 0
            self._sorted_edges = []
            self._current_idx = 0
            self._set_status(f"Done. Origin: {origin}, max generation: {max_gen}.")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _build_sorted_edges(self) -> bool:
        """Populate _sorted_edges sorted by descending generation. Returns False on error."""
        graph = self.viewer.data.skeleton_graph.graph
        gen_dict = nx.get_edge_attributes(graph, GENERATION_KEY)
        if not gen_dict:
            self._set_status("No generation attribute. Use 'Recompute' first.")
            return False
        self._sorted_edges = sorted(
            gen_dict,
            key=lambda e: gen_dict[e] if not np.isnan(gen_dict[e]) else -1,
            reverse=True,
        )
        return True

    def _highlight_at_index(self) -> None:
        """Highlight the path to _sorted_edges[_current_idx]."""
        try:
            skeleton_graph = self.viewer.data.skeleton_graph
            graph = skeleton_graph.graph
            origin = skeleton_graph.origin

            if origin is None:
                self._set_status("No origin set. Use 'Recompute' first.")
                return

            if not self._sorted_edges:
                self._set_status("Run 'Find deepest path' first.")
                return

            max_edge = self._sorted_edges[self._current_idx]
            gen_dict = nx.get_edge_attributes(graph, GENERATION_KEY)
            max_gen = gen_dict.get(max_edge, float("nan"))

            u, v = max_edge
            try:
                node_path = nx.shortest_path(graph, origin, u)
            except nx.NetworkXNoPath:
                self._set_status(
                    f"No path from origin {origin} to edge {max_edge}."
                )
                return

            path_edges: set[tuple[int, int]] = set()
            for i in range(len(node_path) - 1):
                path_edges.add((node_path[i], node_path[i + 1]))
            path_edges.add((u, v))

            length_dict = nx.get_edge_attributes(graph, LENGTH_KEY)
            path_lengths = {e: length_dict.get(e, float("inf")) for e in path_edges}
            shortest_edge = min(path_lengths, key=path_lengths.get)

            color_dict: dict[tuple[int, int], np.ndarray] = {}
            for graph_edge in graph.edges():
                key = (graph_edge[0], graph_edge[1])
                if key == shortest_edge:
                    color_dict[key] = self._SHORTEST_COLOR.copy()
                elif key in path_edges:
                    color_dict[key] = self._PATH_COLOR.copy()
                else:
                    color_dict[key] = self._OTHER_COLOR.copy()

            self.viewer.data.edge_colormap = EdgeColormap.from_arrays(
                colormap=color_dict,
                default_color=self._OTHER_COLOR,
            )
            self._set_status(
                f"Rank {self._current_idx + 1}/{len(self._sorted_edges)} — "
                f"generation: {int(max_gen)} at edge {max_edge}. "
                f"Path: {len(path_edges)} edge(s). "
                f"Shortest on path: {shortest_edge} "
                f"(length: {path_lengths[shortest_edge]:.1f})."
            )
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _on_highlight_clicked(self) -> None:
        """Build the sorted edge list and show the deepest path (rank 1)."""
        if not self._build_sorted_edges():
            return
        self._current_idx = 0
        self._highlight_at_index()

    def _on_next_clicked(self) -> None:
        """Step to the next deepest path."""
        if not self._sorted_edges:
            if not self._build_sorted_edges():
                return
            self._current_idx = 0
        else:
            self._current_idx = min(
                self._current_idx + 1, len(self._sorted_edges) - 1
            )
        self._highlight_at_index()

    def _on_reset_clicked(self) -> None:
        """Restore the default uniform blue colormap."""
        try:
            self.viewer.data.edge_colormap = EdgeColormap.from_arrays(
                colormap={},
                default_color=self._DEFAULT_BLUE,
            )
            self._set_status("")
        except Exception as e:
            self._set_status(f"Reset error: {e}")


def change_color_attr(
    viewer,
    edge_attribute: str = GENERATION_KEY,
    cmap=plt.cm.viridis,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Change the color of edges based on a specific attribute.

    This function retrieves the specified edge attribute from the skeleton graph,
    normalizes the values using the provided colormap, and updates the edge colors
    in the viewer's data.

    Parameters
    ----------
    viewer : SkelePlexApp
        The SkelePlex application instance containing the skeleton graph data.
    edge_attribute : str
        The name of the edge attribute to use for coloring.
        Defaults to GENERATION_KEY.
    cmap : Colormap
        The colormap to use for normalizing the edge attribute values.
        Defaults to plt.cm.viridis.
    vmin : float | None
        The minimum value for normalization. If None, it will be set to the minimum
        value of the edge attribute.
    vmax : float | None
        The maximum value for normalization. If None, it will be set to the maximum
        value of the edge attribute.
    """
    color_dict = nx.get_edge_attributes(
        viewer.data.skeleton_graph.graph, edge_attribute
    )

    for key, value in color_dict.items():
        if not value:
            color_dict[key] = np.nan

    if vmin is None:
        vmin = np.nanmin(list(color_dict.values()))
    if vmax is None:
        vmax = np.nanmax(list(color_dict.values()))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    color_dict = {k: np.array(cmap(norm(v))) for k, v in color_dict.items()}

    # update
    edge_colormap = EdgeColormap.from_arrays(
        colormap=color_dict,
        default_color=np.array([0, 0, 0, 1], dtype=np.float32),
    )
    viewer.data.edge_colormap = edge_colormap


def filter_edges_by_attribute(
    viewer,
    edge_attribute: str = GENERATION_KEY,
    cmap=plt.cm.viridis,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Filter edges based on a specific attribute and its value range.

    Edges that do not have the specified attribute or whose values
    are outside the specified range will be rendered as transparent.

    Parameters
    ----------
    viewer : SkelePlexApp
        The SkelePlex application instance containing the skeleton graph data.
    edge_attribute : str
        The name of the edge attribute to use for filtering.
        Defaults to GENERATION_KEY.
    cmap : Colormap
        The colormap to use for normalizing the edge attribute values.
        Defaults to plt.cm.viridis.
    vmin : float | None
        The minimum value for normalization. If None, it will be set to the minimum
        value of the edge attribute.
    vmax : float | None
        The maximum value for normalization. If None, it will be set to the maximum
        value of the edge attribute.

    """
    color_dict = nx.get_edge_attributes(
        viewer.data.skeleton_graph.graph, edge_attribute
    )
    for key, value in color_dict.items():
        if not value:
            color_dict[key] = np.nan

    if vmin is None:
        vmin = np.nanmin(list(color_dict.values()))
    if vmax is None:
        vmax = np.nanmax(list(color_dict.values()))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    color_dict = {k: np.array(cmap(norm(v))) for k, v in color_dict.items()}

    edges_to_remove = [
        edge
        for edge, value in nx.get_edge_attributes(
            viewer.data.skeleton_graph.graph, edge_attribute
        ).items()
        if not isinstance(value, numbers.Number) or not (vmin <= value <= vmax)
    ]

    for edge in viewer.data.skeleton_graph.graph.edges:
        # if generation_dict[edge] > levels:
        #     continue
        if edge in edges_to_remove:
            # Render edges that are filtered out as transparent
            color_dict[edge] = np.array([0, 0, 0, 0], dtype=np.float32)  # Transparent
        else:
            color_dict[edge] = color_dict.get(
                edge, np.array([0, 0, 0, 1], dtype=np.float32)
            )

    edge_colormap = EdgeColormap.from_arrays(
        colormap=color_dict,
        default_color=np.array([0, 0, 0, 1], dtype=np.float32),
    )
    viewer.data.edge_colormap = edge_colormap
