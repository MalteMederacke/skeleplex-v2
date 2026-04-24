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
from qtpy.QtWidgets import QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

from skeleplex.graph.constants import (
    EDGE_SPLINE_KEY,
    GENERATION_KEY,
    NODE_COORDINATE_KEY,
)
from skeleplex.graph.modify_graph import (
    connect_without_merging,
    delete_edge,
    merge_nodes,
    split_edge,
)
from skeleplex.visualize import EdgeColormap

if TYPE_CHECKING:
    # prevent circular import
    from skeleplex.app._data import DataManager

import ast


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
        redraw: bool = True,
    ) -> None:
        """Delete an edge from the skeleton graph.

        Parameters
        ----------
        edge : tuple[int, int] | None
            The edge to delete, represented as a tuple of node IDs.
            If None, no action is taken.
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
            delete_edge(skeleton_graph=self._data.skeleton_graph, edge=edge)

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

        split_edge_widget.point_store.coordinates = np.array(
            [point_pos[::-1]], dtype=np.float32
        )
        split_edge_widget.point_visual.appearance.visible = True
        viewer._viewer._backend.reslice_all()

    split_edge_widget.split_pos.changed.connect(preview_split)
    split_edge_widget.point_visual, split_edge_widget.point_store = viewer.add_points()
    split_edge_widget.point_visual.appearance.visible = False

    return split_edge_widget


class ChangeBranchColorWidget:
    def __init__(self, viewer):
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

        # Layout to contain both magicgui and colorbar
        self.container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.widget.native)
        layout.addWidget(self.run_button)
        layout.addWidget(self.filter_button)
        layout.addWidget(self.colorbar_label)
        self.container.setLayout(layout)

        self.widget.edge_attribute.changed.connect(self._on_attribute_change)
        self._on_attribute_change(self.widget.edge_attribute.value)

        self.widget.vmin.changed.connect(self._on_slider_value_change)
        self.widget.vmax.changed.connect(self._on_slider_value_change)

        # Add the full container widget to the viewer
        self.viewer.add_auxiliary_widget(self.container, name="Change Branch Color")

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
            return 0,0
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
    graph: nx.Graph, edge: tuple[int, ...]
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

    Returns
    -------
    set[tuple[int, int]]
        Set of (u, v) edge keys for all reachable edges, including the input edge.
    """
    u, v = edge[0], edge[1]

    if isinstance(graph, nx.DiGraph):
        # Downstream: BFS from v along outgoing edges
        reachable: set[tuple[int, int]] = {(u, v)}
        for a, b in nx.dfs_edges(graph, v):
            reachable.add((a, b))
    else:
        # Undirected: entire connected component containing u/v
        component = nx.node_connected_component(graph, u)
        subgraph = graph.subgraph(component)
        reachable = {(a, b) for a, b in subgraph.edges()}

    return reachable


class RenderReachableEdgesWidget:
    """Widget that highlights edges reachable from a given input edge.

    For directed graphs, highlights all downstream edges following outgoing
    directions from the edge's target node. For undirected graphs, highlights
    all edges in the same connected component.

    The input edge is shown in red, reachable edges in green, and all
    other edges are made transparent.

    Parameters
    ----------
    viewer : SkelePlexApp
        The SkelePlex application instance.
    """

    _INPUT_COLOR = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    _REACHABLE_COLOR = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    _OTHER_COLOR = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def __init__(self, viewer) -> None:
        self.viewer = viewer

        label = QLabel("Edge key:")
        self._edge_input = QLineEdit()
        self._edge_input.setPlaceholderText("e.g. {(1, 2)}")

        run_button = QPushButton("Render reachable edges")
        run_button.clicked.connect(self._on_run_clicked)

        reset_button = QPushButton("Reset colors")
        reset_button.clicked.connect(self._on_reset_clicked)

        self.container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self._edge_input)
        layout.addWidget(run_button)
        layout.addWidget(reset_button)
        self.container.setLayout(layout)

        viewer.add_auxiliary_widget(self.container, name="Render Reachable Edges")

    def _on_run_clicked(self) -> None:
        """Parse the input edge, find reachable edges, and update edge colors."""
        try:
            edges = edge_string_to_key(self._edge_input.text())
        except ValueError:
            return
        if not edges:
            return

        edge = next(iter(edges))
        graph = self.viewer.data.skeleton_graph.graph
        reachable = get_reachable_edges(graph, edge)
        input_key = (edge[0], edge[1])

        color_dict: dict[tuple[int, int], np.ndarray] = {}
        for graph_edge in graph.edges():
            key = (graph_edge[0], graph_edge[1])
            if key == input_key:
                color_dict[key] = self._INPUT_COLOR.copy()
            elif key in reachable:
                color_dict[key] = self._REACHABLE_COLOR.copy()
            else:
                color_dict[key] = self._OTHER_COLOR.copy()

        self.viewer.data.edge_colormap = EdgeColormap.from_arrays(
            colormap=color_dict,
            default_color=self._OTHER_COLOR,
        )

    def _on_reset_clicked(self) -> None:
        """Restore the default uniform blue colormap."""
        self.viewer.data.edge_colormap = EdgeColormap(
            colormap={},
            default_color=EdgeColormap.from_arrays(
                colormap={},
                default_color=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
            ).default_color,
        )


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
