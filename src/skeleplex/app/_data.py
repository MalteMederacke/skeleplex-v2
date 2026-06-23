"""Module for handling data in the SkelePlex application."""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import zarr
from cmap import Color
from psygnal import EventedModel, Signal, SignalGroup
from pydantic.types import FilePath

from skeleplex.graph import SkeletonGraph
from skeleplex.graph.constants import NODE_COORDINATE_KEY
from skeleplex.utils import line_segments_in_aabb, points_in_aabb
from skeleplex.visualize import EdgeColormap, line_segment_coordinates_from_spline

log = logging.getLogger(__name__)


class SkeletonGraphFile(EventedModel):
    """A class storing the state of the skeleton graph file.

    Parameters
    ----------
    path : Path
        The path to the skeleplex skeleton graph file.
    """

    path: Path


class ImageFile(EventedModel):
    """A class storing the state of the segmentation file.

    Parameters
    ----------
    path : Path
        The path to the segmentation image file.
    voxel_size_um : tuple[float, float, float]
        The voxel size of the segmentation in micrometers.
    """

    path: Path
    voxel_size_um: tuple[float, float, float]


class SkeletonDataPaths(EventedModel):
    """A class storing the state of the skeleton dataset.

    Parameters
    ----------
    image : FilePath | None
        The path to the image file.
    segmentation : Path | None
        The path to the segmentation image file.
    skeleton_graph : FilePath
        The path to the skeleton graph file.
    """

    image: FilePath | None = None
    segmentation: ImageFile | None = None
    skeleton_graph: SkeletonGraphFile | None = None

    def has_paths(self) -> bool:
        """Returns true if any of the paths are set."""
        return any([self.image, self.segmentation, self.skeleton_graph])


class ViewMode(Enum):
    """The different viewing modes.

    NONE: Show no data.
    ALL: Show all data.
    BOUNDING_BOX: Show data in a specified bounding box.
    NODE: Show data around a specified node.
    """

    NONE = "none"
    ALL = "all"
    BOUNDING_BOX = "bounding_box"
    NODE = "node"


class BoundingBoxEvents(SignalGroup):
    """Events for the DataManager class."""

    data = Signal()


class BoundingBoxData:
    """The current bounding box parameters."""

    events = BoundingBoxEvents()

    def __init__(
        self,
        min_coordinate: np.ndarray | None = None,
        max_coordinate: np.ndarray | None = None,
    ):
        self._min_coordinate = min_coordinate
        self._max_coordinate = max_coordinate

    @property
    def min_coordinate(self) -> np.ndarray | None:
        """Returns the minimum corner of the bounding box."""
        return self._min_coordinate

    @min_coordinate.setter
    def min_coordinate(self, min_coordinate: np.ndarray | None):
        """Set the minimum coordinate of the bounding box to be rendered.

        Parameters
        ----------
        min_coordinate : np.ndarray | None
            The minimum coordinate of the axis-aligned bounding box.
            If None, no corner is set. The bounding box rendering mode
            requires min_coordinate to be set.
        """
        self._min_coordinate = np.asarray(min_coordinate)
        self.events.data.emit()

    @property
    def max_coordinate(self) -> np.ndarray | None:
        """Returns the minimum corner of the bounding box."""
        return self._max_coordinate

    @max_coordinate.setter
    def max_coordinate(self, max_coordinate: np.ndarray | None):
        """Set the maximum coordinate of the bounding box to be rendered.

        Parameters
        ----------
        max_coordinate : np.ndarray | None
            The maximum coordinate of the axis-aligned bounding box.
            If None, no corner is set. The bounding box rendering mode
            requires max_coordinate to be set.
        """
        self._max_coordinate = np.asarray(max_coordinate)
        self.events.data.emit()

    @property
    def is_populated(self) -> bool:
        """Returns True if the min and max coordinate have been set."""
        return (self.min_coordinate is not None) and (self.max_coordinate is not None)


@dataclass(frozen=True)
class ViewRequest:
    """Base Request to view data in the skeleton graph.

    Do not use this class directly, use one of the subclasses instead.
    """

    pass


@dataclass(frozen=True)
class AllViewRequest(ViewRequest):
    """Request to view all data in the skeleton graph.

    This is used for passing requests to view all data in the skeleton graph.
    It does not require any parameters.
    """

    pass


@dataclass(frozen=True)
class NoneViewRequest(ViewRequest):
    """Request to view no data in the skeleton graph.

    This is used for passing requests to view no data in the skeleton graph.
    It does not require any parameters.
    """

    pass


@dataclass(frozen=True)
class BoundingBoxViewRequest(ViewRequest):
    """Request to view an axis-aligned bounding box region.

    Parameters
    ----------
    minimum : np.ndarray
        The minimum corner of the axis-aligned bounding box.
    maximum : np.ndarray
        The maximum corner of the axis-aligned bounding box.
    """

    minimum: np.ndarray
    maximum: np.ndarray


class ViewEvents(SignalGroup):
    """Events for the DataManager class."""

    data = Signal()
    mode = Signal()


class SkeletonDataView:
    """A class to manage the current view on the skeleton data."""

    events = ViewEvents()

    def __init__(
        self,
        data_manager: "DataManager",
        bounding_box: BoundingBoxData,
        mode: ViewMode = ViewMode.ALL,
    ) -> None:
        self._data_manager = data_manager
        self._bounding_box = bounding_box
        self._mode = mode

        # initialize the data
        self._edge_coordinates: np.ndarray | None = None
        self._edge_indices: np.ndarray | None = None
        self._edge_keys: np.ndarray | None = None
        self._edge_colors: np.ndarray | None = None
        self._highlighted_edge_keys: np.ndarray | None = None
        self._node_coordinates: np.ndarray | None = None
        self._node_keys: np.ndarray | None = None
        self._highlighted_node_keys: np.ndarray | None = None

    @property
    def bounding_box(self) -> BoundingBoxData:
        """Get the current bounding box data."""
        return self._bounding_box

    @property
    def mode(self) -> ViewMode:
        """Get the current view mode."""
        return self._mode

    @mode.setter
    def mode(self, mode: ViewMode | str) -> None:
        """Set the current view mode."""
        if not isinstance(mode, ViewMode):
            mode = ViewMode(mode.lower())

        if mode == ViewMode.BOUNDING_BOX and not self.bounding_box.is_populated:
            raise ValueError(
                "The bounding box must be populated to set bounding box mode."
            )
        self._mode = mode
        self.update()

    @property
    def node_coordinates(self) -> np.ndarray | None:
        """Get the coordinates of the current view of the nodes in the skeleton graph.

        (n_nodes, 3) array of node coordinates.
        """
        return self._node_coordinates

    @property
    def node_keys(self) -> np.ndarray | None:
        """Get the keys of the nodes in the rendered graph.

        (n_nodes,) array of node keys. These are index-matched with
        the node_coordinates array.
        """
        return self._node_keys

    @property
    def highlighted_node_keys(self) -> np.ndarray | None:
        """Get the indices of the highlighted nodes in the rendered graph.

        (n_highlighted_nodes,) array of node indices.
        """
        return self._highlighted_node_keys

    @property
    def edge_coordinates(self) -> np.ndarray | None:
        """Get the coordinates of the current view of the edges in the rendered graph.

        (n_edges x 2 x n_points_per_edge, 3) array of edge coordinates.
        """
        return self._edge_coordinates

    @property
    def edge_indices(self) -> np.ndarray | None:
        """Get the indices of the current view of the edges in the rendered graph.

        (n_edges x 2 x n_points_per_edge,) array of edge indices.
        """
        return self._edge_indices

    @property
    def edge_keys(self) -> np.ndarray | None:
        """Get the keys of the edge for each edge coordinate in the rendered graph.

        (n_edges x 2 x n_points_per_edge,) array of edge keys.
        """
        return self._edge_keys

    @property
    def edge_colors(self) -> np.ndarray | None:
        """Get the colors of the edges in the rendered graph.

        (n_edges x 2 x n_points_per_edge, 4) array of RGBA colors.
        """
        return self._edge_colors

    @property
    def highlighted_edge_keys(self) -> np.ndarray | None:
        """Get keys of the highlighted edges for each coordinate in the rendered graph.

        (n_edges x 2 x n_points_per_edge,) array of highlighted edge keys.
        """
        return self._highlighted_edge_keys

    def update(self) -> None:
        """Update the data for the currently specified view.

        This updates the edge coordinates, edge indices, and node indices.
        """
        if self._data_manager.skeleton_graph is None:
            # if the data isn't loaded, nothing to update
            return
        if self._mode == ViewMode.ALL:
            (
                self._node_coordinates,
                self._node_keys,
                self._edge_coordinates,
                self._edge_indices,
                self._edge_keys,
                self._edge_colors,
            ) = self._get_view_all()
            self._highlighted_edge_keys = np.empty((0, 2))
        elif self._mode == ViewMode.BOUNDING_BOX:
            (
                self._node_coordinates,
                self._node_keys,
                self._edge_coordinates,
                self._edge_indices,
                self._edge_keys,
                self._edge_colors,
            ) = self._get_view_bounding_box()
            self._highlighted_edge_keys = np.empty((0, 2))
        else:
            raise NotImplementedError(f"View mode {self._mode} not implemented.")

        # Emit signal that the view data has been updated
        self.events.data.emit()

    def _get_view_all(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the view for all data.

        Returns
        -------
        node_coordinates : np.ndarray
            (n_nodes,3) array of the coordinates of
            the rendered nodes in the skeleton graph.
        node_keys : np.ndarray
            (n_nodes) array of the keys of the nodes in the skeleton graph.
        edge_coordinates : np.ndarray
            (n_edges x 2 x n_points_per_edge, 3) array of the coordinates of
            the rendered edges in the skeleton graph.
        edge_indices : np.ndarray
            The indices of the edges in the skeleton graph.
        edge_keys : np.ndarray
            The keys of the edge for each edge coordinate in the skeleton graph.
        edge_colors : np.ndarray
            (n_edges x 2 x n_points_per_edge, 4) array of RGBA colors for the edges.

        """
        return (
            self._data_manager.node_coordinates,
            self._data_manager.node_keys,
            self._data_manager.edge_coordinates,
            self._data_manager.edge_indices,
            self._data_manager.edge_keys,
            self._data_manager.edge_colors,
        )

    def _get_view_bounding_box(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the view for the current bounding box.

        Returns
        -------
        node_coordinates : np.ndarray
            (n_nodes,3) array of the coordinates of
            the rendered nodes in the skeleton graph.
        node_keys : np.ndarray
            (n_nodes) array of the keys of the nodes in the skeleton graph.
        edge_coordinates : np.ndarray
            (n_edges x 2 x n_points_per_edge, 3) array of the coordinates of
            the rendered edges in the skeleton graph.
        edge_indices : np.ndarray
            The indices of the edges in the skeleton graph.
        edge_keys : np.ndarray
            The keys of the edge for each edge coordinate in the skeleton graph.

        """
        if not self.bounding_box.is_populated:
            raise ValueError(
                "DataView.bounding_box must be set to use bounding box mode."
            )

        # get the vertices in the bounding box
        nodes_inside = points_in_aabb(
            coordinates=self._data_manager.node_coordinates,
            min_bounds=self.bounding_box.min_coordinate,
            max_bounds=self.bounding_box.max_coordinate,
        )
        node_coordinates = self._data_manager.node_coordinates[nodes_inside]
        node_keys = self._data_manager.node_keys[nodes_inside]

        # get the edges in the bounding box
        edges_inside = line_segments_in_aabb(
            line_segments=self._data_manager.edge_coordinates,
            min_bounds=self.bounding_box.min_coordinate,
            max_bounds=self.bounding_box.max_coordinate,
        )
        edge_coordinates = self._data_manager.edge_coordinates[edges_inside]
        edge_indices = self._data_manager.edge_indices[edges_inside]
        edge_keys = self._data_manager.edge_keys[edges_inside]
        edge_colors = self._data_manager.edge_colors[edges_inside]

        return (
            node_coordinates,
            node_keys,
            edge_coordinates,
            edge_indices,
            edge_keys,
            edge_colors,
        )

    def _on_view_request(self, request: BoundingBoxViewRequest):
        """Handle a request to change the view.

        This updates the bounding box and view mode based on the request.
        This is generally called by the GUI widget events.
        """
        if isinstance(request, AllViewRequest):
            self.mode = ViewMode.ALL

        elif isinstance(request, BoundingBoxViewRequest):
            # set the bounding box coordinates
            # use the private attribute to prevent the event from being emitted.
            self.bounding_box._min_coordinate = request.minimum
            self.bounding_box._max_coordinate = request.maximum

            # set the view mode to bounding box
            self.mode = ViewMode.BOUNDING_BOX

        else:
            raise TypeError(f"Unknown view request type: {type(request)}.")


class SegmentationView:
    events = ViewEvents()

    def __init__(
        self,
        data_manager: "DataManager",
        bounding_box: BoundingBoxData,
        mode: ViewMode = ViewMode.ALL,
    ) -> None:
        self._data_manager = data_manager
        self._bounding_box = bounding_box
        self._mode = mode

        # initialize the data
        self._array: np.ndarray | None = None
        self._transform: np.ndarray | None = None

    @property
    def bounding_box(self) -> BoundingBoxData:
        """Get the current bounding box data."""
        return self._bounding_box

    @property
    def mode(self) -> ViewMode:
        """Get the current view mode."""
        return self._mode

    @mode.setter
    def mode(self, mode: ViewMode | str) -> None:
        """Set the current view mode."""
        if not isinstance(mode, ViewMode):
            mode = ViewMode(mode.lower())

        if mode == ViewMode.BOUNDING_BOX and not self.bounding_box.is_populated:
            raise ValueError(
                "The bounding box must be populated to set bounding box mode."
            )
        self._mode = mode
        self.update()

    @property
    def array(self) -> np.ndarray | None:
        """The current segmentation array for the view."""
        return self._array

    @property
    def transform(self) -> np.ndarray | None:
        """The current segmentation transform for the view."""
        return self._transform

    def update(self) -> None:
        """Update the data for the currently specified view."""
        if self._data_manager._segmentation is None:
            # if the data isn't loaded, nothing to update
            return
        if self._mode == ViewMode.NONE:
            # Don't display anything
            self._array = None

        elif self._mode == ViewMode.ALL:
            self._array = np.asarray(self._data_manager._segmentation)

            # build the transform in data (z, y, x) axis order; cellier reverses
            # to pygfx order internally.
            scale = np.array(self._data_manager.segmentation_scale)  # (z, y, x)
            transform = np.eye(4)
            transform[0, 0] = scale[0]  # z
            transform[1, 1] = scale[1]  # y
            transform[2, 2] = scale[2]  # x
            self._transform = transform

        elif self._mode == ViewMode.BOUNDING_BOX:
            scale = np.array(self._data_manager.segmentation_scale)
            segmentation_shape = self._data_manager._segmentation.shape
            # get the bounding box in voxel coordinates
            bounding_box_min_vx = np.round(self.bounding_box.min_coordinate / scale)
            bounding_box_max_vx = np.round(self.bounding_box.max_coordinate / scale)

            # clamp the bounding box to the segmentation shape
            bounding_box_min_vx = np.maximum(bounding_box_min_vx, 0).astype(np.int64)
            bounding_box_max_vx = np.minimum(
                bounding_box_max_vx, segmentation_shape
            ).astype(np.int64)

            self._array = np.asarray(
                self._data_manager.segmentation[
                    bounding_box_min_vx[0] : bounding_box_max_vx[0],
                    bounding_box_min_vx[1] : bounding_box_max_vx[1],
                    bounding_box_min_vx[2] : bounding_box_max_vx[2],
                ]
            )

            # build the transform in data (z, y, x) axis order, including the
            # translation; cellier reverses to pygfx order internally.
            transform = np.eye(4)
            transform[0, 0] = scale[0]  # z
            transform[1, 1] = scale[1]  # y
            transform[2, 2] = scale[2]  # x
            transform[0, 3] = bounding_box_min_vx[0] * scale[0]  # z
            transform[1, 3] = bounding_box_min_vx[1] * scale[1]  # y
            transform[2, 3] = bounding_box_min_vx[2] * scale[2]  # x
            self._transform = transform
        else:
            raise NotImplementedError(f"View mode {self._mode} not implemented.")

        # Emit signal that the view data has been updated
        self.events.data()

    def _on_view_request(self, request: BoundingBoxViewRequest):
        """Handle a request to change the view.

        This updates the bounding box and view mode based on the request.
        This is generally called by the GUI widget events.
        """
        if isinstance(request, NoneViewRequest):
            self.mode = ViewMode.NONE

        elif isinstance(request, AllViewRequest):
            self.mode = ViewMode.ALL

        elif isinstance(request, BoundingBoxViewRequest):
            # set the bounding box coordinates
            # use the private attribute to prevent the event from being emitted.
            self.bounding_box._min_coordinate = request.minimum.astype(np.int64)
            self.bounding_box._max_coordinate = request.maximum.astype(np.int64)

            # set the view mode to bounding box
            self.mode = ViewMode.BOUNDING_BOX

        else:
            raise TypeError(f"Unknown view request type: {type(request)}.")


class EdgeSelectionManager(EventedModel):
    """Class to manage selection of edge in the viewer.

    Parameters
    ----------
    enabled : bool
        Set to true if the edge selection is enabled.
        The default value is False.
    values : set[tuple[int, int]] | None
        The selected edges.
    """

    enabled: bool
    values: set[tuple[int, int]]


class NodeSelectionManager(EventedModel):
    """Class to manage selection of nodes in the viewer.

    Parameters
    ----------
    enabled : bool
        Set to true if the edge selection is enabled.
        The default value is False.
    values : set[int] | None
        The selected nodes.
    """

    enabled: bool
    values: set[int]


@dataclass(frozen=True)
class EdgeSelectionPasteRequest:
    """Selected edges to paste.

    This is used for passing selected edges to the paste operation.
    For example, when pasting edges from the selection to a
    GUI widget.

    Parameters
    ----------
    edge_key : set[tuple[int, int]]
        The keys of the selected edges.
    """

    edge_keys: set[tuple[int, int]]


@dataclass(frozen=True)
class NodeSelectionPasteRequest:
    """Selected nodes to paste.

    This is used for passing selected nodes to the paste operation.
    For example, when pasting nodes from the selection to a
    GUI widget.

    Parameters
    ----------
    node_keys : set[int]
        The keys of the selected nodes.
    """

    node_keys: set[int]


class SelectionManager(EventedModel):
    """Class to manage selection of data in the viewer."""

    edge: EdgeSelectionManager
    node: NodeSelectionManager

    def _make_edge_selection_paste_request(self):
        """Create a paste request for the selected edges.

        This emits the request as a SelectionManager.events.edge signal.
        """
        request = EdgeSelectionPasteRequest(edge_keys=self.edge.values)
        self.events.edge.emit(request)

    def _make_node_selection_paste_request(self):
        """Create a paste request for the selected nodes.

        This emits the request as a SelectionManager.events.node signal.
        """
        request = NodeSelectionPasteRequest(node_keys=self.node.values)
        self.events.node.emit(request)

    def _on_edge_enabled_update(self, event):
        """Callback when the UI updates the edge selection enabled state."""
        self.edge.enabled = event > 0

    def _on_node_enabled_update(self, event):
        """Callback when the UI updates the node selection enabled state."""
        self.node.enabled = event > 0


class DataEvents(SignalGroup):
    """Events for the DataManager class."""

    data = Signal()


class DataManager:
    """A class to manage data.

    Parameters
    ----------
    file_paths : SkeletonDataPaths
        The paths to the data files.
    selection : SelectionManager | None
        The selection manager.

    Attributes
    ----------
    events : DataEvents
        The events for the DataManager class.
    node_coordinates : np.ndarray | None
        The coordinates of the nodes in the skeleton graph.
        This is None when the skeleton hasn't been loaded.
    node_keys : np.ndarray | None
        The keys for the nodes in the skeleton graph.
        This is index-matched with the node_coordinates array.
        This is None when the skeleton hasn't been loaded.
    edge_coordinates : np.ndarray | None
        The coordinates for rendering the edges in the skeleton graph
        as line segments. Each edge is rendered as a sequence of line
        segments. All line segments are stored in this array.
        The array has shape (n_line_segments * 2, 3) where
        line segment n is defined by the start coordinate
        edge_coordinates[n * 2, :] and end coordinate
        edge_coordinates[n * 2 + 1, :].
        This is None when the skeleton hasn't been loaded.
    edge_indices : np.ndarray | None
        The indices for the edges in the skeleton graph.
        This is None when the skeleton hasn't been loaded.
    edge_keys : np.ndarray | None
        The keys for edges of each edge coordinate in the skeleton graph.
        This is None when the skeleton hasn't been loaded.
    edge_colors : np.ndarray | None
        The colors for the edges in the skeleton graph. These are
        index-matched with the edge_coordinates array.
    """

    events = DataEvents()

    def __init__(
        self,
        file_paths: SkeletonDataPaths,
        selection: SelectionManager | None = None,
        edge_colormap: EdgeColormap | None = None,
        load_data: bool = True,
    ) -> None:
        self._file_paths = file_paths

        self._skeleton_view = SkeletonDataView(
            data_manager=self, mode=ViewMode.ALL, bounding_box=BoundingBoxData()
        )

        self._segmentation_view = SegmentationView(
            data_manager=self, mode=ViewMode.NONE, bounding_box=BoundingBoxData()
        )

        # make the selection model
        if selection is None:
            selection = SelectionManager(
                edge=EdgeSelectionManager(enabled=False, values=set()),
                node=NodeSelectionManager(enabled=False, values=set()),
            )
        self._selection = selection

        # make the edge colormap
        if edge_colormap is None:
            # default edge color is blue
            edge_colormap = EdgeColormap(
                colormap={}, default_color=Color([0.0, 0.0, 1.0, 1.0])
            )
        self._edge_colormap = edge_colormap

        # initialize the skeleton data data
        self._skeleton_graph: SkeletonGraph | None = None
        self._node_coordinates: np.ndarray | None = None
        self._edge_coordinates: np.ndarray | None = None
        self._edge_indices: np.ndarray | None = None
        self._edge_keys: np.ndarray | None = None
        self._edge_colors: np.ndarray | None = None

        # initialize the segmentation data
        self._segmentation = None
        self._segmentation_scale = None

        if self.files.has_paths() and load_data:
            self.load()

        # connect the event for updating the view when the data is changed
        self.events.data.connect(self._skeleton_view.update)

    @property
    def files(self) -> SkeletonDataPaths:
        """Get the file paths."""
        return self._file_paths

    @property
    def skeleton_view(self) -> SkeletonDataView:
        """Get the current data view."""
        return self._skeleton_view

    @property
    def segmentation_view(self) -> SegmentationView:
        """Get the current segmentation view."""
        return self._segmentation_view

    @property
    def selection(self) -> SelectionManager:
        """Get the current data selection."""
        return self._selection

    @property
    def skeleton_graph(self) -> SkeletonGraph | None:
        """Get the skeleton graph."""
        return self._skeleton_graph

    @property
    def segmentation(self) -> zarr.Array | None:
        """Get the segmentation zarr array."""
        return self._segmentation

    @property
    def segmentation_scale(self) -> tuple[float, float, float] | None:
        """Get the scale for the segmentation rendering.

        Since the viewer coordinate system is in microns,
        this is equivalent to segmentation voxel size in micrometers.
        """
        if self.files.segmentation is not None:
            return self.files.segmentation.voxel_size_um
        return None

    @property
    def node_coordinates(self) -> np.ndarray | None:
        """Get the coordinates of the nodes in the skeleton graph.

        (n_nodes, 3) array of node coordinates.
        """
        return self._node_coordinates

    @property
    def node_keys(self) -> np.ndarray | None:
        """Get the keys of the nodes in the skeleton graph.

        (n_nodes,) array of node keys. These are index-matched with
        the node_coordinates array.
        """
        return self._node_keys

    @property
    def edge_coordinates(self) -> np.ndarray | None:
        """Get the coordinates of the edges in the skeleton graph.

        (n_edges x 2 x n_points_per_edge, 3) array of edge coordinates.
        """
        return self._edge_coordinates

    @property
    def edge_indices(self) -> np.ndarray | None:
        """Get the indices of the edges in the skeleton graph.

        (n_edges x 2 x n_points_per_edge,) array of edge indices.
        """
        return self._edge_indices

    @property
    def edge_keys(self) -> np.ndarray | None:
        """Get the keys of the edge for each edge coordinate in the skeleton graph.

        (n_edges x 2 x n_points_per_edge,) array of edge keys.
        """
        return self._edge_keys

    @property
    def edge_colors(self) -> np.ndarray | None:
        """Get the colors of each edge line segment.

        (n_edges x 2 x n_points_per_edge, 4) array of RGBA colors.
        """
        return self._edge_colors

    @property
    def edge_colormap(self) -> EdgeColormap:
        """Get the edge colormap."""
        return self._edge_colormap

    @edge_colormap.setter
    def edge_colormap(self, colormap: EdgeColormap, emit_event: bool = True) -> None:
        """Set the edge colormap.

        Parameters
        ----------
        colormap : EdgeColormap
            The new edge colormap to use.
        emit_event : bool
            If set to True, the DataManager.events.data signal is emitted.
            This will trigger a redraw of the data view.
        """
        self._edge_colormap = colormap

        # update the edge colors
        self._update_edge_colors()

        if emit_event:
            # update the data event for triggering redraw
            self.events.data.emit()

    def _load_skeleton(self):
        """Load the skeleton graph data."""
        if self.files.skeleton_graph:
            # load the skeleton graph
            log.info(f"Loading skeleton graph from {self.files.skeleton_graph}")
            self._skeleton_graph = SkeletonGraph.from_json_file(
                self.files.skeleton_graph.path
            )
            self._update_node_coordinates()
            self._update_edge_coordinates()
            self._update_edge_colors()
        else:
            log.info("No skeleton graph loaded.")
            self._skeleton_graph = None

    def _load_segmentation(self):
        """Load the segmentation data."""
        if self.files.segmentation:
            log.info(f"Segmentation path set to {self.files.segmentation}")
            self._segmentation = zarr.open(self.files.segmentation.path, mode="r")

        else:
            log.info("No segmentation path set.")
            self._segmentation = None

    def load(self) -> None:
        """Load data."""
        self._load_skeleton()
        self._load_segmentation()

        # Emit the event to trigger re-slicing and re-rendering
        self.events.data.emit()

    def to_dict(self) -> dict:
        """Convert to json-serializable dictionary."""
        return self._data.to_dict()

    def _update_node_coordinates(self) -> None:
        """Get and store the node coordinates from the skeleton graph.

        todo: make it possible to update without recompute everything
        """
        if self._skeleton_graph is None:
            # don't do anything if the skeleton graph is not loaded
            return

        node_coordinates = []
        node_keys = []
        for key, node_data in self.skeleton_graph.graph.nodes(data=True):
            node_keys.append(key)
            node_coordinates.append(node_data[NODE_COORDINATE_KEY])
        self._node_coordinates = np.array(node_coordinates, dtype=np.float32)
        self._node_keys = np.array(node_keys, dtype=np.int32)

    def _update_edge_coordinates(self) -> None:
        """Get and store the edge spline coordinates from the skeleton graph.

        todo: make it possible to update without recomputing everything
        """
        if self._skeleton_graph is None:
            return None

        edge_coordinates = []
        edge_indices = []
        edge_keys = []
        for edge_index, (edge_key, edge_spline) in enumerate(
            self.skeleton_graph.edge_splines.items()
        ):
            # get the edge coordinates
            new_coordinates = line_segment_coordinates_from_spline(edge_spline)
            edge_coordinates.append(new_coordinates)

            # get the edge indices
            n_coordinates = len(new_coordinates)
            edge_indices.append(np.full((n_coordinates,), edge_index))

            # get the edge keys
            edge_keys.append(np.tile(edge_key, (n_coordinates, 1)))

        self._edge_coordinates = np.concatenate(edge_coordinates)
        self._edge_indices = np.concatenate(edge_indices)
        self._edge_keys = np.concatenate(edge_keys)

    def _update_edge_colors(self) -> None:
        """Update the edge colors based on the current edge colormap.

        If the edge coordinates or the edges are not set, this does
        not do anything. Thus, this must be called after
        DataManager._update_edge_coordinates().
        """
        if self._edge_coordinates is None or self._edge_keys is None:
            # don't do anything if the edge coordinates or keys are not set
            return

        # map the edge keys to colors using the colormap
        self._edge_colors = self.edge_colormap.map_array(self._edge_keys)

    def _update_skeleton_file_load_data(
        self,
        new_skeleton_file: SkeletonGraphFile,
    ) -> None:
        """Update the file paths and load the new data.

        This is a method intended to be used to generate a magicgui widget
        for the GUI.
        """
        self.files.skeleton_graph = new_skeleton_file
        self._load_skeleton()
        self.events.data.emit()

    def _update_segmentation_file_load_data(
        self,
        new_segmentation_file: ImageFile,
    ) -> None:
        self.files.segmentation = new_segmentation_file
        self._load_segmentation()
        self.events.data.emit()

    def _on_edge_selection_click(self, event, click_source: str = "data"):
        """Callback for the edge picking event from the renderer.

        Parameters
        ----------
        event : cellier.events.CanvasMousePress3DEvent
            The 3D canvas press event from cellier.
        click_source : str
            The source of the click event. One of "data" (the main visual),
            "highlight" (the highlight visual), or "background" (empty space or a
            non-skeleton visual).
        """
        if ("Control" not in event.modifiers) or (event.button != 1):
            # only pick with control + LMB (left button == 1)
            return

        if click_source == "background":
            # clicking empty space clears the selection, unless Shift is held
            # (Shift+Ctrl is the additive gesture, so a miss is a no-op).
            if "Shift" not in event.modifiers and self.selection.edge.values:
                self.selection.edge.values.clear()
                self.selection.edge.events.values.emit(self.selection.edge.values)
            return

        pick = event.pick_info
        if pick.details is None:
            # no element-level pick info; nothing to select
            return

        # LinesPickInfo.edge_index is the store segment index; edge_keys is
        # per-vertex, so the two vertices of segment s share edge_keys[2 * s].
        edge_index = pick.details.edge_index

        # get the edge key from the edge index
        if click_source == "data":
            edge_key_numpy = self.skeleton_view.edge_keys[2 * edge_index]
            edge_key = (int(edge_key_numpy[0]), int(edge_key_numpy[1]))
        elif click_source == "highlight":
            edge_key = tuple(self.skeleton_view.highlighted_edge_keys[2 * edge_index])
        else:
            raise ValueError(f"Unknown click source: {click_source}")

        if edge_key in self.selection.edge.values:
            # if the edge is already selected, deselect it.
            self.selection.edge.values.remove(edge_key)
        else:
            # if the edge is not selected, select it.
            if "Shift" not in event.modifiers:
                # if shift is not pressed, clear the selection
                self.selection.edge.values.clear()
            self.selection.edge.values.add(edge_key)
        self.selection.edge.events.values.emit(self.selection.edge.values)

    def _on_node_selection_click(self, event, click_source: str = "data"):
        """Callback for the node picking event from the renderer.

        Parameters
        ----------
        event : cellier.events.CanvasMousePress3DEvent
            The 3D canvas press event from cellier.
        click_source : str
            The source of the click event. One of "data" (the main visual),
            "highlight" (the highlight visual), or "background" (empty space or a
            non-skeleton visual).
        """
        if ("Control" not in event.modifiers) or (event.button != 1):
            # only pick with control + LMB (left button == 1)
            return

        if click_source == "background":
            # clicking empty space clears the selection, unless Shift is held
            # (Shift+Ctrl is the additive gesture, so a miss is a no-op).
            if "Shift" not in event.modifiers and self.selection.node.values:
                self.selection.node.values.clear()
                self.selection.node.events.values.emit(self.selection.node.values)
            return

        pick = event.pick_info
        if pick.details is None:
            # no element-level pick info; nothing to select
            return

        # PointsPickInfo.point_index is 1:1 with node_keys
        point_index = pick.details.point_index

        # get the node key from the point index
        if click_source == "data":
            node_key = int(self.skeleton_view.node_keys[point_index])
        elif click_source == "highlight":
            node_key = int(self.skeleton_view.highlighted_node_keys[point_index])
        else:
            raise ValueError(f"Unknown click source: {click_source}")

        if node_key in self.selection.node.values:
            # if the node is already selected, deselect it.
            self.selection.node.values.remove(node_key)
        else:
            # if the node is not selected, select it.
            if "Shift" not in event.modifiers:
                # if shift is not pressed, clear the selection
                self.selection.node.values.clear()
            self.selection.node.values.add(node_key)
        self.selection.node.events.values.emit(self.selection.node.values)
