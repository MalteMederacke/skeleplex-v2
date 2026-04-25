"""The main application model."""

import logging

import numpy as np
from app_model import Application
from app_model.types import Action, KeyBindingRule, KeyCode, KeyMod, MenuRule
from cellier.models.data_stores import PointsMemoryStore
from cellier.models.visuals import PointsUniformAppearance, PointsVisual

from skeleplex.app._constants import CommandId, MenuId
from skeleplex.app._curate import CurationManager
from skeleplex.app._data import (
    DataManager,
    SelectionManager,
    SkeletonDataPaths,
)
from skeleplex.app._viewer_controller import ViewerController
from skeleplex.app.actions import ACTIONS
from skeleplex.app.qt import MainWindow

log = logging.getLogger(__name__)


class SkelePlexApp(Application):
    """The main application class."""

    def __init__(
        self, data: DataManager | None = None, selection: SelectionManager | None = None
    ) -> None:
        super().__init__("SkelePlex")

        # make the data model
        if data is None:
            data = DataManager(file_paths=SkeletonDataPaths(), selection=selection)
        self._data = data

        # add the curation manager
        self._curate = CurationManager(
            data_manager=self._data,
        )

        # make the viewer model
        self._viewer = ViewerController()
        # self._viewer = ViewerController(parent_widget=self._main_window)

        # ACTIONS is a list of Action objects.
        for action in ACTIONS:
            self.register_action(action)
        self._register_data_actions()

        self._main_window = MainWindow(
            app=self,
        )
        # This will build a menu bar based on these menus
        self._main_window.setModelMenuBar([MenuId.FILE, MenuId.EDIT, MenuId.DATA])

        # populate the renderer
        self._viewer._populate_viewer_from_model(
            canvas_widget_parent=self._main_window._main_viewer_widget.main_viewer_frame
        )

        for canvas in self._viewer._backend._canvas_widgets.values():
            # add the canvas widgets
            self._main_window._set_main_viewer_widget(canvas)

        # connect the data events
        self._connect_data_events()

        # connect the selection events
        self._connect_selection_events()

        # update the data views
        self.data.skeleton_view.update()
        self.data.segmentation_view.update()

    @property
    def data(self) -> DataManager:
        """Get the data manager."""
        return self._data

    @property
    def curate(self) -> CurationManager:
        """Get the curation manager."""
        return self._curate

    def load_main_viewer_skeleton(self) -> None:
        """Add the data to the main viewer."""
        log.debug("Loading data into the main viewer...")
        if self.data.skeleton_graph is None:
            log.debug("No skeleton graph loaded.")
            return

        self._viewer.main_canvas.update_skeleton_geometry(
            node_coordinates=self.data.skeleton_view.node_coordinates,
            edge_coordinates=self.data.skeleton_view.edge_coordinates,
            edge_colors=self.data.skeleton_view.edge_colors,
        )

    def load_main_viewer_segmentation(self) -> None:
        """Add the segmentation to the main viewer."""
        log.debug("Loading segmentation into the main viewer...")
        if self.data.segmentation is None:
            log.debug("No segmentation image loaded.")
            return

        self._viewer.main_canvas.update_segmentation_image(
            image=self.data.segmentation_view.array,
            transform=self.data.segmentation_view.transform,
        )

    def look_at_skeleton(self) -> None:
        """Set the camera in the main viewer to look at the skeleton."""
        self._viewer.main_canvas.look_at_skeleton()

    def add_auxiliary_widget(self, widget, name: str, area: str = "right") -> None:
        """Add a widget to the right dock of the main window."""
        self._main_window.add_auxiliary_widget(widget, name, area)

    def show(self) -> None:
        """Show the app."""
        self._main_window.show()

    def _on_edge_selection_change(self, event) -> None:
        """Handle a change in the edge selection."""
        if len(event) == 0:
            coordinates = np.empty((0, 3))
            highlighted_edge_keys = np.empty((0, 2))
        else:
            coordinates = []
            highlighted_edge_keys = []
            for edge_index in event:
                edge_mask = np.all(
                    np.equal(self.data.skeleton_view.edge_keys, np.asarray(edge_index)),
                    axis=1,
                )
                coordinates.append(self.data.skeleton_view.edge_coordinates[edge_mask])
                highlighted_edge_keys.append(
                    self.data.skeleton_view.edge_keys[edge_mask]
                )
            coordinates = np.concatenate(coordinates)
            highlighted_edge_keys = np.concatenate(highlighted_edge_keys)

        # set the highlight in the rendered scene
        self._viewer.main_canvas.set_edge_highlight(edge_coordinates=coordinates)

        # store the highlighted edge keys
        self.data.skeleton_view._highlighted_edge_keys = highlighted_edge_keys

    def _on_node_selection_change(self, event) -> None:
        """Handle a change in the node selection."""
        if len(event) == 0:
            coordinates = np.empty((0, 3))
            highlighted_node_keys = np.empty((0,))
        else:
            coordinates = []
            highlighted_node_keys = []
            for node_key in event:
                view_coordinate_index = np.argwhere(
                    self.data.skeleton_view.node_keys == node_key
                )[0]
                coordinates.append(
                    self.data.skeleton_view.node_coordinates[view_coordinate_index]
                )
                highlighted_node_keys.append(node_key)
            coordinates = np.atleast_2d(np.concatenate(coordinates))
            highlighted_node_keys = np.array(highlighted_node_keys)

        # set the highlight in the rendered scene
        self._viewer.main_canvas.set_node_highlight(node_coordinates=coordinates)

        # store the highlighted edge keys
        self.data.skeleton_view._highlighted_node_keys = highlighted_node_keys

    def _register_data_actions(self) -> None:
        """Register actions for adding/removing data to/from the viewer."""
        # add points
        self.register_action(
            Action(
                id=CommandId.LOAD_DATA,
                title="Load data",
                icon="fa6-solid:folder-open",
                callback=self.load_main_viewer_skeleton,
                menus=[MenuRule(id=MenuId.DATA)],
            )
        )
        self.register_action(
            Action(
                id=CommandId.LOOK_AT_SKELETON,
                title="Look at skeleton",
                icon="fa6-solid:house",
                callback=self._viewer.main_canvas.look_at_skeleton,
                menus=[MenuRule(id=MenuId.DATA)],
            )
        )

        self.register_action(
            Action(
                id=CommandId.PASTE_EDGE_SELECTION,
                title="Paste edge selection",
                icon="fa6-solid:paste",
                callback=self.data.selection._make_edge_selection_paste_request,
                menus=[MenuRule(id=MenuId.DATA)],
                keybindings=[
                    KeyBindingRule(primary=KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyD)
                ],
            )
        )
        self.register_action(
            Action(
                id=CommandId.PASTE_NODE_SELECTION,
                title="Paste node selection",
                icon="fa6-solid:paste",
                callback=self.data.selection._make_node_selection_paste_request,
                menus=[MenuRule(id=MenuId.DATA)],
                keybindings=[
                    KeyBindingRule(primary=KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyF)
                ],
            )
        )
        self.register_action(
            Action(
                id=CommandId.UNDO,
                title="Undo",
                icon="fa6-solid:rotate-left",
                callback=self.curate.undo,
                menus=[MenuRule(id=MenuId.EDIT)],
                keybindings=[KeyBindingRule(primary=KeyMod.CtrlCmd | KeyCode.KeyZ)],
            )
        )
        self.register_action(
            Action(
                id=CommandId.REDO,
                title="Redo",
                icon="fa6-solid:rotate-right",
                callback=self.curate.redo,
                menus=[MenuRule(id=MenuId.EDIT)],
                keybindings=[
                    KeyBindingRule(primary=KeyMod.CtrlCmd | KeyMod.Shift | KeyCode.KeyZ)
                ],
            )
        )

    def _connect_data_events(self) -> None:
        """Connect the events for handling changes in the data."""
        # events for when the data loading button is pressed
        # this updates the skeleton paths and loads the data.
        self._main_window.app_controls.widget().load_skeleton_group_box.load_widget.called.connect(
            self.data._update_skeleton_file_load_data
        )

        # this updates the segmentation paths and loads the data.
        self._main_window.app_controls.widget().load_segmentation_group_box.load_widget.called.connect(
            self.data._update_segmentation_file_load_data
        )

        # event for updating the skeleton view when the render button is pressed
        self._main_window.app_controls.widget().skeleton_view_box.view_requested.connect(
            self.data.skeleton_view._on_view_request
        )

        # event for updating the segmentation view when the render button is pressed
        self._main_window.app_controls.widget().segmentation_view_box.view_requested.connect(
            self.data.segmentation_view._on_view_request
        )

        # event for updating the main viewer skeleton when the data view is updated
        self.data.skeleton_view.events.data.connect(self.load_main_viewer_skeleton)

        # event for updating the main viewer segmentation when the data view is updated
        self.data.segmentation_view.events.data.connect(
            self.load_main_viewer_segmentation
        )

    def _connect_selection_events(self) -> None:
        """Connect the events for handling changes to the data selection state.

        These all interface with the SelectionManager.
        """
        # connect the edge selection events
        self._connect_edge_selection_events()

        # connect the node selection events
        self._connect_node_selection_events()

    def _connect_edge_selection_events(self) -> None:
        """Connect the events for handling edge selections in the main canvas.

        These all interface with the SelectionManager's edge selection.
        """
        # the widget containing all the selection GUI
        selection_widget = self._main_window.app_controls.widget().selection_box

        # events for synchronizing the edge selection enabled state with the GUI.
        selection_widget.edge_mode_box.enable_checkbox.stateChanged.connect(
            self.data.selection._on_edge_enabled_update
        )

        # event for synchronizing the edge selection values with the viewer.
        self.data.selection.edge.events.values.connect(self._on_edge_selection_change)

        # event for attaching/detaching the edge selection callback.
        self.data.selection.edge.events.enabled.connect(
            self._on_edge_selection_enabled_changed
        )

        # event for updating the edge selection GUI when the selection changes.
        self.data.selection.edge.events.values.connect(
            selection_widget._on_edge_selection_change
        )

        # event for pasting the edge selection
        self.data.selection.events.edge.connect(selection_widget._on_edge_paste_request)

    def _connect_node_selection_events(self) -> None:
        """Connect the events for handling node selections in the main canvas.

        These all interface with the SelectionManager's node selection.
        """
        # the widget containing all the selection GUI
        selection_widget = self._main_window.app_controls.widget().selection_box

        # events for synchronizing the edge selection enabled state with the GUI.
        selection_widget.node_mode_box.enable_checkbox.stateChanged.connect(
            self.data.selection._on_node_enabled_update
        )

        # event for synchronizing the edge selection values with the viewer.
        self.data.selection.node.events.values.connect(self._on_node_selection_change)

        # event for attaching/detaching the edge selection callback.
        self.data.selection.node.events.enabled.connect(
            self._on_node_selection_enabled_changed
        )

        # event for updating the edge selection GUI when the selection changes.
        self.data.selection.node.events.values.connect(
            selection_widget._on_node_selection_change
        )

        # event for pasting the node selection
        self.data.selection.events.node.connect(selection_widget._on_node_paste_request)

    def _on_edge_selection_enabled_changed(self, enabled: bool):
        """Callback to update the viewer when the edge selection is enabled/disabled.

        This attached/detaches the edge selection callback from the main viewer.
        """
        if enabled:
            # attach the selection callback
            self._viewer.main_canvas.add_skeleton_edge_callback(
                callback=self.data._on_edge_selection_click,
            )

        else:
            # detach the selection callback
            self._viewer.main_canvas.remove_skeleton_edge_callback(
                callback=self.data._on_edge_selection_click,
            )

    def _on_node_selection_changed(self, enabled: bool):
        """Callback to update the viewer when the node selection is enabled/disabled.

        This attached/detaches the node selection callback from the main viewer.
        """
        if enabled:
            # attach the selection callback
            self._viewer.main_canvas.add_skeleton_node_callback(
                callback=self.data._on_node_selection_click,
            )

        else:
            # detach the selection callback
            self._viewer.main_canvas.remove_skeleton_node_callback(
                callback=self.data._on_node_selection_click,
            )

    def _on_node_selection_enabled_changed(self, enabled: bool):
        """Callback to update the viewer when the node selection is enabled/disabled.

        This attached/detaches the node selection callback from the main viewer.
        """
        if enabled:
            # attach the selection callback
            self._viewer.main_canvas.add_skeleton_node_callback(
                callback=self.data._on_node_selection_click,
            )

        else:
            # detach the selection callback
            self._viewer.main_canvas.remove_skeleton_node_callback(
                callback=self.data._on_node_selection_click,
            )

    def add_points(
        self, point_coordinates: np.ndarray | None = None, point_size = 50
    ) -> tuple[PointsVisual, PointsMemoryStore]:
        """Add points to the viewer.

        Parameters
        ----------
        point_coordinates : np.ndarray, optional
            The coordinates of the points to add, by default np.array([0,0,0],
                                                                dtype=np.float32)
        point_size : int, optional
            The size of the points to add, by default 50

        Returns
        -------
        tuple[PointsVisual, PointsMemoryStore]
            A tuple containing the PointsVisual and PointsMemoryStore objects.
        """
        # create a list of points to add
        # note these must be Float32
        if point_coordinates is None or len(point_coordinates) == 0:
            point_coordinates = np.array([[0, 0, 0]], dtype=np.float32)
        else:
            point_coordinates = np.asarray(point_coordinates, dtype=np.float32)

        # make the data store for the points
        new_points_store = PointsMemoryStore(
            coordinates=point_coordinates,
        )

        # set up the points appearance
        points_appearance = PointsUniformAppearance(
            size=point_size, color=(0, 1, 0, 1), size_coordinate_space="data"
        )

        # make the highlight points model
        points_visual = PointsVisual(
            name="node_highlight_points",
            data_store_id=new_points_store.id,
            appearance=points_appearance,
        )

        # add the data and visual to the viewer backend (cellier)
        self._viewer._backend.add_data_store(data_store=new_points_store)
        self._viewer._backend.add_visual(
            visual_model=points_visual, scene_id=self._viewer._main_canvas.scene_id
        )

        # reslice the viewer to update the display
        self._viewer._backend.reslice_all()
        return points_visual, new_points_store
