"""Classes for interfacing with the viewer."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
from cellier.models.data_stores.image import ImageMemoryStore
from cellier.models.data_stores.lines import LinesMemoryStore
from cellier.models.data_stores.points import PointsMemoryStore
from cellier.models.visuals import (
    LabelsAppearance,
    LinesUniformAppearance,
    LinesVertexColorAppearance,
    LinesVisual,
    MultiscaleImageVisual,
    MultiscaleLabelsVisual,
    PointsUniformAppearance,
    PointsVisual,
)
from cellier.transform import AffineTransform
from cellier.viewer_controller import CellierController

from skeleplex.app.cellier.utils import make_viewer_controller, make_viewer_model


@dataclass
class RenderedSkeletonComponents:
    """A class for storing the components for a rendered skeleton.

    These data are used for accessing the rendered skeleton in the viewer backend.
    """

    node_store: PointsMemoryStore | None = None
    node_visual: PointsVisual | None = None
    node_highlight_store: PointsMemoryStore | None = None
    node_highlight_visual: PointsVisual | None = None
    edges_store: LinesMemoryStore | None = None
    edges_visual: LinesVisual | None = None
    edge_highlight_store: LinesMemoryStore | None = None
    edge_highlight_visual: LinesVisual | None = None

    def populated(self) -> bool:
        """Returns True if all the components are populated."""
        return all(
            [
                self.node_store is not None,
                self.node_visual is not None,
                self.node_highlight_store is not None,
                self.node_highlight_visual is not None,
                self.edges_store is not None,
                self.edges_visual is not None,
                self.edge_highlight_store is not None,
                self.edge_highlight_visual is not None,
            ]
        )


@dataclass
class RenderedSegmentationComponents:
    """A class for storing the components for a rendered image."""

    data_store: ImageMemoryStore | None = None
    visual: MultiscaleImageVisual | None = None


class MainCanvasController:
    """A class for controlling the main canvas."""

    def __init__(self, scene_id: str, backend: CellierController):
        self._scene_id = scene_id
        self._backend = backend

        # this will store the rendered skeleton components
        self._skeleton = RenderedSkeletonComponents()
        self._segmentation = RenderedSegmentationComponents()

    @property
    def scene_id(self) -> str:
        """Get the scene ID of the main canvas."""
        return self._scene_id

    def update_skeleton_geometry(
        self,
        edge_coordinates: np.ndarray,
        edge_colors: np.ndarray,
        node_coordinates: np.ndarray,
    ):
        """Update the geometry of the skeleton in the viewer.

        Parameters
        ----------
        edge_coordinates : np.ndarray
            (n_edges * 2 * n_segments_per_edge) array of coordinates of
            the edges of the skeleton to be rendered.
        edge_colors : np.ndarray
            (n_edges * 2 * n_segments_per_edge, 4 ) RGBA array of colors
            of the edges of the skeleton to be rendered.
        node_coordinates : np.ndarray
            (n_nodes, 3) array of coordinates of the nodes of the skeleton
            to be rendered.
        """
        # make the highlight lines store
        if self._skeleton.edge_highlight_store is None:
            # if the highlight store is not populated, create it
            self._skeleton.edge_highlight_store = LinesMemoryStore(
                coordinates=np.empty((0, 3))
            )
            self._backend.add_data_store(data_store=self._skeleton.edge_highlight_store)

        if self._skeleton.edge_highlight_visual is None:
            # if the highlight visual is not populated, create it
            edge_highlight_material_3d = LinesUniformAppearance(
                color=(1, 0, 1, 1), size=6, size_coordinate_space="data", opacity=1.0
            )

            # make the highlight lines model
            edge_highlight_visual = LinesVisual(
                name="edge_highlight",
                data_store_id=self._skeleton.edge_highlight_store.id,
                appearance=edge_highlight_material_3d,
                pick_write=True,
            )
            self._skeleton.edge_highlight_visual = edge_highlight_visual

            # add the visual model to the viewer
            self._backend.add_visual(
                visual_model=edge_highlight_visual, scene_id=self.scene_id
            )

        # update the lines store
        if self._skeleton.edges_store is None:
            self._skeleton.edges_store = LinesMemoryStore(
                coordinates=edge_coordinates[:, [2, 1, 0]], colors=edge_colors
            )
            self._backend.add_data_store(data_store=self._skeleton.edges_store)
        else:
            self._skeleton.edges_store.coordinates = edge_coordinates.astype(
                np.float32
            )[:, [2, 1, 0]]
            self._skeleton.edges_store.colors = edge_colors.astype(np.float32)

        if self._skeleton.edges_visual is None:
            # if the lines visual is not populated, create it
            edge_lines_material_3d = LinesVertexColorAppearance(
                size=2, size_coordinate_space="data"
            )

            # make the lines model
            edge_lines_visual = LinesVisual(
                name="edge_lines",
                data_store_id=self._skeleton.edges_store.id,
                appearance=edge_lines_material_3d,
            )
            self._skeleton.edges_visual = edge_lines_visual
            self._backend.add_visual(
                visual_model=edge_lines_visual, scene_id=self.scene_id
            )

        if self._skeleton.node_highlight_store is None:
            # make the highlight points store if it is not already created
            self._skeleton.node_highlight_store = PointsMemoryStore(
                coordinates=np.empty((0, 3), dtype=np.float32)
            )
            self._backend.add_data_store(data_store=self._skeleton.node_highlight_store)

        if self._skeleton.node_highlight_visual is None:
            # make the highlight points material
            highlight_points_material_3d = PointsUniformAppearance(
                size=20, color=(0, 1, 0, 1), size_coordinate_space="data"
            )

            # make the highlight points model
            highlight_points_visual_3d = PointsVisual(
                name="node_highlight_points",
                data_store_id=self._skeleton.node_highlight_store.id,
                appearance=highlight_points_material_3d,
            )
            self._skeleton.node_highlight_visual = highlight_points_visual_3d

            # add the highlights to the viewer
            self._backend.add_visual(
                visual_model=highlight_points_visual_3d, scene_id=self.scene_id
            )

        if self._skeleton.node_store is None:
            # make the points store if it is not already created
            self._skeleton.node_store = PointsMemoryStore(
                coordinates=node_coordinates[:, [2, 1, 0]]
            )
            self._backend.add_data_store(data_store=self._skeleton.node_store)
        else:
            # update the points store with the new coordinates
            self._skeleton.node_store.coordinates = node_coordinates.astype(np.float32)[
                :, [2, 1, 0]
            ]

        if self._skeleton.node_visual is None:
            # make the points material
            points_material_3d = PointsUniformAppearance(
                size=8, color=(0, 0, 0, 1), size_coordinate_space="data"
            )

            # make the points model
            points_visual_3d = PointsVisual(
                name="node_points",
                data_store_id=self._skeleton.node_store.id,
                appearance=points_material_3d,
            )
            self._skeleton.node_visual = points_visual_3d
            self._backend.add_visual(
                visual_model=points_visual_3d, scene_id=self.scene_id
            )

        # reslice the scene
        self._backend.reslice_scene(scene_id=self.scene_id)

    def update_segmentation_image(
        self,
        image: np.ndarray | None,
        transform: np.ndarray | None = None,
    ):
        """Update the segmentation image in the viewer.

        Parameters
        ----------
        image : np.ndarray | None
            The segmentation image to render.
            If None, the segmentation image will be removed.
        transform : np.ndarray | None
            The 4x4 affine transform to apply to the segmentation image.
        """
        if image is None:
            return

        # make the segmentation data store
        if self._segmentation.data_store is None:
            # if the segmentation data store is not populated, create it
            self._segmentation.data_store = ImageMemoryStore(
                data=image, name="label_image"
            )
            self._backend.add_data_store(data_store=self._segmentation.data_store)

        else:
            # update the data store with the new image
            self._segmentation.data_store.data = image.astype(np.uint32)

        # make the transform into a cellier AffineTransform
        if transform is None:
            transform = AffineTransform(matrix=np.eye(4))
        else:
            transform = AffineTransform(matrix=transform)

        if self._segmentation.visual is None:
            # if the segmentation visual is not populated, create it
            labels_material = LabelsAppearance(color_map="glasbey:glasbey")
            labels_visual_model = MultiscaleLabelsVisual(
                name="labels_node",
                data_store_id=self._segmentation.data_store.id,
                appearance=labels_material,
                transform=transform,
                downscale_factors=[1],
            )
            self._segmentation.visual = labels_visual_model

            # add the visual model to the viewer
            self._backend.add_visual(
                visual_model=labels_visual_model, scene_id=self.scene_id
            )

        else:
            self._segmentation.visual.transform = transform

        # reslice the scene
        self._backend.reslice_scene(scene_id=self.scene_id)

    def look_at_skeleton(
        self,
        view_direction: tuple[int, int, int] = (0, 0, 1),
        up: tuple[int, int, int] = (0, 1, 0),
    ):
        """Adjust the camera to look at the skeleton."""
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        self._backend.look_at_visual(
            visual_id=self._skeleton.node_visual.id,
            view_direction=view_direction,
            up_direction=up,
        )

    def set_edge_highlight(
        self,
        edge_coordinates: np.ndarray,
    ):
        """Set the edge highlight coordinates.

        Parameters
        ----------
        edge_coordinates : np.ndarray
            The coordinates of the edge to highlight.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        self._skeleton.edge_highlight_store.coordinates = edge_coordinates.astype(
            np.float32
        )[:, [2, 1, 0]]
        self._backend.reslice_scene(scene_id=self.scene_id)

    def set_node_highlight(
        self,
        node_coordinates: np.ndarray,
    ) -> None:
        """Set the node highlight coordinates.

        Parameters
        ----------
        node_coordinates : np.ndarray
            The coordinates of the node to highlight.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        self._skeleton.node_highlight_store.coordinates = node_coordinates.astype(
            np.float32
        )[:, [2, 1, 0]]
        self._backend.reslice_scene(scene_id=self.scene_id)

    def add_skeleton_edge_callback(
        self,
        callback: Callable,
    ):
        """Add a callback to the skeleton edge visual.

        Parameters
        ----------
        callback : Callable
            The callback function.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        # add for the data visual
        if (
            self._skeleton.edges_visual.id
            not in self._backend.events.mouse.visual_signals
        ):
            # if the visual isn't registered, register it
            self._backend.events.mouse.register_visual(
                visual_id=self._skeleton.edges_visual.id
            )
        self._backend.events.mouse.subscribe_to_visual(
            visual_id=self._skeleton.edges_visual.id,
            callback=partial(callback, click_source="data"),
        )

        # add for the highlight visual
        if (
            self._skeleton.edge_highlight_visual.id
            not in self._backend.events.mouse.visual_signals
        ):
            # if the visual isn't registered, register it
            self._backend.events.mouse.register_visual(
                visual_id=self._skeleton.edge_highlight_visual.id
            )
        self._backend.events.mouse.subscribe_to_visual(
            visual_id=self._skeleton.edge_highlight_visual.id,
            callback=partial(callback, click_source="highlight"),
        )

    def add_skeleton_node_callback(self, callback: Callable):
        """Add a callback to the skeleton node visual.

        Parameters
        ----------
        callback : Callable
            The callback function.
        callback_type : tuple[str, ...]
            The type of callback. See the pygfx documentation for event types.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        # add for the data visual
        if (
            self._skeleton.node_visual.id
            not in self._backend.events.mouse.visual_signals
        ):
            # if the visual isn't registered, register it
            self._backend.events.mouse.register_visual(
                visual_id=self._skeleton.node_visual.id
            )
        self._backend.events.mouse.subscribe_to_visual(
            visual_id=self._skeleton.node_visual.id,
            callback=partial(callback, click_source="data"),
        )

        # add for the highlight visual
        if (
            self._skeleton.node_highlight_visual.id
            not in self._backend.events.mouse.visual_signals
        ):
            # if the visual isn't registered, register it
            self._backend.events.mouse.register_visual(
                visual_id=self._skeleton.node_highlight_visual.id
            )
            self._backend.events.mouse.subscribe_to_visual(
                visual_id=self._skeleton.node_highlight_visual.id,
                callback=partial(callback, click_source="highlight"),
            )

    def remove_skeleton_edge_callback(self, callback: Callable):
        """Remove a callback from the skeleton edge visual.

        Parameters
        ----------
        callback : Callable
            The callback function.
        callback_type : tuple[str, ...]
            The type of callback. See the pygfx documentation for event types.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        self._backend.remove_visual_callback(
            visual_id=self._skeleton.edges_visual.id,
            callback=callback,
        )

    def remove_skeleton_node_callback(self, callback: Callable):
        """Remove a callback from the skeleton node visual.

        Parameters
        ----------
        callback : Callable
            The callback function.
        callback_type : tuple[str, ...]
            The type of callback. See the pygfx documentation for event types.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        self._backend.remove_visual_callback(
            visual_id=self._skeleton.node_visual.id,
            callback=callback,
        )


class ViewerController:
    """A class for controlling the viewer backend."""

    def __init__(self):
        viewer_model, main_canvas_scene_id = make_viewer_model()
        self._backend = make_viewer_controller(viewer_model=viewer_model)

        # make the main canvas controller
        self._main_canvas = MainCanvasController(
            scene_id=main_canvas_scene_id, backend=self._backend
        )

    @property
    def main_canvas(self) -> MainCanvasController:
        """Get the controller for the main canvas."""
        return self._main_canvas

    def _populate_viewer_from_model(self, canvas_widget_parent):
        self._backend.populate_from_viewer_model(
            self._backend._model,
            widget_parent=canvas_widget_parent,
            overwrite_model=True,
        )
