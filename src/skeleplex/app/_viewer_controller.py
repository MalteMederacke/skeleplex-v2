"""Classes for interfacing with the viewer."""

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID, uuid4

import numpy as np
from cellier.controller import CellierController
from cellier.data.label import LabelMemoryStore
from cellier.data.lines import LinesMemoryStore
from cellier.data.points import PointsMemoryStore
from cellier.transform import AffineTransform
from cellier.visuals import (
    InMemoryLabelsAppearance,
    LabelMemoryVisual,
    LinesMemoryAppearance,
    LinesVisual,
    PointsMarkerAppearance,
    PointsVisual,
)

from skeleplex.app.cellier.utils import make_controller

#: Default opacity for the rendered segmentation labels. Rendered semi-transparent
#: so the skeleton lines drawn underneath remain visible. Kept in sync with the
#: initial value of the opacity slider in the Qt controls.
DEFAULT_SEGMENTATION_OPACITY: float = 0.3


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
    """A class for storing the components for a rendered segmentation.

    ``data_store`` / ``visual`` are typed for the in-memory label path today.
    The multiscale path (a ``MultiscaleLabelVisual`` + a multiscale label store)
    slots in behind the same two fields via a future
    ``add_multiscale_segmentation`` method; only the constructor call differs.
    """

    data_store: LabelMemoryStore | None = None
    visual: LabelMemoryVisual | None = None


class MainCanvasController:
    """A class for controlling the main canvas."""

    def __init__(self, scene_id: UUID, canvas_id: UUID, backend: CellierController):
        self._scene_id = scene_id
        self._canvas_id = canvas_id
        self._backend = backend

        # owns this canvas's mouse subscriptions for bulk teardown
        self._owner_id = uuid4()

        # this will store the rendered skeleton components
        self._skeleton = RenderedSkeletonComponents()
        self._segmentation = RenderedSegmentationComponents()

        # maps a user callback -> list of (SubscriptionHandle, closure) pairs.
        # the closure is retained so the strong reference outlives registration.
        self._mouse_handles: dict = {}

    @property
    def scene_id(self) -> UUID:
        """Get the scene ID of the main canvas (a UUID, was a str in v1)."""
        return self._scene_id

    @property
    def canvas_id(self) -> UUID:
        """Get the ID of the underlying cellier canvas."""
        return self._canvas_id

    def _compute_sizes(self, node_coordinates: np.ndarray) -> dict:
        """Compute display sizes scaled to the node coordinate span."""
        if len(node_coordinates) > 1:
            span = float(np.ptp(node_coordinates, axis=0).max())
        else:
            span = 100.0
        node_size = max(8.0, span * 0.003)
        return {
            "node": node_size,
            "node_highlight": node_size * 2.5,
            "edge": max(2.0, span * 0.002),
            "edge_highlight": max(6.0, span * 0.006),
        }

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
            the edges of the skeleton to be rendered. Stored in array (z, y, x)
            order; cellier reverses to pygfx (x, y, z) internally.
        edge_colors : np.ndarray
            (n_edges * 2 * n_segments_per_edge, 4 ) RGBA array of colors
            of the edges of the skeleton to be rendered.
        node_coordinates : np.ndarray
            (n_nodes, 3) array of coordinates of the nodes of the skeleton
            to be rendered. Stored in array (z, y, x) order.
        """
        sizes = self._compute_sizes(node_coordinates)

        # make the highlight lines store
        if self._skeleton.edge_highlight_store is None:
            # if the highlight store is not populated, create it
            self._skeleton.edge_highlight_store = LinesMemoryStore(
                positions=np.empty((0, 3), dtype=np.float32)
            )

        if self._skeleton.edge_highlight_visual is None:
            # if the highlight visual is not populated, create it
            self._skeleton.edge_highlight_visual = self._backend.add_lines(
                data=self._skeleton.edge_highlight_store,
                scene_id=self._scene_id,
                appearance=LinesMemoryAppearance(
                    color=(1, 0, 1, 1),
                    thickness=sizes["edge_highlight"],
                    thickness_space="world",
                    color_mode="uniform",
                ),
                name="edge_highlight",
            )

        # update the lines store
        if self._skeleton.edges_store is None:
            self._skeleton.edges_store = LinesMemoryStore(
                positions=edge_coordinates.astype(np.float32), colors=edge_colors
            )
            self._skeleton.edges_visual = self._backend.add_lines(
                data=self._skeleton.edges_store,
                scene_id=self._scene_id,
                appearance=LinesMemoryAppearance(
                    thickness=sizes["edge"],
                    thickness_space="world",
                    color_mode="vertex",
                ),
                name="edge_lines",
            )
        else:
            self._skeleton.edges_store.positions = edge_coordinates.astype(np.float32)
            self._skeleton.edges_store.colors = edge_colors.astype(np.float32)

        # make the highlight points store
        if self._skeleton.node_highlight_store is None:
            self._skeleton.node_highlight_store = PointsMemoryStore(
                positions=np.empty((0, 3), dtype=np.float32)
            )

        if self._skeleton.node_highlight_visual is None:
            self._skeleton.node_highlight_visual = self._backend.add_points(
                data=self._skeleton.node_highlight_store,
                scene_id=self._scene_id,
                appearance=PointsMarkerAppearance(
                    size=sizes["node_highlight"],
                    color=(0, 1, 0, 1),
                    size_space="world",
                    color_mode="uniform",
                ),
                name="node_highlight_points",
            )

        # update the points store
        if self._skeleton.node_store is None:
            self._skeleton.node_store = PointsMemoryStore(
                positions=node_coordinates.astype(np.float32)
            )
            self._skeleton.node_visual = self._backend.add_points(
                data=self._skeleton.node_store,
                scene_id=self._scene_id,
                appearance=PointsMarkerAppearance(
                    size=sizes["node"],
                    color=(0, 0, 0, 0.8),
                    size_space="world",
                    color_mode="uniform",
                ),
                name="node_points",
            )
        else:
            # update the points store with the new coordinates
            self._skeleton.node_store.positions = node_coordinates.astype(np.float32)
            # update appearance sizes in case the dataset changed
            self._skeleton.node_visual.appearance.size = sizes["node"]
        if self._skeleton.node_highlight_visual is not None:
            self._skeleton.node_highlight_visual.appearance.size = sizes[
                "node_highlight"
            ]
        if self._skeleton.edges_visual is not None:
            self._skeleton.edges_visual.appearance.thickness = sizes["edge"]
        if self._skeleton.edge_highlight_visual is not None:
            self._skeleton.edge_highlight_visual.appearance.thickness = sizes[
                "edge_highlight"
            ]

        # reslice the scene
        self._backend.reslice_scene(self._scene_id)

    def update_segmentation_image(
        self,
        image: np.ndarray | None,
        transform: np.ndarray | None = None,
    ):
        """Update the segmentation image in the viewer.

        Parameters
        ----------
        image : np.ndarray | None
            The segmentation image to render, in array (z, y, x) order.
            If None, the segmentation is hidden (e.g. "None" view mode). An
            already-rendered visual is hidden rather than removed so it can be
            shown again on the next non-None update.
        transform : np.ndarray | None
            The 4x4 affine transform to apply to the segmentation image, in
            data (z, y, x) axis order. cellier reverses to pygfx order
            internally.
        """
        if image is None:
            # nothing has ever been rendered, so there is nothing to hide
            if self._segmentation.visual is None:
                return

            # hide the existing visual and reslice so the change is shown
            self._backend.set_visual_visible(
                self._segmentation.visual.id, visible=False
            )
            self._backend.reslice_scene(self._scene_id)
            return

        # make / update the segmentation data store (int32 required by cellier)
        if self._segmentation.data_store is None:
            self._segmentation.data_store = LabelMemoryStore(
                data=image.astype(np.int32), name="label_image"
            )
        else:
            self._segmentation.data_store.data = image.astype(np.int32)

        # make the transform into a cellier AffineTransform
        if transform is None:
            resolved_transform = AffineTransform(matrix=np.eye(4))
        else:
            resolved_transform = AffineTransform(matrix=transform)

        if self._segmentation.visual is None:
            self._segmentation.visual = self._backend.add_labels(
                data=self._segmentation.data_store,
                scene_id=self._scene_id,
                # Render the labels semi-transparently so the skeleton lines
                # drawn underneath remain visible. weighted_blend is
                # order-independent (OIT) and depth_write is disabled so the
                # translucent surface does not occlude the lines behind it.
                # render_order is raised above the skeleton visuals (default 0)
                # so the translucent labels always composite last, instead of
                # being distance-sorted against the other visuals — this avoids
                # the composite-order "pop" when orbiting the camera.
                #
                # opacity is intentionally left at the default (1.0) here and
                # applied via update_appearance_field below — see the workaround
                # note for why.
                appearance=InMemoryLabelsAppearance(
                    transparency_mode="weighted_blend",
                    depth_write=False,
                    render_order=2,
                ),
                name="labels_node",
                transform=resolved_transform,
            )
            # Workaround for a cellier bug: the 3D label volume material is
            # constructed without applying appearance.opacity (only the 2D image
            # material honors it at construction), so the volume renders fully
            # opaque until an opacity appearance event fires. The appearance is
            # an evented model that suppresses no-op sets, so we must set a value
            # that differs from the constructed one to actually fire the event —
            # hence constructing at the default 1.0 above and changing it here.
            # Remove once cellier applies opacity at construction.
            self._backend.update_appearance_field(
                self._segmentation.visual.id, "opacity", DEFAULT_SEGMENTATION_OPACITY
            )
        else:
            # the visual may have been hidden by a prior None update; ensure it
            # is visible again now that there is data to show.
            self._backend.set_visual_visible(self._segmentation.visual.id, visible=True)

            # transform is an evented field on the live model; assignment fires
            # the psygnal event the controller wired, which reslices the scene.
            self._segmentation.visual.transform = resolved_transform

        # Future: add_multiscale_segmentation(self, levels, level_transforms, ...)
        #   store  = <multiscale label store>(...)              # larger-than-memory
        #   visual = self._backend.add_labels_multiscale(
        #       data=store, scene_id=self._scene_id,
        #       appearance=MultiscaleLabelsAppearance(),
        #       level_transforms=level_transforms,              # list[AffineTransform]
        #   )
        # The two RenderedSegmentationComponents fields hold the multiscale
        # store/visual unchanged; only the constructor call differs.

        # reslice the scene
        self._backend.reslice_scene(self._scene_id)

    def set_segmentation_opacity(self, opacity: float) -> None:
        """Set the opacity of the rendered segmentation labels.

        Parameters
        ----------
        opacity : float
            The opacity in the range [0, 1]. 0 is fully transparent, 1 is
            fully opaque. Has no effect if no segmentation has been rendered.
        """
        if self._segmentation.visual is None:
            return
        self._backend.update_appearance_field(
            self._segmentation.visual.id, "opacity", float(opacity)
        )

    def look_at_skeleton(
        self,
        view_direction: tuple[int, int, int] = (0, 0, 1),
        up: tuple[int, int, int] = (0, 1, 0),
    ):
        """Adjust the camera to look at the skeleton.

        Geometry is uploaded to pygfx asynchronously by the slicer, so the
        world bounding box is not available synchronously after a reslice. Fit
        the camera from a one-shot ``on_reslice_completed`` callback (fired once
        the node visual's positions are committed), then trigger a reslice.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        fit_owner = uuid4()

        def _fit_when_ready(event=None) -> None:
            self._backend.look_at_visual(
                visual_id=self._skeleton.node_visual.id,
                canvas_id=self._canvas_id,
                view_direction=view_direction,
                up=up,
            )
            # one-shot: drop this subscription so later reslices don't refit.
            self._backend.unsubscribe_owner(fit_owner)

        self._backend.on_reslice_completed(
            self._skeleton.node_visual.id, _fit_when_ready, owner_id=fit_owner
        )
        # trigger the slice cycle that uploads positions and fires the callback.
        self._backend.reslice_scene(self._scene_id)

    def set_edge_highlight(
        self,
        edge_coordinates: np.ndarray,
    ):
        """Set the edge highlight coordinates.

        Parameters
        ----------
        edge_coordinates : np.ndarray
            The coordinates of the edge to highlight, in array (z, y, x) order.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        self._skeleton.edge_highlight_store.positions = edge_coordinates.astype(
            np.float32
        )
        self._backend.reslice_scene(self._scene_id)

    def set_node_highlight(
        self,
        node_coordinates: np.ndarray,
    ) -> None:
        """Set the node highlight coordinates.

        Parameters
        ----------
        node_coordinates : np.ndarray
            The coordinates of the node to highlight, in array (z, y, x) order.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        self._skeleton.node_highlight_store.positions = node_coordinates.astype(
            np.float32
        )
        self._backend.reslice_scene(self._scene_id)

    def add_skeleton_edge_callback(
        self,
        callback: Callable,
    ):
        """Add a callback for edge presses on the main (3D) canvas.

        Parameters
        ----------
        callback : Callable
            The callback function, invoked as ``callback(event, click_source=...)``.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        def _dispatch_edge_press(event):
            hit = event.pick_info.hit_visual_id
            if hit == self._skeleton.edges_visual.id:
                callback(event, click_source="data")
            elif hit == self._skeleton.edge_highlight_visual.id:
                callback(event, click_source="highlight")
            elif hit in (
                self._skeleton.node_visual.id,
                self._skeleton.node_highlight_visual.id,
            ):
                # a skeleton node was hit; the node callback owns it, not us
                return
            else:
                # empty space or a non-skeleton visual (e.g. segmentation)
                callback(event, click_source="background")

        handle = self._backend.on_mouse_press_3d(
            self._canvas_id, _dispatch_edge_press, owner_id=self._owner_id
        )
        self._mouse_handles.setdefault(callback, []).append(
            (handle, _dispatch_edge_press)
        )

    def add_skeleton_node_callback(self, callback: Callable):
        """Add a callback for node presses on the main (3D) canvas.

        Parameters
        ----------
        callback : Callable
            The callback function, invoked as ``callback(event, click_source=...)``.
        """
        if not self._skeleton.populated():
            # don't do anything if the skeleton is not rendered
            return

        def _dispatch_node_press(event):
            hit = event.pick_info.hit_visual_id
            if hit == self._skeleton.node_visual.id:
                callback(event, click_source="data")
            elif hit == self._skeleton.node_highlight_visual.id:
                callback(event, click_source="highlight")
            elif hit in (
                self._skeleton.edges_visual.id,
                self._skeleton.edge_highlight_visual.id,
            ):
                # a skeleton edge was hit; the edge callback owns it, not us
                return
            else:
                # empty space or a non-skeleton visual (e.g. segmentation)
                callback(event, click_source="background")

        handle = self._backend.on_mouse_press_3d(
            self._canvas_id, _dispatch_node_press, owner_id=self._owner_id
        )
        self._mouse_handles.setdefault(callback, []).append(
            (handle, _dispatch_node_press)
        )

    def remove_skeleton_edge_callback(self, callback: Callable):
        """Remove a previously registered edge callback.

        Parameters
        ----------
        callback : Callable
            The callback function passed to ``add_skeleton_edge_callback``.
        """
        for handle, _closure in self._mouse_handles.pop(callback, []):
            self._backend.unsubscribe_mouse(handle)

    def remove_skeleton_node_callback(self, callback: Callable):
        """Remove a previously registered node callback.

        Parameters
        ----------
        callback : Callable
            The callback function passed to ``add_skeleton_node_callback``.
        """
        for handle, _closure in self._mouse_handles.pop(callback, []):
            self._backend.unsubscribe_mouse(handle)


class ViewerController:
    """A class for controlling the viewer backend."""

    def __init__(self):
        self._backend, self._scene_id = make_controller()
        self._canvas_id: UUID | None = None
        self._main_canvas: MainCanvasController | None = None

    @property
    def main_canvas(self) -> MainCanvasController:
        """Get the controller for the main canvas."""
        if self._main_canvas is None:
            raise RuntimeError(
                "create_main_canvas() must be called before accessing main_canvas."
            )
        return self._main_canvas

    def create_main_canvas(self, parent):
        """Create the render canvas parented to *parent* and return its widget.

        Must be called after the parent widget exists. v2's ``add_canvas``
        constructs the render widget at call time and needs ``widget_parent``
        set first, so canvas creation is deferred until the parent is known.
        """
        self._backend.set_widget_parent(parent)
        widget = self._backend.add_canvas(self._scene_id)
        # add_canvas returns the widget; recover the canvas UUID from the scene.
        self._canvas_id = self._backend.get_canvas_ids(self._scene_id)[-1]
        self._main_canvas = MainCanvasController(
            scene_id=self._scene_id,
            canvas_id=self._canvas_id,
            backend=self._backend,
        )
        return widget
