"""Classes for interfacing with the viewer."""

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID, uuid4

import numpy as np
import pygfx as gfx
from cellier.controller import CellierController
from cellier.data.label import LabelMemoryStore
from cellier.data.lines import LinesMemoryStore
from cellier.data.points import PointsMemoryStore
from cellier.events import LinesPickEvent, PointsPickEvent
from cellier.scene import BackgroundAppearance
from cellier.transform import AffineTransform
from cellier.visuals import (
    InMemoryLabelsAppearance,
    LabelMemoryVisual,
    LinesMemoryAppearance,
    LinesVisual,
    PointsMarkerAppearance,
    PointsVisual,
)

from skeleplex.app._constants import THEMES, ViewerTheme
from skeleplex.app.cellier.utils import make_controller

#: Default opacity for the rendered segmentation labels. Rendered semi-transparent
#: so the skeleton lines drawn underneath remain visible. Kept in sync with the
#: initial value of the opacity slider in the Qt controls.
DEFAULT_SEGMENTATION_OPACITY: float = 0.3

#: Minimum thickness of the skeleton edges in screen pixels.
MIN_EDGE_THICKNESS: float = 3.0

#: Minimum thickness of the highlighted skeleton edges in screen pixels.
MIN_EDGE_HIGHLIGHT_THICKNESS: float = 6.0


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
    edges_min_visual: LinesVisual | None = None
    edge_highlight_store: LinesMemoryStore | None = None
    edge_highlight_visual: LinesVisual | None = None
    edge_highlight_min_visual: LinesVisual | None = None

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
                self.edges_min_visual is not None,
                self.edge_highlight_store is not None,
                self.edge_highlight_visual is not None,
                self.edge_highlight_min_visual is not None,
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

        self._theme = THEMES["light"]

        # this will store the rendered skeleton components
        self._skeleton = RenderedSkeletonComponents()
        self._segmentation = RenderedSegmentationComponents()

        # map a user callback -> list of (SubscriptionHandle, closure) pairs.
        # the closure is retained so the strong reference outlives registration.
        # Mouse events only say that something was hit (used for background
        # clicks); pick events say what was hit (used for skeleton clicks).
        self._mouse_handles: dict = {}
        self._pick_handles: dict = {}

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

    def _add_edge_visuals(
        self,
        store: LinesMemoryStore,
        name: str,
        thickness: float,
        min_thickness: float,
        **appearance,
    ) -> tuple[LinesVisual, LinesVisual]:
        """Add the two visuals that draw a set of edges with a minimum thickness.

        The materials have no minimum thickness, so the same data are drawn in
        world units and in screen pixels. The wider of the two shows, at every
        point along the edge. This is exact under perspective, unlike updating
        one thickness when the camera moves.

        Parameters
        ----------
        store : LinesMemoryStore
            The data store for the edges, shared by the two visuals.
        name : str
            The name of the visual. The other one gets the suffix "_min".
        thickness : float
            The thickness in world units.
        min_thickness : float
            The minimum thickness in screen pixels.
        **appearance
            The other fields of the appearance of both visuals.

        Returns
        -------
        tuple[LinesVisual, LinesVisual]
            The visual in world units and the visual in screen pixels.
        """
        world_visual = self._backend.add_lines(
            data=store,
            scene_id=self._scene_id,
            appearance=LinesMemoryAppearance(
                thickness=thickness, thickness_space="world", **appearance
            ),
            name=name,
        )
        screen_visual = self._backend.add_lines(
            data=store,
            scene_id=self._scene_id,
            appearance=LinesMemoryAppearance(
                thickness=min_thickness, thickness_space="screen", **appearance
            ),
            name=f"{name}_min",
        )
        return world_visual, screen_visual

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
            (
                self._skeleton.edge_highlight_visual,
                self._skeleton.edge_highlight_min_visual,
            ) = self._add_edge_visuals(
                self._skeleton.edge_highlight_store,
                name="edge_highlight",
                thickness=sizes["edge_highlight"],
                min_thickness=MIN_EDGE_HIGHLIGHT_THICKNESS,
                color=(1, 0, 1, 1),
                color_mode="uniform",
            )

        # update the lines store
        if self._skeleton.edges_store is None:
            self._skeleton.edges_store = LinesMemoryStore(
                positions=edge_coordinates.astype(np.float32), colors=edge_colors
            )
            (
                self._skeleton.edges_visual,
                self._skeleton.edges_min_visual,
            ) = self._add_edge_visuals(
                self._skeleton.edges_store,
                name="edge_lines",
                thickness=sizes["edge"],
                min_thickness=MIN_EDGE_THICKNESS,
                color_mode="vertex",
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
                    color=self._theme.node_color,
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
                appearance=InMemoryLabelsAppearance(
                    opacity=DEFAULT_SEGMENTATION_OPACITY,
                    transparency_mode="weighted_blend",
                    depth_write=False,
                    render_order=2,
                ),
                name="labels_node",
                # Disable picking for the segmentation so the user can click
                # through it to select the skeleton nodes/edges rendered
                # underneath. With pick_write off the labels material is skipped
                # in pygfx's pick pass, so the pick buffer reports the nearest
                # pick-writing visual behind it (a node/edge) instead of the
                # segmentation.
                pick_write=False,
            )
        else:
            # the visual may have been hidden by a prior None update; ensure it
            # is visible again now that there is data to show.
            self._backend.set_visual_visible(self._segmentation.visual.id, visible=True)

        # transform is an evented field on the live model; assignment fires
        # the psygnal event the controller wired, which reslices the scene.
        # A transform names the spaces it maps between, so it is built after
        # add_labels has given the store its coordinate system.
        store_coordinates = self._segmentation.data_store.data_coordinate_systems[0]
        scene = self._backend.get_scene(self._scene_id)
        self._segmentation.visual.transform = AffineTransform.from_matrix(
            np.eye(4) if transform is None else transform,
            input_coordinate_system=store_coordinates,
            output_coordinate_system=scene.dims.world_coordinate_system,
        )

        # Future: add_multiscale_segmentation(self, store, ...)
        #   store  = <multiscale label store>(...)              # larger-than-memory
        #   visual = self._backend.add_labels_multiscale(
        #       data=store, scene_id=self._scene_id,
        #       appearance=MultiscaleLabelsAppearance(),
        #   )
        # The two RenderedSegmentationComponents fields hold the multiscale
        # store/visual unchanged; only the constructor call differs.

        # reslice the scene
        self._backend.reslice_scene(self._scene_id)

    def set_theme(self, theme: ViewerTheme) -> None:
        """Set the background and node colors of the scene.

        Parameters
        ----------
        theme : ViewerTheme
            The theme to apply.
        """
        self._theme = theme
        self._backend.set_background(
            self._scene_id,
            BackgroundAppearance(mode="uniform", color=theme.background),
        )
        if self._skeleton.node_visual is not None:
            self._skeleton.node_visual.appearance.color = theme.node_color

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
            # cellier creates the 3D camera with a fixed (1, 8000) depth range,
            # which clips skeletons larger than that. Scale the clipping planes
            # with the scene before fitting, so the fitted state includes them.
            self._fit_depth_range_to_scene()
            self._backend.look_at_visual(
                visual_id=self._skeleton.node_visual.id,
                canvas_id=self._canvas_id,
                # the vectors are in pygfx (x, y, z) order; cellier names the axes.
                view_direction=dict(zip("xyz", view_direction, strict=True)),
                up=dict(zip("xyz", up, strict=True)),
            )
            # one-shot: drop this subscription so later reslices don't refit.
            self._backend.unsubscribe_owner(fit_owner)

        self._backend.on_reslice_completed(
            self._skeleton.node_visual.id, _fit_when_ready, owner_id=fit_owner
        )
        # trigger the slice cycle that uploads positions and fires the callback.
        self._backend.reslice_scene(self._scene_id)

    def _fit_depth_range_to_scene(self) -> None:
        """Scale the camera clipping planes with the size of the scene.

        The near/far ratio is kept constant so depth precision is the same
        regardless of the scale of the data. cellier has no public hook for
        the scene bounds, so it reaches into the private render manager.
        """
        gfx_scene = self._backend._render_manager.get_scene(self._scene_id)
        bounding_sphere = gfx_scene.get_world_bounding_sphere()
        if bounding_sphere is None:
            return
        radius = float(bounding_sphere[3])
        if not np.isfinite(radius) or radius <= 0:
            return
        self._backend.set_camera_depth_range(
            self._canvas_id, (radius * 1e-4, radius * 20)
        )

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

        def _dispatch_edge_pick(event):
            if event.action != "press":
                return
            if event.visual_id in (
                self._skeleton.edges_visual.id,
                self._skeleton.edges_min_visual.id,
            ):
                callback(event, click_source="data")
            elif event.visual_id in (
                self._skeleton.edge_highlight_visual.id,
                self._skeleton.edge_highlight_min_visual.id,
            ):
                callback(event, click_source="highlight")

        self._add_click_callback(callback, LinesPickEvent, _dispatch_edge_pick)

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

        def _dispatch_node_pick(event):
            if event.action != "press":
                return
            if event.visual_id == self._skeleton.node_visual.id:
                callback(event, click_source="data")
            elif event.visual_id == self._skeleton.node_highlight_visual.id:
                callback(event, click_source="highlight")

        self._add_click_callback(callback, PointsPickEvent, _dispatch_node_pick)

    def _add_click_callback(
        self, callback: Callable, pick_event_type: type, dispatch_pick: Callable
    ) -> None:
        """Route presses on the canvas to a skeleton click callback.

        Parameters
        ----------
        callback : Callable
            The callback function, invoked as ``callback(event, click_source=...)``.
        pick_event_type : type
            The cellier pick event for the visuals the callback owns.
        dispatch_pick : Callable
            Called with each pick event of ``pick_event_type``. It calls
            ``callback`` with click source "data" or "highlight".
        """

        def _dispatch_background_press(event):
            if event.pick_info.hit_visual_id not in (
                self._skeleton.edges_visual.id,
                self._skeleton.edges_min_visual.id,
                self._skeleton.edge_highlight_visual.id,
                self._skeleton.edge_highlight_min_visual.id,
                self._skeleton.node_visual.id,
                self._skeleton.node_highlight_visual.id,
            ):
                # empty space or a non-skeleton visual (e.g. segmentation)
                callback(event, click_source="background")

        mouse_handle = self._backend.on_mouse_press_3d(
            self._canvas_id, _dispatch_background_press, owner_id=self._owner_id
        )
        self._mouse_handles.setdefault(callback, []).append(
            (mouse_handle, _dispatch_background_press)
        )
        pick_handle = self._backend.on_pick(
            self._canvas_id, pick_event_type, dispatch_pick, owner_id=self._owner_id
        )
        self._pick_handles.setdefault(callback, []).append((pick_handle, dispatch_pick))

    def _remove_click_callback(self, callback: Callable) -> None:
        """Remove the subscriptions made by ``_add_click_callback``."""
        for handle, _closure in self._mouse_handles.pop(callback, []):
            self._backend.unsubscribe_mouse(handle)
        for handle, _closure in self._pick_handles.pop(callback, []):
            self._backend.unsubscribe_pick(handle)

    def remove_skeleton_edge_callback(self, callback: Callable):
        """Remove a previously registered edge callback.

        Parameters
        ----------
        callback : Callable
            The callback function passed to ``add_skeleton_edge_callback``.
        """
        self._remove_click_callback(callback)

    def remove_skeleton_node_callback(self, callback: Callable):
        """Remove a previously registered node callback.

        Parameters
        ----------
        callback : Callable
            The callback function passed to ``add_skeleton_node_callback``.
        """
        self._remove_click_callback(callback)


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
        self._use_trackball_controller()
        self._main_canvas = MainCanvasController(
            scene_id=self._scene_id,
            canvas_id=self._canvas_id,
            backend=self._backend,
        )
        return widget

    def _use_trackball_controller(self) -> None:
        """Replace the canvas's 3D orbit controller with a trackball controller.

        cellier hardcodes pygfx's ``OrbitController``, which clamps the
        elevation to +/-89 degrees around the camera's up vector, so the
        skeleton cannot be flipped over the pole. The trackball controller has
        the same mouse bindings but rotates freely. cellier has no public hook
        for this, so it reaches into the private ``CanvasView`` attributes.
        """
        canvas_view = self._backend._render_manager._canvases[self._canvas_id]
        orbit = canvas_view._controller_3d
        trackball = gfx.TrackballController(
            camera=canvas_view._camera_3d,
            enabled=orbit.enabled,
            # the canvas view ticks its controllers and applies the camera state
            auto_update=False,
            register_events=canvas_view._renderer,
        )
        # the orbit controller's event handler cannot be unregistered, so
        # leave it permanently disabled.
        orbit.enabled = False
        if canvas_view._controller is orbit:
            canvas_view._controller = trackball
        canvas_view._controller_3d = trackball
