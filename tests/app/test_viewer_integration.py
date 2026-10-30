"""Tests of the viewer through the real cellier render and pick path.

These guard the contract between skeleplex and cellier: run them when bumping
the cellier version. Pointer events enter at the renderer, so a click goes
through the GPU pick buffer and cellier's event routing exactly like a real one.
"""

import asyncio
import os
import time

import networkx as nx
import numpy as np
import pytest
import zarr
from cmap import Color

from skeleplex.app._constants import THEMES
from skeleplex.graph.constants import (
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    NODE_COORDINATE_KEY,
)
from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.graph.spline import B3Spline
from skeleplex.utils import write_ome_zarr_labels
from skeleplex.visualize import EdgeColormap

# larger than cellier's default far clipping plane (8000)
SPAN = 20000.0
NODE_SIZE = 900.0
VOXEL_SIZE = 100.0
TRANSLATION_Z = 1000.0
NODES = {
    0: [SPAN * 0.1, 0.0, SPAN * 0.1],
    1: [SPAN * 0.5, SPAN * 0.5, SPAN * 0.5],
    2: [SPAN * 0.9, SPAN * 0.9, SPAN * 0.2],
    3: [SPAN * 0.2, SPAN * 0.9, SPAN * 0.9],
}

pytestmark = pytest.mark.filterwarnings("ignore")


def _has_gpu() -> bool:
    import wgpu

    try:
        return wgpu.gpu.request_adapter_sync() is not None
    except RuntimeError:
        return False


@pytest.fixture
def data_paths(tmp_path):
    """Write a three edge skeleton and a segmentation with two labels."""
    graph = nx.DiGraph()
    for node, coordinate in NODES.items():
        graph.add_node(node, **{NODE_COORDINATE_KEY: np.array(coordinate)})
    for u, v in [(0, 1), (1, 2), (1, 3)]:
        coordinates = np.linspace(NODES[u], NODES[v], 6)
        graph.add_edge(
            u,
            v,
            **{
                EDGE_COORDINATES_KEY: coordinates,
                EDGE_SPLINE_KEY: B3Spline.from_points(coordinates),
            },
        )
    graph_path = tmp_path / "graph.json"
    SkeletonGraph(graph=graph).to_json_file(str(graph_path))

    segmentation = np.zeros((200, 200, 200), dtype=np.uint16)
    segmentation[80:120, 80:120, 80:120] = 1
    segmentation[20:40, 0:30, 20:40] = 2
    segmentation_path = tmp_path / "segmentation.zarr"
    array = zarr.open(
        str(segmentation_path),
        mode="w",
        shape=segmentation.shape,
        chunks=(64, 64, 64),
        dtype=segmentation.dtype,
    )
    array[:] = segmentation
    return graph_path, segmentation_path


@pytest.fixture
def run_in_viewer(data_paths):
    """Run an async scenario against a live viewer inside a qasync loop."""
    if not _has_gpu():
        pytest.skip("no GPU adapter available")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import qasync
    from app_model import Application
    from qtpy.QtWidgets import QApplication

    from skeleplex.app import view_skeleton

    qapp = QApplication.instance() or QApplication([])
    graph_path, segmentation_path = data_paths

    def _run(scenario, theme: str = "light", segmentation=None) -> None:
        async def _main() -> None:
            app = view_skeleton(
                str(graph_path),
                segmentation_path=str(segmentation or segmentation_path),
                segmentation_voxel_size_um=(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE),
                theme=theme,
            )
            # the docked widgets need room to leave a usable canvas
            app._main_window.resize(2400, 1400)
            viewer = _Viewer(app)
            try:
                await viewer.wait_for_fit()
                await scenario(viewer)
            finally:
                app._main_window.close()
                # app_model only allows one application per name
                Application.destroy(app.name)

        event_loop = qasync.QEventLoop(qapp)
        asyncio.set_event_loop(event_loop)
        with event_loop:
            event_loop.run_until_complete(asyncio.wait_for(_main(), 120))

    return _run


class _Viewer:
    """Drive a viewer through its renderer and read back what it did."""

    def __init__(self, app):
        self.app = app
        self.data = app.data
        self.canvas = app._viewer.main_canvas
        render_manager = app._viewer._backend._render_manager
        self.view = render_manager._canvases[app._viewer._canvas_id]

    async def settle(self, seconds: float = 0.5) -> None:
        """Keep drawing so that slices commit and the pick buffer is current."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.view._canvas.request_draw()
            await asyncio.sleep(0.02)

    async def wait_for_fit(self) -> None:
        """Wait for the startup camera fit, then make the nodes easy to hit."""
        deadline = time.monotonic() + 60
        while self.view._camera_3d.near == 1.0:
            assert time.monotonic() < deadline, "the camera was never fitted"
            await self.settle(0.2)
        await self.settle()
        self.canvas._skeleton.node_visual.appearance.size = NODE_SIZE
        self.canvas._skeleton.node_highlight_visual.appearance.size = NODE_SIZE * 1.3
        await self.settle()

    def project(self, coordinate) -> np.ndarray:
        """Return the canvas position of a world (z, y, x) coordinate."""
        camera_matrix = np.asarray(self.view._camera_3d.camera_matrix)
        clip = camera_matrix @ np.array([*np.asarray(coordinate)[::-1], 1.0])
        x, y, _ = clip[:3] / clip[3]
        width, height = self.view._renderer.logical_size
        return np.array([(x + 1) / 2 * width, (1 - y) / 2 * height])

    def edge_position(self, u: int, v: int) -> np.ndarray:
        """Return the canvas position of the middle of an edge."""
        return self.project(np.mean([NODES[u], NODES[v]], axis=0))

    def node_position(self, node: int) -> np.ndarray:
        """Return a canvas position on a node, clear of its attached edges."""
        graph = self.data.skeleton_graph.graph
        center = self.project(NODES[node])
        directions = [
            self.project(NODES[other]) - center
            for other in set(graph.predecessors(node)) | set(graph.successors(node))
        ]
        away = -sum(d / np.linalg.norm(d) for d in directions)
        camera_right = np.asarray(self.view._camera_3d.world.matrix)[:3, 0]
        edge_of_node = self.project(
            np.asarray(NODES[node]) + camera_right[::-1] * NODE_SIZE / 2
        )
        radius = np.linalg.norm(edge_of_node - center)
        return center + away / np.linalg.norm(away) * radius * 0.7

    def _pointer(self, event_type, position, button, buttons, modifiers=()) -> None:
        self.view._renderer.convert_event(
            {
                "event_type": event_type,
                "x": float(position[0]),
                "y": float(position[1]),
                "button": button,
                "buttons": buttons,
                "modifiers": tuple(modifiers),
                "ntouches": 0,
                "touches": {},
                "time_stamp": time.perf_counter(),
            }
        )

    async def click(self, position, modifiers=("Control",)) -> None:
        """Click and wait for the result to be drawn."""
        self._pointer("pointer_down", position, 1, (1,), modifiers)
        self._pointer("pointer_up", position, 1, (), modifiers)
        await self.settle()

    async def drag(self, start, end) -> None:
        """Drag with the left button and wait for the camera to settle."""
        start, end = np.asarray(start), np.asarray(end)
        self._pointer("pointer_down", start, 1, (1,))
        for fraction in np.linspace(0.1, 1, 10):
            self._pointer("pointer_move", start + (end - start) * fraction, 0, (1,))
        self._pointer("pointer_up", end, 1, ())
        await self.settle(0.8)

    def corner_color(self) -> np.ndarray:
        """Return the RGB color of a corner of the last frame, in [0, 1]."""
        return np.asarray(self.view._renderer.snapshot())[2, 2, :3] / 255

    def edge_colors(self) -> np.ndarray:
        """Return the unique RGB colors of the rendered edges."""
        colors = self.canvas._skeleton.edges_store.colors[:, :3]
        return np.unique(np.round(colors, 2), axis=0)

    def count_pixels(self, is_color) -> int:
        """Count the pixels of the last frame matching a color predicate."""
        frame = np.asarray(self.view._renderer.snapshot())[..., :3].astype(int)
        return int(is_color(frame[..., 0], frame[..., 1], frame[..., 2]).sum())


def _is_magenta(r, g, b):
    return (r > 170) & (g < 90) & (b > 170)


def _is_green(r, g, b):
    return (r < 90) & (g > 170) & (b < 90)


def test_startup_frames_the_whole_skeleton(run_in_viewer):
    """The camera is fitted with clipping planes that scale with the data."""

    async def scenario(viewer):
        camera = viewer.view._camera_3d
        assert camera.far > 2 * SPAN
        assert np.allclose(camera.world.reference_up, (0, 1, 0))
        # (z, y, x) world axes are (x, y, z) in pygfx; looks along +z
        assert camera.world.position[2] < 0
        width, height = viewer.view._renderer.logical_size
        for coordinate in NODES.values():
            x, y = viewer.project(coordinate)
            assert 0 <= x <= width and 0 <= y <= height

    run_in_viewer(scenario)


def test_edge_selection(run_in_viewer):
    """Edges are selected by ctrl-click, in all the supported gestures."""

    async def scenario(viewer):
        selection = viewer.data.selection.edge
        highlight = viewer.canvas._skeleton.edge_highlight_store

        selection.enabled = True
        await viewer.settle()
        await viewer.click(viewer.edge_position(1, 2))
        assert selection.values == {(1, 2)}
        assert highlight.positions.shape == (6, 3)
        assert viewer.count_pixels(_is_magenta) > 0

        # shift adds, and clicking the highlight deselects
        await viewer.click(viewer.edge_position(1, 3), ("Control", "Shift"))
        assert selection.values == {(1, 2), (1, 3)}
        await viewer.click(viewer.edge_position(1, 3), ("Control", "Shift"))
        assert selection.values == {(1, 2)}

        # no control key: not a selection gesture
        await viewer.click(viewer.edge_position(0, 1), ())
        assert selection.values == {(1, 2)}

        # a click on the background clears, unless shift is held
        await viewer.click((3, 3), ("Control", "Shift"))
        assert selection.values == {(1, 2)}
        await viewer.click((3, 3))
        assert selection.values == set()
        assert highlight.positions.shape == (0, 3)

        # disabling the mode detaches the callback
        selection.enabled = False
        await viewer.settle()
        await viewer.click(viewer.edge_position(1, 2))
        assert selection.values == set()

    run_in_viewer(scenario)


def test_node_selection(run_in_viewer):
    """Nodes are selected by ctrl-click, in all the supported gestures."""

    async def scenario(viewer):
        selection = viewer.data.selection.node
        highlight = viewer.canvas._skeleton.node_highlight_store
        position = viewer.node_position(2)

        selection.enabled = True
        await viewer.settle()
        await viewer.click(position)
        assert selection.values == {2}
        assert highlight.positions.shape == (1, 3)
        assert viewer.count_pixels(_is_green) > 0

        await viewer.click(position)
        assert selection.values == set()

        await viewer.click(position)
        await viewer.click((3, 3))
        assert selection.values == set()
        assert highlight.positions.shape == (0, 3)

    run_in_viewer(scenario)


def test_node_and_edge_modes_do_not_interfere(run_in_viewer):
    """With both modes on, a click only changes the selection it hits."""

    async def scenario(viewer):
        edges, nodes = viewer.data.selection.edge, viewer.data.selection.node
        edges.enabled = True
        nodes.enabled = True
        await viewer.settle()

        await viewer.click(viewer.node_position(2))
        assert (nodes.values, edges.values) == ({2}, set())

        await viewer.click(viewer.edge_position(0, 1))
        assert (nodes.values, edges.values) == ({2}, {(0, 1)})

    run_in_viewer(scenario)


def test_segmentation_view_modes(run_in_viewer):
    """The segmentation is placed, shown, hidden and can be clicked through."""

    async def scenario(viewer):
        view = viewer.data.segmentation_view
        components = viewer.canvas._segmentation

        view.mode = "all"
        await viewer.settle(1.0)
        assert components.data_store.data.shape == (200, 200, 200)
        assert components.visual.appearance.opacity == pytest.approx(0.3)
        assert np.allclose(
            np.diag(components.visual.transform.matrix),
            (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1),
        )

        # node 1 is inside label 1, drawn over the skeleton
        viewer.data.selection.node.enabled = True
        await viewer.settle()
        await viewer.click(viewer.node_position(1))
        assert viewer.data.selection.node.values == {1}

        controls = viewer.app._main_window.app_controls.widget()
        controls.segmentation_view_box.opacity_changed.emit(0.75)
        assert components.visual.appearance.opacity == pytest.approx(0.75)

        # a new shape and offset on the same visual
        view.bounding_box._min_coordinate = np.array([7000.0, 7000.0, 7000.0])
        view.bounding_box._max_coordinate = np.array([13000.0, 13000.0, 13000.0])
        view.mode = "bounding_box"
        await viewer.settle(1.0)
        assert components.data_store.data.shape == (60, 60, 60)
        assert np.allclose(components.visual.transform.matrix[:3, 3], 7000.0)

        view.mode = "none"
        await viewer.settle()
        assert components.visual.appearance.visible is False

        view.mode = "all"
        await viewer.settle(1.0)
        assert components.visual.appearance.visible is True
        assert components.data_store.data.shape == (200, 200, 200)

    run_in_viewer(scenario)


def test_curation_updates_the_rendered_skeleton(run_in_viewer):
    """Edits reach the viewer, and the new geometry can still be picked."""

    async def scenario(viewer):
        from skeleplex.app._curate import make_split_edge_widget

        edges = viewer.canvas._skeleton.edges_store

        viewer.app.curate.delete_edge({(1, 3)}, force=True)
        await viewer.settle()
        assert viewer.data.skeleton_graph.graph.number_of_edges() == 2
        assert edges.positions.shape == (12, 3)

        viewer.app.curate.undo()
        await viewer.settle()
        assert viewer.data.skeleton_graph.graph.number_of_edges() == 3
        assert edges.positions.shape == (18, 3)

        selection = viewer.data.selection.edge
        selection.enabled = True
        await viewer.settle()
        await viewer.click(viewer.edge_position(1, 3))
        assert selection.values == {(1, 3)}

        # the split widget previews the split with its own points visual
        widget = make_split_edge_widget(viewer.app)
        widget.edge_to_split_ID.value = "{(1, 3)}"
        widget.split_pos.value = 0.25
        await viewer.settle()
        assert widget.point_visual.appearance.visible is True
        assert widget.point_store.positions.shape == (1, 3)

        widget()
        await viewer.settle()
        assert viewer.data.skeleton_graph.graph.number_of_edges() == 4
        assert edges.positions.shape == (24, 3)

    run_in_viewer(scenario)


def test_render_around_node_with_empty_view(run_in_viewer):
    """A view without any complete edge renders and recovers."""

    async def scenario(viewer):
        viewer.app.curate.render_around_node(
            "{1}", bounding_box_width=4000, render_segmentation=True
        )
        await viewer.settle()
        assert viewer.canvas._skeleton.edges_store.positions.shape == (0, 3)
        assert viewer.canvas._segmentation.data_store.data.shape == (40, 40, 40)

        viewer.data.skeleton_view.mode = "all"
        await viewer.settle()
        assert viewer.canvas._skeleton.edges_store.positions.shape == (18, 3)

    run_in_viewer(scenario)


def test_trackball_camera(run_in_viewer):
    """The camera follows the mouse and can rotate over the pole."""

    async def scenario(viewer):
        camera = viewer.view._camera_3d
        width, height = viewer.view._renderer.logical_size
        start = np.array(camera.world.position)

        ups = []
        for _ in range(3):
            await viewer.drag((width / 2, height * 0.1), (width / 2, height * 0.9))
            camera_up = np.asarray(camera.world.matrix)[:3, 1]
            ups.append(camera_up @ np.asarray(camera.world.reference_up))
        assert np.linalg.norm(np.array(camera.world.position) - start) > 1
        # an orbit controller clamps the elevation, so this would stay positive
        assert min(ups) < 0

    run_in_viewer(scenario)


def test_themes(run_in_viewer):
    """The theme sets the scene colors, and keeps colors from a colormap."""

    async def scenario(viewer):
        light, dark = THEMES["light"], THEMES["dark"]
        node_appearance = viewer.canvas._skeleton.node_visual.appearance

        assert np.allclose(viewer.corner_color(), light.background[:3], atol=0.01)
        assert np.allclose(viewer.edge_colors(), [light.edge_color[:3]])

        viewer.app.toggle_theme()
        await viewer.settle()
        assert np.allclose(viewer.corner_color(), dark.background[:3], atol=0.01)
        assert np.allclose(node_appearance.color, dark.node_color)
        assert np.allclose(viewer.edge_colors(), [dark.edge_color[:3]], atol=0.01)

        # the highlight stays visible and picking is unaffected
        viewer.data.selection.edge.enabled = True
        await viewer.settle()
        await viewer.click(viewer.edge_position(1, 2))
        assert viewer.data.selection.edge.values == {(1, 2)}
        assert viewer.count_pixels(_is_magenta) > 0

        # colors from a colormap survive a theme change
        red = (1.0, 0.0, 0.0, 1.0)
        viewer.data.edge_colormap = EdgeColormap(colormap={}, default_color=Color(red))
        viewer.app.toggle_theme()
        await viewer.settle()
        assert np.allclose(viewer.corner_color(), light.background[:3], atol=0.01)
        assert np.allclose(viewer.edge_colors(), [red[:3]])

        viewer.app.reset_edge_colormap()
        await viewer.settle()
        assert np.allclose(viewer.edge_colors(), [light.edge_color[:3]])

    run_in_viewer(scenario)


def test_start_in_dark_theme(run_in_viewer):
    """A viewer started in the dark theme has its colors from the first frame."""

    async def scenario(viewer):
        dark = THEMES["dark"]
        assert np.allclose(viewer.corner_color(), dark.background[:3], atol=0.01)
        assert np.allclose(viewer.edge_colors(), [dark.edge_color[:3]], atol=0.01)
        assert np.allclose(
            viewer.canvas._skeleton.node_visual.appearance.color, dark.node_color
        )

    run_in_viewer(scenario, theme="dark")


def test_window_fits_a_laptop_screen(run_in_viewer):
    """With all the widgets, the window can be smaller than a laptop screen."""

    async def scenario(viewer):
        minimum_size = viewer.app._main_window.minimumSizeHint()
        assert minimum_size.width() <= 1000
        assert minimum_size.height() <= 700

    run_in_viewer(scenario)


def test_edge_thickness_has_a_minimum_on_screen(run_in_viewer):
    """Edges are as thick as their world size, but at least the minimum in pixels."""

    def _is_blue(r, g, b):
        return (r < 90) & (g < 90) & (b > 170)

    def _rendered_width(viewer) -> int:
        """Return the width in logical pixels of edge (1, 2), across its middle."""
        frame = np.asarray(viewer.view._renderer.snapshot())[..., :3].astype(int)
        pixel_ratio = frame.shape[1] / viewer.view._renderer.logical_size[0]
        row = int(viewer.edge_position(1, 2)[1] * pixel_ratio)
        pixels = _is_blue(frame[row, :, 0], frame[row, :, 1], frame[row, :, 2])
        center = int(viewer.edge_position(1, 2)[0] * pixel_ratio)
        window = pixels[center - 60 : center + 60]
        return round(window.sum() / pixel_ratio)

    async def scenario(viewer):
        skeleton = viewer.canvas._skeleton
        assert skeleton.edges_visual.appearance.thickness_space == "world"
        assert skeleton.edges_min_visual.appearance.thickness_space == "screen"
        assert skeleton.edges_min_visual.appearance.thickness == 3.0
        assert skeleton.edge_highlight_min_visual.appearance.thickness == 6.0

        # the world thickness is below a pixel here, so the minimum shows
        skeleton.edges_visual.appearance.thickness = 1.0
        await viewer.settle()
        minimum_width = _rendered_width(viewer)
        assert minimum_width >= 3

        # above the minimum, the world thickness shows
        skeleton.edges_visual.appearance.thickness = SPAN * 0.01
        await viewer.settle()
        assert _rendered_width(viewer) > minimum_width

        # a redraw keeps both and they follow the data
        viewer.app.curate.delete_edge({(1, 3)}, force=True)
        await viewer.settle()
        assert skeleton.edges_min_visual.appearance.thickness == 3.0
        assert skeleton.edges_store.positions.shape == (12, 3)

    run_in_viewer(scenario)


def test_panels_are_grouped_in_tabs(run_in_viewer):
    """The right side has four rows of tabs, grouped by what they do."""
    from qtpy.QtWidgets import QTabBar

    async def scenario(viewer):
        tab_bars = viewer.app._main_window.findChildren(QTabBar)
        groups = {
            frozenset(bar.tabText(i) for i in range(bar.count())) for bar in tab_bars
        }
        for group in (
            {"Undo", "Delete", "Split"},
            {"Connect", "Merge"},
            {"Around node", "Components"},
            {"Reachable", "Paths"},
        ):
            assert frozenset(group) in groups

    run_in_viewer(scenario)


def test_panel_rows_are_sized_to_their_content(run_in_viewer):
    """A row of small widgets does not take the space of a larger one."""

    async def scenario(viewer):
        docks = viewer.app._main_window._auxiliary_docks
        assert docks["Undo"].height() < docks["Around node"].height()

    run_in_viewer(scenario)


def test_slider_values_are_readable(run_in_viewer):
    """The value box of a slider has room for its text."""
    from qtpy.QtWidgets import QAbstractSpinBox

    async def scenario(viewer):
        window = viewer.app._main_window
        value_boxes = [
            box
            for box in window.findChildren(QAbstractSpinBox)
            if box.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons
            and box.isVisible()
        ]
        assert value_boxes
        for box in value_boxes:
            text_width = box.fontMetrics().horizontalAdvance(box.text())
            assert box.lineEdit().width() >= text_width

    run_in_viewer(scenario)


def test_theme_sets_the_window_colors(run_in_viewer):
    """The theme reaches the whole window, including the styled group boxes."""
    from qtpy.QtGui import QColor, QPalette
    from qtpy.QtWidgets import QApplication

    async def scenario(viewer):
        def is_dark(widget) -> bool:
            image = widget.grab().toImage()
            return image.pixelColor(1, image.height() // 2).lightnessF() < 0.4

        controls = viewer.app._main_window.app_controls.widget()
        group = controls.skeleton_view_box.frame
        window_role = QPalette.ColorRole.Window

        assert not is_dark(group)

        viewer.app.set_theme("dark")
        await viewer.settle()
        assert QApplication.palette().color(window_role) == QColor(
            THEMES["dark"].window.window
        )
        assert is_dark(group)

        viewer.app.set_theme("light")
        await viewer.settle()
        assert not is_dark(group)

    run_in_viewer(scenario)


@pytest.fixture
def multiscale_segmentation(data_paths):
    """Write the test segmentation as an OME-Zarr pyramid."""
    _, segmentation_path = data_paths
    output_path = segmentation_path.with_name("segmentation.ome.zarr")
    return write_ome_zarr_labels(
        segmentation_path, output_path, (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE), n_levels=3
    )


def test_multiscale_segmentation_view_modes(run_in_viewer, multiscale_segmentation):
    """An OME-Zarr pyramid is streamed in all mode and cropped in bounding box mode."""

    def _visibility(viewer) -> tuple[bool, bool]:
        """Return if the multiscale and the in-memory visual are visible."""
        multiscale = viewer.canvas._multiscale_segmentation.visual
        in_memory = viewer.canvas._segmentation.visual
        return (
            multiscale is not None and multiscale.appearance.visible,
            in_memory is not None and in_memory.appearance.visible,
        )

    async def scenario(viewer):
        view = viewer.data.segmentation_view
        assert viewer.data.segmentation_is_multiscale

        view.mode = "all"
        await viewer.settle(1.5)
        components = viewer.canvas._multiscale_segmentation
        assert view.array is None
        assert _visibility(viewer) == (True, False)
        assert components.data_store.n_levels == 3
        assert np.allclose(
            np.diag(components.visual.transform.matrix),
            (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE, 1),
        )
        assert components.visual.appearance.opacity == pytest.approx(0.3)

        # node 1 is inside of label 1, drawn over the skeleton
        viewer.data.selection.node.enabled = True
        await viewer.settle()
        await viewer.click(viewer.node_position(1))
        assert viewer.data.selection.node.values == {1}

        controls = viewer.app._main_window.app_controls.widget()
        controls.segmentation_view_box.opacity_changed.emit(0.75)
        assert components.visual.appearance.opacity == pytest.approx(0.75)

        view.bounding_box._min_coordinate = np.array([7000.0, 7000.0, 7000.0])
        view.bounding_box._max_coordinate = np.array([13000.0, 13000.0, 13000.0])
        view.mode = "bounding_box"
        await viewer.settle(1.0)
        assert _visibility(viewer) == (False, True)
        assert viewer.canvas._segmentation.data_store.data.shape == (60, 60, 60)
        assert viewer.canvas._segmentation.visual.appearance.opacity == pytest.approx(
            0.75
        )

        view.mode = "none"
        await viewer.settle()
        assert _visibility(viewer) == (False, False)

        view.mode = "all"
        await viewer.settle(1.0)
        assert _visibility(viewer) == (True, False)

    run_in_viewer(scenario, segmentation=multiscale_segmentation)


def test_conflicting_voxel_size_is_refused(data_paths, multiscale_segmentation):
    """A voxel size that differs from the metadata raises instead of being ignored."""
    from skeleplex.app import view_skeleton

    graph_path, _ = data_paths
    with pytest.raises(ValueError, match="differs"):
        view_skeleton(
            str(graph_path),
            segmentation_path=str(multiscale_segmentation),
            segmentation_voxel_size_um=(1.0, 1.0, 1.0),
        )


@pytest.fixture
def translated_multiscale_segmentation(data_paths):
    """Write a pyramid whose metadata places it 1000 um along z.

    The labels are written 100 voxels before the position of the plain test
    segmentation, so that the translation brings them back on the skeleton.
    """
    _, segmentation_path = data_paths
    output_path = segmentation_path.with_name("translated.ome.zarr")
    shift_voxels = int(TRANSLATION_Z / VOXEL_SIZE)
    shifted_path = segmentation_path.with_name("shifted.zarr")
    source = zarr.open(str(segmentation_path), mode="r")[:]
    shifted = zarr.open(
        str(shifted_path), mode="w", shape=source.shape, chunks=(64, 64, 64), dtype="u2"
    )
    shifted[:-shift_voxels] = source[shift_voxels:]
    write_ome_zarr_labels(
        shifted_path, output_path, (VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE), n_levels=3
    )
    group = zarr.open_group(str(output_path), mode="r+")
    ome = dict(group.attrs["ome"])
    ome["multiscales"][0]["coordinateTransformations"] = [
        {"type": "translation", "translation": [TRANSLATION_Z, 0.0, 0.0]}
    ]
    group.attrs["ome"] = ome
    return output_path


def test_segmentation_translation_is_applied(
    run_in_viewer, translated_multiscale_segmentation
):
    """A translation in the metadata places the visual and the bounding box crop."""

    async def scenario(viewer):
        view = viewer.data.segmentation_view
        expected = np.array([TRANSLATION_Z, 0.0, 0.0])

        view.mode = "all"
        await viewer.settle(1.0)
        components = viewer.canvas._multiscale_segmentation
        assert np.allclose(components.visual.transform.matrix[:3, 3], expected)
        assert np.allclose(components.data_store.physical_translation, expected)

        # the same world region has to give the same labels as without translation
        view.bounding_box._min_coordinate = np.array([7000.0, 7000.0, 7000.0])
        view.bounding_box._max_coordinate = np.array([13000.0, 13000.0, 13000.0])
        view.mode = "bounding_box"
        await viewer.settle(1.0)
        store = viewer.canvas._segmentation.data_store
        assert store.data.shape == (60, 60, 60)
        assert np.allclose(
            viewer.canvas._segmentation.visual.transform.matrix[:3, 3], 7000.0
        )
        # label 1 fills 8000 to 12000 um on all the axes in world coordinates
        assert store.data[30, 30, 30] == 1
        assert store.data[5, 30, 30] == 0
        assert store.data[9, 30, 30] == 0
        assert store.data[11, 30, 30] == 1

    run_in_viewer(scenario, segmentation=translated_multiscale_segmentation)
