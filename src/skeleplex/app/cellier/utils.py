"""Construction helpers for the cellier v2 backend."""

from uuid import UUID

from cellier.controller import CellierController
from cellier.render import RenderManagerConfig, TemporalAccumulationConfig
from cellier.scene import spatial_axes


def make_controller() -> tuple[CellierController, UUID]:
    """Create a CellierController with a single 3D scene.

    Returns
    -------
    tuple[CellierController, UUID]
        The controller and the id of its main 3D scene.
    """
    # Temporal accumulation is an EMA blend of frames; a higher ``blend_weight``
    # weights the current frame more, so the image fades/converges faster after
    # a camera move (default blend_weight is 0.1).
    render_config = RenderManagerConfig(
        temporal=TemporalAccumulationConfig(blend_weight=0.5),
    )
    controller = CellierController(render_config=render_config)
    scene = controller.add_scene(
        name="main_viewer_scene",
        displayed_axes=("z", "y", "x"),
        coordinate_system=spatial_axes("z", "y", "x"),
    )
    return controller, scene.id
