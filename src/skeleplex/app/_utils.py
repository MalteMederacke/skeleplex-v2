from pathlib import Path

from IPython import get_ipython
from magicgui import magicgui
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication

from skeleplex.app import (
    DataManager,
    ImageFile,
    SkelePlexApp,
    SkeletonDataPaths,
    SkeletonGraphFile,
)
from skeleplex.app._curate import (
    BreakDetectionWidget,
    ChangeBranchColorWidget,
    ConnectedComponentsWidget,
    RenderAroundNodeWidget,
    RenderReachableEdgesWidget,
    make_split_edge_widget,
)

# store reference to QApplication to prevent garbage collection
_app_ref: QApplication | None = None


def view_skeleton(
    graph_path: str,
    segmentation_path: str | None = None,
    segmentation_voxel_size_um: tuple[float, float, float] = (1, 1, 1),
    launch_widgets: bool = True,
):
    """Launch the skeleton viewer application.

    Parameters
    ----------
    graph_path : str
        Path to the skeleton graph JSON file.
    segmentation_path : str | None
        Path to the segmentation image file.
        Must be a zarr file.
    segmentation_voxel_size_um : tuple[float, float, float]
        The voxel size of the segmentation in micrometers.
    launch_widgets : bool, optional
        Whether to launch the auxiliary widgets for curation.
        Defaults to True.

    Returns
    -------
    SkelePlexApp
        The SkelePlex application instance for viewing the skeleton.
    """
    global _app_ref

    # get the qapplication instance
    qapp = QApplication.instance() or QApplication([])

    # Store reference to prevent garbage collection
    _app_ref = qapp

    # load the data
    skeleton_graph_file = SkeletonGraphFile(path=Path(graph_path))
    if segmentation_path is not None:
        segmentation_file = ImageFile(
            path=Path(segmentation_path),
            voxel_size_um=segmentation_voxel_size_um,
        )
    else:
        segmentation_file = None
    data_manager = DataManager(
        file_paths=SkeletonDataPaths(
            skeleton_graph=skeleton_graph_file,
            segmentation=segmentation_file,
        )
    )

    # make the viewer
    viewer = SkelePlexApp(data=data_manager)
    viewer.show()

    # The scene is built and resliced during SkelePlexApp construction, before
    # the event loop is driving, so cellier's async slicer produces no buffers
    # then. Once the loop is running (script: run() via qasync; Jupyter: the
    # kernel's own asyncio loop via start_qt_loop_ipython()), this singleShot
    # fires inside a live asyncio loop, so look_at_skeleton's one-shot
    # fit-on-reslice runs and frames the geometry.
    timer = QTimer()
    timer.singleShot(100, viewer.look_at_skeleton)

    # start the Qt event loop if in Jupyter/IPython
    if should_launch_ipython_event_loop():
        start_qt_loop_ipython()

    if launch_widgets:
        try:
            undo_widget = magicgui(viewer.curate.undo)
            delete_edge_widget = magicgui(
                viewer.curate.delete_edge,
            )
            _ = RenderAroundNodeWidget(viewer)

            connect_without_merging_widget = magicgui(
                viewer.curate.connect_without_merging,
            )
            connect_with_merging_widget = magicgui(
                viewer.curate.connect_with_merging,
            )
            split_edge_widget = make_split_edge_widget(viewer)

            ChangeBranchColorWidget(viewer)
            ConnectedComponentsWidget(viewer)
            BreakDetectionWidget(viewer)

            # add to viewer
            viewer.add_auxiliary_widget(undo_widget.native, name="Undo")
            viewer.add_auxiliary_widget(delete_edge_widget.native, name="Delete edge")

            viewer.add_auxiliary_widget(
                connect_without_merging_widget.native, name="Connect without merging"
            )
            viewer.add_auxiliary_widget(
                connect_with_merging_widget.native, name="Connect with merging"
            )
            viewer.add_auxiliary_widget(split_edge_widget.native, name="Split edge")
            RenderReachableEdgesWidget(viewer)

        except Exception as e:
            print(f"Error launching widgets: {e}")
        return viewer


# GUI servicing cadence for the Jupyter pump (seconds). ~5 ms ≈ 200 Hz, which
# is smoother than any monitor refresh; only Qt input/repaint latency is bounded
# by this. cellier's I/O and asyncio tasks are event-driven on the kernel loop's
# selector and are NOT throttled by this interval.
_QT_PUMP_INTERVAL_S = 0.003


def start_qt_loop_ipython() -> None:
    """Integrate Qt with the event loop in an IPython/Jupyter environment.

    The Jupyter kernel already runs an asyncio loop on the main thread, but
    ipykernel's stock ``enable_gui("qt")`` integration *parks* that loop while
    the kernel is idle and runs Qt's own event loop instead. cellier schedules
    its slice/render coroutines with ``asyncio.ensure_future`` / ``create_task``
    from Qt callbacks; on the parked loop those never step, so the canvas stays
    blank until the next cell wakes the kernel.

    Instead of enabling the stock integration, we keep the kernel's own loop in
    charge and schedule a lightweight pump coroutine on it that calls
    ``QApplication.processEvents`` every :data:`_QT_PUMP_INTERVAL_S` seconds.
    The loop is therefore never parked: cellier's tasks (bound to this same
    loop) keep stepping and Qt stays responsive. Crucially, the interval only
    bounds Qt input/repaint latency — asyncio I/O is serviced by the loop's
    selector the moment it is ready, independent of the pump.

    Idempotent: re-running ``view_skeleton`` in the same kernel reuses the
    existing pump rather than starting a second one.

    Works for both the Jupyter kernel and the IPython console.
    """
    import asyncio

    qapp = QApplication.instance() or QApplication([])

    existing = getattr(qapp, "_skeleplex_qt_pump_task", None)
    if existing is not None and not existing.done():
        # a pump is already running on this kernel's loop; do not stack another.
        return

    async def _pump_qt() -> None:
        while True:
            try:
                qapp.processEvents()
            except Exception:
                # keep the pump alive across any transient Qt event error
                pass
            await asyncio.sleep(_QT_PUMP_INTERVAL_S)

    # Scheduled on the kernel's running loop (this runs during cell execution),
    # so the pump and cellier's tasks share that single, never-parked loop.
    qapp._skeleplex_qt_pump_task = asyncio.ensure_future(_pump_qt())


def should_launch_ipython_event_loop() -> bool:
    """
    Check if the IPython Qt event loop should be launched.

    This means we are in an IPython/Jupyter environment. Idempotency (not
    starting a second pump) is handled inside ``start_qt_loop_ipython``.

    Returns
    -------
    bool
        True if running in IPython and the loop is needed.
        False otherwise.
    """
    shell = get_ipython()

    if not shell:
        # not in IPython environment
        return False

    return True


def run():
    """Start a unified Qt + asyncio event loop and block until close.

    This is meant to be used in a script, after the viewer is set up.

    cellier v2 schedules asyncio tasks from within Qt callbacks — the async
    slicer (``asyncio.ensure_future``) on every reslice and the camera-settle
    debounce (``asyncio.create_task``) on every camera move. A plain
    ``QApplication.exec()`` runs only the Qt loop, so those calls raise
    ``RuntimeError: no running event loop``. ``qasync.QEventLoop`` is an
    asyncio loop backed by Qt, so Qt events and cellier's slicer tasks share
    a single loop.
    """
    import asyncio

    import qasync

    # ensure a QApplication exists before starting the integrated loop
    qapp = QApplication.instance() or QApplication([])
    event_loop = qasync.QEventLoop(qapp)
    asyncio.set_event_loop(event_loop)

    app_close = asyncio.Event()
    qapp.aboutToQuit.connect(app_close.set)
    with event_loop:
        event_loop.run_until_complete(app_close.wait())
