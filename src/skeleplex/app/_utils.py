import sys
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
from skeleplex.app._curate import (ChangeBranchColorWidget,
                                   make_split_edge_widget,
                                   RenderReachableEdgesWidget
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

    # Wait a short time for things to load and then look at the skeleton
    # this is a hack...do something smarter later
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
            render_around_node_widget = magicgui(
                viewer.curate.render_around_node,
                node_id={
                    "widget_type": "LineEdit",
                },
                bounding_box_width={"min": 0, "max": sys.float_info.max},
            )

            connect_without_merging_widget = magicgui(
                viewer.curate.connect_without_merging,
            )
            connect_with_merging_widget = magicgui(
                viewer.curate.connect_with_merging,
            )
            split_edge_widget = make_split_edge_widget(viewer)

            ChangeBranchColorWidget(viewer)

            # add to viewer
            viewer.add_auxiliary_widget(undo_widget.native, name="Undo")
            viewer.add_auxiliary_widget(delete_edge_widget.native, name="Delete edge")
            viewer.add_auxiliary_widget(
                render_around_node_widget.native, name="Render around node"
            )
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


def start_qt_loop_ipython():
    """Start the Qt event loop in an IPython environment.

    This works for both jupyter and ipython console environments.
    """
    ipython = get_ipython()
    ipython.enable_gui("qt")


def should_launch_ipython_event_loop() -> bool:
    """
    Check if the IPython Qt event loop should be launched.

    This means that we are both in an IPython environment and the
    event loop is not already running.

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

    return not shell.active_eventloop == "qt"


def run():
    """Start the Qt application event loop.

    This is meant to be used in a script.
    This should be called after the viewer is set up.
    """
    # get the qapplication instance
    qapp = QApplication.instance() or QApplication([])

    # start the Qt application event loop
    qapp.exec_()
