"""Example script of launching the viewer and adding points to it."""

import numpy as np
from cellier.data.points import PointsMemoryStore
from cellier.visuals import PointsMarkerAppearance
from magicgui import magicgui

import skeleplex
from skeleplex.app import view_skeleton

# path_to_graph = "e13_skeleton_graph_image_skel_clean_new_model_v2.json"
path_to_graph = "../scripts/e16_skeleplex_v2.json"

viewer = view_skeleton(graph_path=path_to_graph)


def add_points_to_viewer():
    """Add points to the viewer."""
    # create a list of points to add
    # note these must be Float32
    point_coordinates = np.array(
        [
            [1000, 1000, 1000],
            [1500, 1500, 1500],
        ],
        dtype=np.float32,
    )

    # make the data store for the points (zyx; cellier reverses internally)
    new_points_store = PointsMemoryStore(positions=point_coordinates)

    # add the data and visual to the viewer backend (cellier)
    points_visual = viewer._viewer._backend.add_points(
        data=new_points_store,
        scene_id=viewer._viewer._scene_id,
        appearance=PointsMarkerAppearance(
            size=50, color=(0, 1, 0, 1), size_space="world", color_mode="uniform"
        ),
        name="node_highlight_points",
    )

    # reslice the viewer to update the display
    viewer._viewer._backend.reslice_all()
    return points_visual, new_points_store


points_visual, points_store = add_points_to_viewer()

# set the points visibility to False
points_visual.appearance.visible = False


@magicgui
def update_points():
    """Example widget the updates the point coordinates."""
    # make new point coordinates
    new_point_coordinates = np.random.uniform(0, 5000, (100, 3)).astype(np.float32)

    # set the new coordinates in the points store
    points_store.positions = new_point_coordinates

    # set the points visibility to True
    points_visual.appearance.visible = True

    # reslice the viewer to update the display
    viewer._viewer._backend.reslice_all()


viewer.add_auxiliary_widget(update_points.native, name="Update points")


# start the GUI event loop and block until the application is closed
skeleplex.app.run()
