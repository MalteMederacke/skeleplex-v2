import dask.array as da  # noqa: D100
from skimage.morphology import skeletonize as ski_skeletonize

from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.skeleton._skeletonize import skeletonize
from skeleplex.skeleton.distance_field import local_normalized_distance_gpu

# This script processes a segmentation to a skeleton graph in memory

# First run examples/image_to_graph_lazy/create_example_data.py
# to create the segmentation
segmentation_path = "../example_data/bifurcating_tree.zarr"
segmentation = da.from_zarr(segmentation_path)

distance_field = da.map_overlap(
    local_normalized_distance_gpu, segmentation, depth=5
).compute()


skeleton_prediction = skeletonize(distance_field)

# Mask with segmentation to restict to segmented area
skeleton_prediction = skeleton_prediction * segmentation.compute()

skeleton = skeleton_prediction > 0.3
skeleton = ski_skeletonize(skeleton)
skeleton_graph = SkeletonGraph.from_skeleton_image(skeleton)
skeleton_graph.to_json_file("../example_data/skeleton_graph.json")
