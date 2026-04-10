"""Data class for a skeleton graph."""

import json
import logging
import os

import dask
import dask.array as da
import networkx as nx
import numpy as np
from dask import delayed
from dask.diagnostics import ProgressBar
from splinebox import Spline as SplineboxSpline
from splinebox.spline_curves import _prepared_dict_for_constructor

from skeleplex.graph.constants import (
    DAUGHTER_EDGES_KEY,
    EDGE_COORDINATES_KEY,
    EDGE_SPLINE_KEY,
    GENERATION_KEY,
    LENGTH_KEY,
    NODE_COORDINATE_KEY,
    PARENT_EDGE_KEY,
    SISTER_EDGE_KEY,
)
from skeleplex.graph.image_to_graph import image_to_graph_skan
from skeleplex.graph.spline import B3Spline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def skeleton_graph_encoder(object_to_encode):
    """JSON encoder for the networkx skeleton graph.

    This function is to be used with the Python json.dump(s) functions
    as the `default` keyword argument.
    """
    if isinstance(object_to_encode, np.ndarray):
        return object_to_encode.tolist()
    elif isinstance(object_to_encode, SplineboxSpline):
        spline_dict = object_to_encode._to_dict(version=2)
        if "__class__" in spline_dict:
            raise ValueError(
                "The Spline object to encode already has a '__class__' key."
            )
        spline_dict.update({"__class__": "splinebox.Spline"})
        return spline_dict
    elif isinstance(object_to_encode, B3Spline):
        return object_to_encode.to_json_dict()
    raise TypeError(f"Object of type {type(object_to_encode)} is not JSON serializable")


def skeleton_graph_decoder(json_object):
    """JSON decoder for the networkx skeleton graph.

    This function is to be used with the Python json.load(s) functions
    as the `object_hook` keyword argument.
    """
    if "__class__" in json_object:
        # all custom classes are identified by the __class__ key
        if json_object["__class__"] == "splinebox.Spline":
            json_object.pop("__class__")
            spline_kwargs = _prepared_dict_for_constructor(json_object)
            return SplineboxSpline(**spline_kwargs)
        if json_object["__class__"] == "skeleplex.B3Spline":
            return B3Spline.from_json_dict(json_object)

    # Convert lists back to numpy arrays
    for key, value in json_object.items():
        if isinstance(value, list):
            if key in [NODE_COORDINATE_KEY, EDGE_COORDINATES_KEY]:
                try:
                    json_object[key] = np.array(value)
                except Exception:
                    pass  # If conversion fails, keep it as a list

            if key in [SISTER_EDGE_KEY, DAUGHTER_EDGES_KEY, PARENT_EDGE_KEY]:
                if key == DAUGHTER_EDGES_KEY:
                    json_object[key] = [tuple(edge) for edge in value]
                else:
                    json_object[key] = tuple(value)

    # Convert lists back to numpy arrays
    for key, value in json_object.items():
        if isinstance(value, list):
            if key in [NODE_COORDINATE_KEY, EDGE_COORDINATES_KEY]:
                try:
                    json_object[key] = np.array(value)
                except Exception:
                    pass  # If conversion fails, keep it as a list

            if key in [SISTER_EDGE_KEY, DAUGHTER_EDGES_KEY, PARENT_EDGE_KEY]:
                if key == DAUGHTER_EDGES_KEY:
                    json_object[key] = [tuple(edge) for edge in value]
                else:
                    json_object[key] = tuple(value)

    return json_object


def make_graph_directed(graph: nx.Graph, origin: int) -> nx.DiGraph:
    """Return a directed graph from an undirected graph.

    The directed graph has the same nodes and edges as the undirected graph.
    If the graph is fragmented, meaning has multiple unconnected subgraphs,
    the function will choose the node with the highest degree as the origin node
    for each fragment.

    Parameters
    ----------
    graph : nx.Graph
        The undirected graph to convert to a directed graph.
    origin : int
        The node to use as the origin node for the directed graph.
        The origin node will have no incoming edges.
    """
    if isinstance(graph, nx.DiGraph):
        logger.info("The input graph is already a directed graph.")
        return graph
    if isinstance(graph, nx.MultiGraph):
        logger.info("The input graph is a multi-graph." \
        "Converting to a simple graph first.")
        graph = nx.Graph(graph)
    if len(list(nx.connected_components(graph))) > 1:
        logger.warning("""
        The input graph is not connected.
        The unconnected components might lose edges
        """)
        origin_part = nx.node_connected_component(graph, origin)
        fragments = graph.subgraph(set(graph.nodes()) - origin_part)
        graph = graph.subgraph(origin_part)
    else:
        fragments = None

    di_graph = nx.DiGraph()
    for u, v in nx.bfs_edges(graph, origin):
        di_graph.add_edge(u, v, **graph[u][v])
        di_graph.add_node(u, **graph.nodes[u])
        di_graph.add_node(v, **graph.nodes[v])
    if fragments is not None:
        # choose a node with the highest degree as the origin node
        # Do this for each fragment
        for fragment in nx.connected_components(fragments):
            fragment_subgraph = fragments.subgraph(fragment)
            """Choose a origin of the fragment with the highest degree.
            This is arbitrary but finding a better node
            without knowledge were the network broke is hard"""
            origin = max(fragment_subgraph.degree, key=lambda x: x[1])[0]
            di_fragment = nx.DiGraph()
            for u, v in nx.bfs_edges(fragment_subgraph, origin):
                di_fragment.add_edge(u, v, **fragment_subgraph[u][v])
                di_fragment.add_node(u, **fragment_subgraph.nodes[u])
                di_fragment.add_node(v, **fragment_subgraph.nodes[v])
            di_graph.add_edges_from(di_fragment.edges(data=True))
            di_graph.add_nodes_from(di_fragment.nodes(data=True))

    return di_graph


def get_next_node_key(graph: nx.Graph) -> int:
    """Return the next available node key in the graph.

    This function assumes the graph node keys are integers.

    Parameters
    ----------
    graph : nx.Graph
        The graph to get the next node key from.

    Returns
    -------
    int
        The next available node key.
        If there are no nodes, the function returns 0.
    """
    node_numbers = list(graph.nodes)
    node_numbers.sort()

    if len(node_numbers) == 0:
        return 0

    free_node = node_numbers[0] + 1
    while free_node in node_numbers:
        node_numbers.pop(0)
        free_node = node_numbers[0] + 1
    return free_node


def orient_splines(
    graph: nx.DiGraph, approximate_positions: bool = False
) -> nx.DiGraph:
    """Checks if the splines are oriented correctly.

    If the beginning of the spline is closer to the end node than the start node,
    it gets flipped and vice versa.

    Also checks if the edge coordinates are aligned with the spline.
    This only checks, if the splines are correctly connected to the nodes,
    not the order in the Graph. Best used on a directed graph.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to orient the splines in.
    approximate_positions : bool
        Whether to use approximate positions for the spline evaluation.
        This is faster but less accurate. Default is False.

    Returns
    -------
    nx.DiGraph
        The graph with the splines oriented correctly.

    """
    edge_spline_dict = {}
    edge_coordinates_dict = {}

    for u, v, attr in graph.edges(data=True):
        spline = attr[EDGE_SPLINE_KEY]
        u_coord = graph.nodes[u][NODE_COORDINATE_KEY]
        v_coord = graph.nodes[v][NODE_COORDINATE_KEY]

        # Evaluate spline endpoints
        spline_start, spline_end = spline.eval(
            np.array([0.01, 0.99]), approx=approximate_positions
        )

        # Compute distances from spline endpoints to node positions
        dist_start_to_u = np.linalg.norm(spline_start - u_coord)
        dist_start_to_v = np.linalg.norm(spline_start - v_coord)
        dist_end_to_u = np.linalg.norm(spline_end - u_coord)
        dist_end_to_v = np.linalg.norm(spline_end - v_coord)

        # Orientation is correct if start→u and end→v are both closest
        start_correct = dist_start_to_u < dist_start_to_v
        end_correct = dist_end_to_v < dist_end_to_u

        # If either endpoint is closer to the wrong node, flip it
        if not (start_correct and end_correct):
            logger.info(f"Flipped spline of edge ({u},{v}).")

            edge_coordinates = attr[EDGE_COORDINATES_KEY]
            flipped_spline, flipped_cords = spline.flip_spline(edge_coordinates)
            edge_spline_dict[(u, v)] = flipped_spline
            edge_coordinates_dict[(u, v)] = flipped_cords

    nx.set_edge_attributes(graph, edge_spline_dict, EDGE_SPLINE_KEY)
    nx.set_edge_attributes(graph, edge_coordinates_dict, EDGE_COORDINATES_KEY)

    return graph


def sample_slices_for_edge(
    u: int,
    v: int,
    spline: B3Spline,
    volume_path: str,
    segmentation_path: str | None,
    positions: np.ndarray,
    slice_size_um: float,
    sample_grid_spacing_um: float,
    interpolation_order: int,
    image_voxel_size_um: float,
    approx: bool,
):
    """Helper function for parallel execution.

    This function is used to sample slices from a spline in a skeleton graph.

    Parameters
    ----------
    u : int
        The source node of the edge.
    v : int
        The target node of the edge.
    spline : B3Spline
        The spline to sample from.
    volume_path : str
        The path to the volume to sample from.
    segmentation_path : str
        The path to the segmentation to sample from.
    positions : np.ndarray
        The positions to sample from.
    slice_size_um : float
        The edge width of the slice in microns.
    sample_grid_spacing_um : float
        The spacing of the sample points in microns.
    interpolation_order : int
        The order of the interpolation to use.
    image_voxel_size_um : float
        The voxel size of the image in micrometers.
    approx : bool
        Whether to use approximate sampling.

    Returns
    -------
    tuple
        A tuple containing the edge (u, v), image slice, and segmentation slice.

    """
    logger.info(f"Sampling slices from edge ({u}, {v})")

    volume = da.from_zarr(volume_path, chunks="auto")

    image_slice = spline.sample_volume_2d(
        volume,
        positions,
        grid_shape_um=(slice_size_um, slice_size_um),
        grid_spacing_um=(sample_grid_spacing_um, sample_grid_spacing_um),
        sample_interpolation_order=interpolation_order,
        sample_fill_value=0,
        image_voxel_size_um=image_voxel_size_um,
        approx=approx,
    )

    segmentation_slice = None
    if segmentation_path is not None:
        segmentation = da.from_zarr(segmentation_path, chunks="auto")
        segmentation_slice = spline.sample_volume_2d(
            segmentation,
            positions,
            grid_shape_um=(slice_size_um, slice_size_um),
            grid_spacing_um=(sample_grid_spacing_um, sample_grid_spacing_um),
            sample_interpolation_order=0,
            image_voxel_size_um=image_voxel_size_um,
            sample_fill_value=0,
            approx=approx,
        )

    return (u, v), image_slice, segmentation_slice


class SkeletonGraph:
    """Data class for a skeleton graph.

    Parameters
    ----------
    graph : nx.Graph
        The skeleton graph.
    origin : int, optional
        The origin node of the graph.
    image_path : str, optional
        The path to the image the graph was extracted from.
    image_key : str, optional
        The key of the image in the image file.
    voxel_size_um : float, optional
        The voxel size of the image in the image file.
        We keep the graph scaled to um for easier handling.


    origin : int, optional
        The origin node of the graph.
    image_path : str, optional
        The path to the image the graph was extracted from.
    image_key : str, optional
        The key of the image in the image file.
    voxel_size_um : float, optional
        The voxel size of the image in the image file.
        We keep the graph scaled to um for easier handling.


    """

    _backend = "networkx"

    def __init__(
        self,
        graph: nx.Graph,
        origin: int | None = None,
        image_path: str | None = None,
        image_key: str | None = None,
        voxel_size_um: float | None = None,
    ):
        self.graph = graph
        self.origin = origin
        self.image_path = image_path
        self.image_key = image_key
        self.voxel_size_um = voxel_size_um
        self.origin = origin
        self.image_path = image_path
        self.image_key = image_key
        self.voxel_size_um = voxel_size_um

    @property
    def backend(self) -> str:
        """Return the backend used to store the graph."""
        return self._backend

    @property
    def nodes(self):
        """Return a list of nodes."""
        return self.graph.nodes()

    @property
    def node_coordinates(self) -> dict:
        """Return a dictionary of node coordinates."""
        node_coordinates = {}
        for node, node_data in self.graph.nodes(data=True):
            node_coordinates[node] = node_data[NODE_COORDINATE_KEY]
        return node_coordinates

    @property
    def node_coordinates_array(self) -> np.ndarray:
        """Return a numpy array of node coordinates.

        The array is of shape (n_nodes, n_dimensions).
        The order of the nodes is the same as the order of the nodes attribute.
        """
        return np.array(
            [
                node_data[NODE_COORDINATE_KEY]
                for _, node_data in self.graph.nodes(data=True)
            ]
        )

    @property
    def edges(self):
        """Return a list of edges."""
        return self.graph.edges()

    @property
    def edge_splines(self) -> dict:
        """Return a list of edge splines."""
        edge_splines = {}
        for edge_start, edge_end, edge_data in self.graph.edges(data=True):
            edge_splines[(edge_start, edge_end)] = edge_data[EDGE_SPLINE_KEY]
        return edge_splines

    def to_json_file(self, file_path: str):
        """Return a JSON representation of the graph."""
        graph_dict = nx.node_link_data(self.graph, edges="edges")
        # if one of the attributes is not None, add it to the dict
        # if not add a placeholder

        object_dict = {
            "graph": graph_dict,
            "origin": self.origin,
            "image_path": self.image_path,
            "image_key": self.image_key,
            "voxel_size_um": self.voxel_size_um,
        }

        with open(file_path, "w") as file:
            json.dump(object_dict, file, indent=2, default=skeleton_graph_encoder)

    @classmethod
    def from_json_file(cls, file_path: str):
        """Return a SkeletonGraph from a JSON file."""
        with open(file_path) as file:
            object_dict = json.load(file, object_hook=skeleton_graph_decoder)
        graph = nx.node_link_graph(object_dict["graph"], edges="edges")
        skeleton_object = cls(graph=graph)
        # do only if keys exist
        if "origin" in object_dict:
            skeleton_object.origin = object_dict["origin"]
        if "image_path" in object_dict:
            skeleton_object.image_path = object_dict["image_path"]
        if "image_key" in object_dict:
            skeleton_object.image_key = object_dict["image_key"]
        if "voxel_size_um" in object_dict:
            voxel_size_um = object_dict["voxel_size_um"]
            if isinstance(voxel_size_um, float | int):
                voxel_size_um = (voxel_size_um, voxel_size_um, voxel_size_um)
            skeleton_object.voxel_size_um = voxel_size_um

        return skeleton_object

    @classmethod
    def from_skeleton_image(
        cls,
        skeleton_image: np.ndarray,
        max_spline_knots: int = 10,
        image_voxel_size_um: float = 1,
    ) -> "SkeletonGraph":
        """Return a SkeletonGraph from a skeleton image.

        Parameters
        ----------
        skeleton_image : np.ndarray
            The skeleton image to convert to a graph.
        max_spline_knots : int
            The maximum number of knots to use for the spline fit to the branch path.
            If the number of data points in the branch is less than this number,
            the spline will use n_data_points - 1 knots.
            See the splinebox Spline class docs for more information.
        image_voxel_size_um  : float or array of float
            Spacing of the voxels. Used to transform graph coordinates to um.
        """
        graph = image_to_graph_skan(
            skeleton_image=skeleton_image,
            max_spline_knots=max_spline_knots,
            image_voxel_size_um=image_voxel_size_um,
        )
        return cls(graph=graph)

    @classmethod
    def from_graph(
        cls,
        graph,
        edge_coordinate_key,
        node_coordinate_key,
        voxel_size_um: float | None = None,
        scale_to_um: bool = True,
    ) -> "SkeletonGraph":
        """Return a SkeletonGraph from a networkx graph.

        The edges and nodes need to have an attribute with the specified keys
        containing the coordinates of the nodes and edges as an np.ndarray.
        Requires edge coordinates of length greater than 4
        to successfully create a spline.

        Parameters
        ----------
        graph : nx.Graph
            The graph to convert to a SkeletonGraph.
        edge_coordinate_key : str
            The key to use for the edge coordinates.
        node_coordinate_key : str
            The key to use for the node coordinates.
        voxel_size_um : float | None
            Spacing of the voxels. Used to transform graph coordinates to um.
        scale_to_um : bool
            Whether to scale the coordinates to micrometers using the voxel size.
        """
        graph_mod = graph.copy()
        for _, _, attr in graph_mod.edges(data=True):
            # attr[EDGE_COORDINATES_KEY] = attr.pop(edge_coordinate_key)
            edge_coords = attr[edge_coordinate_key]
            # add spline
            if scale_to_um and voxel_size_um is not None:
                edge_coords = edge_coords * voxel_size_um
            attr[EDGE_COORDINATES_KEY] = edge_coords
            spline = B3Spline.from_points(attr[EDGE_COORDINATES_KEY])
            attr[EDGE_SPLINE_KEY] = spline
        for _, node_data in graph_mod.nodes(data=True):
            node_coords = node_data.pop(node_coordinate_key)
            if scale_to_um and voxel_size_um is not None:
                node_coords = node_coords * voxel_size_um
            node_data[NODE_COORDINATE_KEY] = node_coords
        return cls(graph=graph_mod, voxel_size_um=voxel_size_um)

    def __eq__(self, other: "SkeletonGraph"):
        """Check if two SkeletonGraph objects are equal."""
        if set(self.nodes) != set(other.nodes):
            # check if the nodes are the same
            return False
        elif set(self.edges) != set(other.edges):
            # check if the edges are the same
            return False
        else:
            return True

    def to_directed(self, origin: int) -> nx.DiGraph:
        """Return a directed graph from the skeleton graph.

        The directed graph has the same nodes and edges as the skeleton graph.
        Stores the origin node as an attribute.

        The directed graph is stored in-place.

        Parameters
        ----------
        origin : int
            The node to use as the origin node for the directed graph.
            The origin node will have no incoming edges.
        """
        self.graph = make_graph_directed(self.graph, origin)
        self.origin = origin
        return self.graph

    def orient_splines(self, approximate_positions: bool = False) -> nx.DiGraph:
        """Orient the splines in the graph.

        This modifies the graph in place.

        Parameters
        ----------
        approximate_positions : bool
            Whether to use approximate positions for the spline evaluation.
            This is faster but less accurate. Default is False.
        """
        self.graph = orient_splines(
            self.graph, approximate_positions=approximate_positions
        )
        return self.graph

    def compute_branch_lengths(self) -> dict:
        """Return a dictionary of edge lengths.

        The keys of the dictionary are the edge tuples, the values are arc lengths
        of the fitted splines. Units will be the same as voxel scale.
        """
        edge_lengths = {}
        for u, v, attr in self.graph.edges(data=True):
            edge_lengths[(u, v)] = attr[EDGE_SPLINE_KEY].arc_length

        nx.set_edge_attributes(self.graph, edge_lengths, LENGTH_KEY)
        return edge_lengths

    def sample_volume_slices_from_spline(
        self,
        volume: np.ndarray,
        slice_spacing: float,
        slice_size_um: float,
        sample_grid_spacing_um: float,
        interpolation_order: int = 3,
        max_generation: int | None = None,
        segmentation: np.ndarray | None = None,
        approx: bool = False,
    ):
        """Sample volume slices from the splines in the graph.

        Parameters
        ----------
        volume : np.ndarray
            The volume to sample slices from.
        slice_spacing : float
            The spacing between slices. Normalized between 0 and 1.
        slice_size_um : float
            The edge length of the slices in microns.
            The slice is square.
        sample_grid_spacing_um : float
            The spacing of the sample points in microns.
            This is the size of the pixels in microns in the resulting slices.
        interpolation_order : int
            The order of the interpolation to use for the spline.
            For labels use 0
        max_generation : int
            The maximum generation of the spline to sample.
            If None, all levels are sampled.
        segmentation : np.ndarray | None
            The segmentation to sample slices from.
            If None, only the image is sampled.
        approx : bool
            If True, use a quick conversion from normalized arc length
            coordinates to spline parameter coordinates.
            The more evenly spaced the spline knots are, the more accurate this
            approximation becomes.
            If False, use a binary search to find the parameterized arc length
            that corresponds to the normalized arc length coordinates.
            Default value is False.

        Returns
        -------
        dict
            A dictionary of image slices.

        Optional: tuple(dict, dict)
            A tuple of dictionaries of image and segmentation slices.
            Only if segmentation is not None.

        """
        image_voxel_size_um = self.voxel_size_um

        origin = self.origin
        if not origin and origin != 0:
            raise ValueError("No origin node provided. Please set origin.")

        if image_voxel_size_um is None:
            logger.warning("No voxel size provided. Assuming pixel size is 1 µm.")
            image_voxel_size_um = (1, 1, 1)

        graph = self.graph.copy()
        generation_dict = nx.get_edge_attributes(graph, GENERATION_KEY)
        spline_dict = nx.get_edge_attributes(graph, EDGE_SPLINE_KEY)

        image_slice_dict = {}
        segmentation_slice_dict = {}

        for u, v in nx.breadth_first_search.bfs_edges(graph, source=origin):
            logger.info(f"Sampling slices from edge ({u}, {v})")

            if max_generation and (generation_dict[(u, v)] >= max_generation):
                break
            spline = spline_dict[(u, v)]
            positions = np.linspace(0.1, 0.9, np.ceil(1 / slice_spacing).astype(int))
            image_slice = spline.sample_volume_2d(
                volume,
                positions,
                grid_shape_um=(slice_size_um, slice_size_um),
                grid_spacing_um=(sample_grid_spacing_um, sample_grid_spacing_um),
                sample_interpolation_order=interpolation_order,
                image_voxel_size_um=image_voxel_size_um,
                approx=approx,
            )
            image_slice_dict[(u, v)] = image_slice

            if segmentation is not None:
                segmentation_slice = spline.sample_volume_2d(
                    segmentation,
                    positions,
                    grid_shape_um=(slice_size_um, slice_size_um),
                    grid_spacing_um=(sample_grid_spacing_um, sample_grid_spacing_um),
                    sample_interpolation_order=0,
                    image_voxel_size_um=image_voxel_size_um,
                    approx=approx,
                )
                segmentation_slice_dict[(u, v)] = segmentation_slice

        if segmentation is not None:
            return image_slice_dict, segmentation_slice_dict
        else:
            return image_slice_dict

    def sample_volume_slices_from_spline_parallel(
        self,
        volume_path: da.Array,
        slice_spacing: float,
        slice_size_um: float,
        sample_grid_spacing_um: float,
        interpolation_order: int = 3,
        max_generation: int | None = None,
        min_generation: int | None = None,
        segmentation_path: da.Array | None = None,
        num_workers: int = os.cpu_count() - 2,
        approx: bool = False,
    ):
        """Sample volume slices from the splines in the graph in parallel.

        Loads volume and segmentation data lazily and processes edges in parallel.

        Parameters
        ----------
        volume_path : str
            The volume path to sample slices from.
            Must be path to a zarr file. Otherwise its not pickleable.
        slice_spacing : float
            The spacing between slices. Normalized between 0 and 1.
        slice_size_um : int
            The edge length slices in microns. The slices are square.
        sample_grid_spacing_um : float
            The spacing of the sample points in microns.
            This is the size of the pixels in microns in the resulting slices.
        interpolation_order : int
            The order of the interpolation to use for the spline.
            For labels use 0
        max_generation : int
            The maximum generation of the spline to sample.
            If None, all levels are sampled.
        min_generation : int
            The minimum generation of the spline to sample.
            If None, all levels are sampled.
        segmentation_path : str | None
            The segmentation path to sample slices from.
            Must be path to a zarr file. Otherwise its not pickleable.
            If None, only the image is sampled.
        num_workers : int
            The number of workers to use for parallel processing.
            Defaults to the number of CPU cores minus 2.
        approx : bool
            If True, use the approximate spline evaluation.
            If False, use the exact spline evaluation.

        """
        image_voxel_size_um = self.voxel_size_um
        origin = self.origin
        if not origin and origin != 0:
            raise ValueError("No origin node provided. Please set origin.")

        if image_voxel_size_um is None:
            logger.warning("No voxel size provided. Assuming pixel size is 1 µm.")
            image_voxel_size_um = (1, 1, 1)

        generation_dict = nx.get_edge_attributes(self.graph, GENERATION_KEY)
        spline_dict = nx.get_edge_attributes(self.graph, EDGE_SPLINE_KEY)

        # Prepare a list of edges to process
        edges_to_process = [
            (u, v)
            for u, v in nx.breadth_first_search.bfs_edges(self.graph, source=origin)
            if (max_generation is None or generation_dict[(u, v)] <= max_generation)
            and (min_generation is None or generation_dict[(u, v)] > min_generation)
        ]

        # Define a function to process each edge,
        # needs to be local to be pickable, required for laziness
        def process_edge(edge):
            u, v = edge
            spline = spline_dict[(u, v)]
            positions = np.linspace(0.1, 0.9, np.ceil(1 / slice_spacing).astype(int))
            return sample_slices_for_edge(
                u,
                v,
                spline,
                volume_path,
                segmentation_path,
                positions,
                slice_size_um,
                sample_grid_spacing_um,
                interpolation_order,
                image_voxel_size_um,
                approx,
            )

        # Use Dask delayed to process edges in parallel
        tasks = [delayed(process_edge)(edge) for edge in edges_to_process]

        # Compute all tasks in parallel
        logger.info(f"Processing {len(tasks)} edges in parallel.")
        with ProgressBar():
            results = dask.compute(
                *tasks, num_workers=num_workers, scheduler="processes"
            )

        # Combine results into dictionaries
        image_slice_dict = {}
        segmentation_slice_dict = {}

        for result in results:
            (u, v), image_slice, segmentation_slice = result
            image_slice_dict[(u, v)] = image_slice
            if segmentation_path is not None:
                segmentation_slice_dict[(u, v)] = segmentation_slice
        if segmentation_path is not None:
            return (
                image_slice_dict,
                segmentation_slice_dict,
            )
        return image_slice_dict
