import logging  # noqa
import os
from typing import TYPE_CHECKING

import networkx as nx
import h5py
import numpy as np
import skimage as ski
from skimage.morphology import dilation, disk
import torch
from tqdm import tqdm
import concurrent.futures

from skeleplex.measurements.utils import grey2rgb, radius_from_area
from skeleplex.graph.skeleton_graph import SkeletonGraph
from skeleplex.graph.constants import (
    DIAMETER_KEY,
    TISSUE_THICKNESS_KEY,
)

if TYPE_CHECKING:
    from skeleplex.measurements.lumen_classifier import ConvNext3ClassPredictor

    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def filter_and_segment_lumen(
    data_path,
    save_path,
    sam_checkpoint_path: str | None = None,
    lumen_classification_model: "ConvNext3ClassPredictor | None" = None,
    eccentricity_thresh=0.7,
    circularity_thresh=0.5,
    find_lumen=True,
    sam_quality_threshold=0.5,
    segmentation_key: str = "segmentation",
):
    """
    Filter and segment the lumen in the image slices.

    Uses the spline to seed an prompt for SAM2
    https://github.com/facebookresearch/sam2/tree/main

    And a resnet classifier to classify the slices into lumen, branches and bad.

    By convention, the lumen is labeled as 2, the tissue as 1 and the background as 0.

    Only the central label of the slice is considered, which is the label at the
    center of the slice. If there is no central label, the slice is dropped.
    If the central label is touching the border of the slice, the slice is dropped.

    The lumen is the central "open" part of the slice, which is not
    touching the border, has a low eccentricity and high circularity and is usually
    characterized by a low intensity in the image. If the segmentation only covers the
    lumen, find_lumen should be set to False, which will only filter the slices
    based on eccentricity and circularity.

    Eccentricity is defined as the ratio of the distance between the foci of the
    ellipse and the length of the major axis. A value close to 0 indicates a circle
    and a value close to 1 indicates a line. Circularity is defined as the ratio
    of the area of the shape to the area of a circle with the same perimeter. A
    value close to 1 indicates a circle and a value close to 0 indicates a
    very irregular/bumpy shape.

    https://en.wikipedia.org/wiki/Eccentricity_(mathematics)


    If the segmentation covers the total diameter of the branch, thus segmenting the
    lumen and the tissue, find_lumen can be set to True, which will use the spline to
    seed a prompt for SAM2

    https://github.com/facebookresearch/sam2/tree/main .

    This will segment the branch and returns different masks, ideally one for the one
    for the lumen and one for the whole branch. To classify the different masks, a
    ResNet classifier is used, which is trained on the lumen, branches and bad slices.
    The classifier is used to classify the lumen and branches, and the bad slices are
    filtered out. The classifier needs to be trained on the lumen, branches and
    bad slices.

    Parameters
    ----------
    data_path : str
        Path to the input data directory containing .h5 files.
    save_path : str
        Path to the output directory where filtered .h5 files will be saved.
    sam_checkpoint_path : str, optional
        Path to the SAM2 checkpoint file.
        only required if find_lumen is True.
    lumen_classification_model : ConvNext3ClassPredictor, optional
        ConvNext classifier for predicting lumen classes.
        Only required if find_lumen is True.
    eccentricity_thresh : float
        Eccentricity threshold for filtering slices.
    circularity_thresh : float
        Circularity threshold for filtering slices.
    find_lumen : bool
        Whether to find the lumen using SAM2 or just to filter for
        eccentricity and circularity.
    sam_quality_threshold : float
        Minimum SAM quality score to consider a mask for classification.
    segmentation_key : str
        The h5 dataset key to use for the segmentation slices.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"Created directory: {save_path}")

    files = [f for f in os.listdir(data_path) if f.endswith(".h5")]
    # only do those that are npot in the save_path
    files = [f for f in files if not os.path.exists(os.path.join(save_path, f))]
    logger.info(f"Found {len(files)} files to process.")
    files = tqdm(files, desc="Processing files")
    for file in files:
        logger.info(f"Processing {file}")

        with h5py.File(os.path.join(data_path, file), "r") as f:
            image_slices = f["image"][:]
            segmentation_slices = f[segmentation_key][:] != 0

        label_slices_filt = np.zeros_like(segmentation_slices, dtype=np.uint8)
        index_to_remove = []

        for i in range(len(image_slices)):
            image_slice = image_slices[i]
            segmentation_slice = segmentation_slices[i]
            label_slice = ski.measure.label(segmentation_slice)
            h, w = image_slice.shape
            central_label = label_slice[h // 2, w // 2]

            # Skip if no central label
            if central_label == 0:
                index_to_remove.append(i)
                continue

            # Remove if touching the border
            if (
                np.any(label_slice[0, :] == central_label)
                or np.any(label_slice[-1, :] == central_label)
                or np.any(label_slice[:, 0] == central_label)
                or np.any(label_slice[:, -1] == central_label)
            ):
                index_to_remove.append(i)
                continue

            label_slice[label_slice != central_label] = 0
            label_slice[label_slice == central_label] = 1

            # Check eccentricity
            props = ski.measure.regionprops(label_slice)
            if props[0].eccentricity > eccentricity_thresh:
                index_to_remove.append(i)
                continue

            # Check circularity
            circularity = 4 * np.pi * props[0].area / (props[0].perimeter ** 2)
            if circularity < circularity_thresh:
                index_to_remove.append(i)
                continue

            if find_lumen:
                # delay imports so that the function can be used without SAM2
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sam2_checkpoint = sam_checkpoint_path
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
                sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
                predictor = SAM2ImagePredictor(sam2_model)

                label_slice = find_lumen_in_slice(
                    image_slice,
                    label_slice,
                    predictor,
                    lumen_classification_model,
                    sam_quality_threshold,
                )
            label_slices_filt[i] = label_slice

        # Remove invalid slices
        image_slice_filt = np.delete(image_slices, index_to_remove, axis=0)
        label_slices_filt = np.delete(label_slices_filt, index_to_remove, axis=0)

        with h5py.File(os.path.join(save_path, file), "w") as f:
            f.create_dataset("image", data=image_slice_filt)
            f.create_dataset("segmentation", data=label_slices_filt)


def find_lumen_in_slice(
    image_slice: np.ndarray,
    label_slice: np.ndarray,
    predictor: "SAM2ImagePredictor",
    lumen_classification_model: "ConvNext3ClassPredictor",
    sam_quality_threshold=0.5,
) -> np.ndarray:
    """
    Find lumen and branch labels in a single 2D image slice.

    Uses a SAM-based segmentation followed by a ResNet-based mask classifier.
    This function performs the following high-level steps:
    1. Converts the grayscale image slice to RGB and runs the provided SAM predictor
        with a single central point to obtain up to three candidate masks and
        associated quality scores.
    2. For each SAM mask, creates a masked image (image * mask), crops it to the
        mask's bounding box, and sends the cropped mask image to the provided
        lumen_classification_model to obtain a predicted class and confidence.
    3. Builds a new label image where:
        - label value 0 denotes background,
        - label value 1 denotes branch tissue,
        - label value 2 denotes lumen.
        Classification decisions follow these rules:
        - Only masks with SAM quality > 0.1 are passed to the ResNet classifier.
        - If no masks pass the quality threshold, the original label_slice is
            returned unchanged.
        - If one or more masks are predicted as lumen (class 0), the highest-quality
            lumen mask is used to mark lumen (value 2). If that lumen mask touches the
            background (according to lumen_touches_background) and another lumen mask
            exists, the second-best lumen mask will be tried.
        - Branch masks (class 1) are chosen by quality. If the best class-1 mask has
            quality < 0.5, branch pixels are copied from the original segmentation;
            otherwise the SAM class-1 mask is used for branch labeling.
        - If all candidate predictions are class 2 (non-lumen, non-branch), or if
            the final lumen segmentation touches the background, the original
            segmentation is returned.
    4. Returns the label slice updated with lumen and branch annotations.
    Parameters.
    ----------
    image_slice : numpy.ndarray
            2D grayscale image array for the current slice (H x W).
    label_slice : numpy.ndarray
            2D integer label array for the current slice (H x W). Non-zero values are
            considered tissue/structures that can be re-assigned to branch (1) or
            lumen (2).
    predictor : object
            SAM predictor instance providing:
            - set_image(image_rgb) to set the image,
            - predict(point_coords, point_labels, multimask_output=True) to return
                (masks, scores, ...) where masks is an array-like of binary masks and
                scores (sam_quality) is an array-like of quality values.
    lumen_classification_model : object
            Classifier for cropped mask images. Must implement:
            - predict(cropped_mask_image) -> (pred_class, confidence)
            Where pred_class is an integer code (0 for lumen, 1 for branch, 2 for other)
            and confidence is a float confidence score.
    sam_quality_threshold : float
            Minimum SAM quality score to consider a mask for classification.

    Returns
    -------
    numpy.ndarray
            A 2D label array (same shape as label_slice) where lumen pixels are set to
            2, branch pixels to 1, and background to 0. In cases where SAM/resnet-based
            postprocessing is deemed unreliable, the original label_slice is returned
            unchanged.

    Notes
    -----
    - The behavior and thresholds (SAM quality > 0.1 to classify, class-1 quality
        threshold 0.5) are tuned heuristics that can be adjusted if needed.


    """
    # Segment using SAM2
    h, w = image_slice.shape
    image_slice_rgb = grey2rgb(image_slice)
    predictor.set_image(image_slice_rgb)
    sam_point = np.array([[h // 2, w // 2]])
    sam_label = np.array([1])
    sam_mask, sam_quality, _ = predictor.predict(
        point_coords=sam_point,
        point_labels=sam_label,
        multimask_output=True,
    )

    mask_img1 = image_slice * sam_mask[0]
    mask_img2 = image_slice * sam_mask[1]
    mask_img3 = image_slice * sam_mask[2]

    # crop to bounding box
    min_x, min_y, max_x, max_y = ski.measure.regionprops(sam_mask[0].astype(int))[
        0
    ].bbox
    mask_img1 = mask_img1[min_x:max_x, min_y:max_y]

    min_x, min_y, max_x, max_y = ski.measure.regionprops(sam_mask[1].astype(int))[
        0
    ].bbox
    mask_img2 = mask_img2[min_x:max_x, min_y:max_y]

    min_x, min_y, max_x, max_y = ski.measure.regionprops(sam_mask[2].astype(int))[
        0
    ].bbox
    mask_img3 = mask_img3[min_x:max_x, min_y:max_y]

    label_with_lumen = np.zeros_like(label_slice, dtype=np.uint8)
    preds = []
    mask_imgs = [mask_img1, mask_img2, mask_img3]

    for j, mask_img in enumerate(mask_imgs):
        # Only classify if SAM mask quality is above threshold
        if sam_quality[j] > sam_quality_threshold:
            pred_class, conf = lumen_classification_model.predict(mask_img)
            preds.append(
                {
                    "index": j,
                    "class": pred_class,
                    "conf": conf,
                    "quality": sam_quality[j],
                }
            )
        else:
            logger.info(
                f"Skipping SAM mask {j},\n" f"due to low quality ({sam_quality[j]:.2f})"
            )

    # If no good-quality masks, fall back to original segmentation
    if len(preds) == 0:
        logger.info("No SAM masks with quality > 0.5," "using original segmentation.")
        label_with_lumen = label_slice.copy()
    else:
        logger.info([p["class"] for p in preds])

        # Handle lumen (class 0)
        lumen_preds = [p for p in preds if p["class"] == 0]
        if lumen_preds:
            logger.info("Found lumen")
            best_lumen = max(lumen_preds, key=lambda x: x["quality"])
            label_with_lumen[sam_mask[best_lumen["index"]] == 1] = 2

        # Handle branches (class 1)
        class1_preds = [p for p in preds if p["class"] == 1]
        if class1_preds:
            best_class1 = max(class1_preds, key=lambda x: x["quality"])
            if best_class1["quality"] < 0.5:
                logger.info(
                    "Low quality for class 1 mask,"
                    "assigning class 1 to original segmentation"
                )
                mask = (label_slice != 0) & (label_with_lumen == 0)
                label_with_lumen[mask] = 1
            else:
                mask = (sam_mask[best_class1["index"]] == 1) & (label_with_lumen == 0)
                label_with_lumen[mask] = 1
        else:
            # If no class 1 was found, use original segmentation
            label_with_lumen[(label_slice != 0) & (label_with_lumen == 0)] = 1

        # Handle case where all are bad (class 2)
        if all(p["class"] == 2 for p in preds):
            label_with_lumen = label_slice.copy()
        if lumen_touches_background(label_with_lumen):
            logger.info(
                "Lumen touches background," "reverting to original segmentation."
            )
            label_with_lumen = label_slice.copy()

    label_slice = label_with_lumen
    return label_slice 


def lumen_touches_background(label_slice):
    """Check if lumen touches background in a label slice.

    Parameters
    ----------
    label_slice : numpy.ndarray
        2D array representing the label slice.

    Returns
    -------
    bool
        True if lumen touches background, False otherwise.

    """
    lumen_label = 2
    background_label = 0
    lumen_mask = label_slice == lumen_label
    background_mask = label_slice == background_label
    # Dilate lumen mask to ensure touching
    dilated_lumen = dilation(lumen_mask, footprint=disk(1))
    touching = np.any(dilated_lumen & background_mask)
    return touching


def filter_for_iterative_lumens(data_path, save_path):
    """Filter for iterative lumens across multiple files in parallel.

    This function processes HDF5 files in the specified data path, filtering out slices
    that do not contain the iterative lumen label (2). It removes slices that are not
    part of the iterative lumen by checking if the label 2 is present in the
    segmentation slices. If a slice does not contain label 2, it checks the neighboring
    slices to determine if it should be removed.
    The filtered slices are saved to the specified save path.

    Parameters
    ----------
    data_path : str
        Path to the input data directory containing .h5 files.
    save_path : str
        Path to the output directory where filtered .h5 files will be saved.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"Created directory: {save_path}")

    files = [f for f in os.listdir(data_path) if f.endswith(".h5")]
    logger.info(f"Found {len(files)} files to process.")
    files = tqdm(files, desc="Processing files")

    # Pack arguments for parallel processing
    file_args = [(f, data_path, save_path) for f in files]

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        list(executor.map(filter_file_for_iterative_lumens, file_args))


def filter_file_for_iterative_lumens(args):
    """Filter a single HDF5 file for iterative lumens.

    This function processes a single HDF5 file, filtering out slices that do not
    contain the iterative lumen label (2). It checks each slice in the segmentation
    data and removes slices that do not contain label 2, or are not surrounded by
    label 2 slices. The filtered slices are saved to the specified output path.


    Parameters
    ----------
    args : tuple
        A tuple containing the file name, data path, and save path.
        The tuple should be in the format (file_name, data_path, save_path).

    """
    file, data_path, save_path = args
    logger.info(f"Processing {file}")

    input_file_path = os.path.join(data_path, file)
    output_file_path = os.path.join(save_path, file)
    try:
        with h5py.File(input_file_path, "r") as f:
            image_slices = f["image"][:]
            segmentation_slices = f["segmentation"][:]
    except Exception as e:
        logger.warning(f"Error loading {file}: {e}")
        return

    if np.sum(segmentation_slices == 2) == 0:
        logger.info(f"No label 2 in file {file}, skipping.")
        return

    index_to_remove = []
    for i, label_slice in enumerate(segmentation_slices):
        if i == 0 or i == len(segmentation_slices) - 1:
            continue

        if np.sum(label_slice == 2) == 0:
            if (
                np.sum(segmentation_slices[i - 1] == 2) > 0
                and np.sum(segmentation_slices[i + 1] == 2) > 0
            ):
                index_to_remove.append(i)

    if not index_to_remove:
        logger.info(f"No slices to remove in file {file}")
        return

    image_slice_filt = np.delete(image_slices, index_to_remove, axis=0)
    label_slices_filt = np.delete(segmentation_slices, index_to_remove, axis=0)

    with h5py.File(output_file_path, "w") as f:
        f.create_dataset("image", data=image_slice_filt)
        f.create_dataset("segmentation", data=label_slices_filt)
    logger.info(f"Filtered and saved {file}")


def fix_only_lumen(segmentation_slice):
    """Fix segmentation slices containing only lumen."""
    boundary_lumen = set(
        map(
            tuple,
            np.round(
                np.concatenate(ski.measure.find_contours(segmentation_slice == 2, 0.5))
            ).astype(np.int32),
        )
    )
    boundary_background = set(
        map(
            tuple,
            np.round(
                np.concatenate(ski.measure.find_contours(segmentation_slice == 0, 0.5))
            ).astype(np.int32),
        )
    )
    return bool(boundary_lumen & boundary_background)


def add_file_to_graph(file):
    """Process a single HDF5 file."""
    try:
        with h5py.File(file, "r") as f:
            image_slices = f["image"][:]
            segmentation_slices = f["segmentation"][:]
            pixel_spacing_um = float(f.attrs.get("sample_grid_spacing_um", 1.0))
    except Exception as e:
        logger.warning(f"Error loading {file}: {e}")
        return None

    file_name = os.path.basename(file)
    start_node = int(file_name.split("sn_")[1].split("_")[0])
    end_node = int(file_name.split("en_")[1].split(".")[0])

    tissue_radius_branch = []
    lumen_radius_branch = []
    minor_axis_branch = []
    major_axis_branch = []
    total_area_branch = []

    for slice_index, (_, segmentation_slice) in enumerate(
        zip(image_slices, segmentation_slices, strict=False)
    ):
        if np.sum(segmentation_slice == 2) > 0:
            if np.sum(segmentation_slice == 1) == 0 and fix_only_lumen(
                segmentation_slice
            ):
                logger.info(f"Fixing {file}, slice {slice_index}")
                segmentation_slice[segmentation_slice == 2] = 1
                segmentation_slices[slice_index] = segmentation_slice

            label_slice = ski.measure.label((segmentation_slice != 0).astype(np.uint8))
            props = ski.measure.regionprops(label_slice)

            if props:
                minor_axis = props[0].minor_axis_length
                major_axis = props[0].major_axis_length
                total_area = props[0].area
                minor_axis_branch.append(minor_axis)
                major_axis_branch.append(major_axis)
                total_area_branch.append(total_area)

            if np.sum(segmentation_slice == 1) > 0:
                lumen_label = (segmentation_slice == 2).astype(np.uint8)
                tissue_label = (segmentation_slice == 1).astype(np.uint8)

                lumen_props = ski.measure.regionprops(ski.measure.label(lumen_label))
                tissue_props = ski.measure.regionprops(ski.measure.label(tissue_label))

                if lumen_props and tissue_props:
                    lumen_area = lumen_props[0].area
                    tissue_area = tissue_props[0].area
                    total_area = lumen_area + tissue_area
                    total_radius = radius_from_area(total_area)
                    lumen_radius = radius_from_area(lumen_area)
                    tissue_radius = total_radius - lumen_radius
                    tissue_radius_branch.append(tissue_radius)
                    lumen_radius_branch.append(lumen_radius)
            else:
                # no tissue label, full region is tissue
                tissue_radius_branch.append(minor_axis / 2)
                lumen_radius_branch.append(0)
        else:
            # completely closed (no lumen)
            label_slice = ski.measure.label((segmentation_slice != 0).astype(np.uint8))
            props = ski.measure.regionprops(label_slice)
            if props:
                minor_axis = props[0].minor_axis_length
                major_axis = props[0].major_axis_length
                total_area = props[0].area
                minor_axis_branch.append(minor_axis)
                major_axis_branch.append(major_axis)
                total_area_branch.append(total_area)
                tissue_radius_branch.append(minor_axis / 2)
                lumen_radius_branch.append(0)

    lumen_diameter = np.array(lumen_radius_branch) * 2 * pixel_spacing_um
    tissue_thickness = np.array(tissue_radius_branch) * pixel_spacing_um
    total_area = np.array(total_area_branch) * pixel_spacing_um**2
    minor_axis = np.array(minor_axis_branch) * pixel_spacing_um
    major_axis = np.array(major_axis_branch) * pixel_spacing_um

    return (
        start_node,
        end_node,
        np.mean(lumen_diameter),
        np.std(lumen_diameter),
        np.mean(tissue_thickness),
        np.std(tissue_thickness),
        np.mean(total_area),
        np.std(total_area),
        np.mean(minor_axis),
        np.std(minor_axis),
        np.mean(major_axis),
        np.std(major_axis),
    )


def add_measurements_from_h5_to_graph(graph_path, input_path):
    """
    Add measurements from HDF5 files to the skeleton graph.

    The slice names need to be in the format:

    {base}_{name}_{start_node}_{end_node}.h5

    Valid files are usually generated using the sample_slices_from_graph function from
    the SkeletonGraph class. Its highly advised to use the filtering
    functions first.

    Parameters
    ----------
    graph_path : str
        Path to the skeleton graph JSON file.
    input_path : str
        Path to the directory containing HDF5 files with the segmented slices.

    Returns
    -------
    SkeletonGraph
        The updated skeleton graph with measurements added.
    """
    # Load skeleton graph
    skeleton_graph = SkeletonGraph.from_json_file(graph_path)

    # Prepare attributes
    attribute_names = [
        "lumen_diameter",
        "tissue_thickness",
        "total_area",
        "minor_axis",
        "major_axis",
    ]
    for attr in attribute_names + [f"{a}_sd" for a in attribute_names]:
        nx.set_edge_attributes(skeleton_graph.graph, {}, name=attr)

    measurement_dicts = {
        key: {}
        for key in (
            "lumen_diameter",
            "tissue_thickness",
            "total_area",
            "minor_axis",
            "major_axis",
            "lumen_diameter_sd",
            "tissue_thickness_sd",
            "total_area_sd",
            "minor_axis_sd",
            "major_axis_sd",
        )
    }

    # Get list of HDF5 files
    files = [f for f in os.listdir(input_path) if f.endswith(".h5")]
    # add input path to files
    files = [os.path.join(input_path, f) for f in files]

    # Process files in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        for result in tqdm(executor.map(add_file_to_graph, files), total=len(files)):
            if result:
                results.append(result)

    # Fill measurement dicts
    for (
        start_node,
        end_node,
        lumen_diameter_mean,
        lumen_diameter_sd,
        tissue_thickness_mean,
        tissue_thickness_sd,
        total_area_mean,
        total_area_sd,
        minor_axis_mean,
        minor_axis_sd,
        major_axis_mean,
        major_axis_sd,
    ) in results:
        edge = (start_node, end_node)
        measurement_dicts[DIAMETER_KEY][edge] = lumen_diameter_mean
        measurement_dicts[f"{DIAMETER_KEY}_sd"][edge] = lumen_diameter_sd
        measurement_dicts[TISSUE_THICKNESS_KEY][edge] = tissue_thickness_mean
        measurement_dicts[f"{TISSUE_THICKNESS_KEY}_sd"][edge] = tissue_thickness_sd
        measurement_dicts["total_area"][edge] = total_area_mean
        measurement_dicts["total_area_sd"][edge] = total_area_sd
        measurement_dicts["minor_axis"][edge] = minor_axis_mean
        measurement_dicts["minor_axis_sd"][edge] = minor_axis_sd
        measurement_dicts["major_axis"][edge] = major_axis_mean
        measurement_dicts["major_axis_sd"][edge] = major_axis_sd

    # Set graph attributes
    for attr, attr_dict in measurement_dicts.items():
        nx.set_edge_attributes(skeleton_graph.graph, attr_dict, name=attr)
    logger.info("save")
    # Save
    skeleton_graph.to_json_file(graph_path)

    return skeleton_graph
