# SkelePlex v2

[![License](https://img.shields.io/pypi/l/skeleplex-v2.svg?color=green)](https://github.com/kevinyamauchi/skeleplex-v2/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/skeleplex-v2.svg?color=green)](https://pypi.org/project/skeleplex-v2)
[![Python Version](https://img.shields.io/pypi/pyversions/skeleplex-v2.svg?color=green)](https://python.org)
[![CI](https://github.com/kevinyamauchi/skeleplex-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/kevinyamauchi/skeleplex-v2/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kevinyamauchi/skeleplex-v2/branch/main/graph/badge.svg)](https://codecov.io/gh/kevinyamauchi/skeleplex-v2)

**A Python toolkit for analyzing complex biological network structures**

## Work in progress
 <ins>**This is a work-in-progress re-write of SkelePlex.You may encounter bugs or broken features.**<ins>

## 
SkelePlex is a comprehensive image analysis tool designed to extract quantitative measurements from complex biological networks while preserving their morphological context. It specializes in analyzing branching structures such as:

-  **Bronchial trees** - Respiratory system analysis
-  **Vasculature** - Blood vessel networks  
-  **Axonal networks** - Neural connectivity
-  **Xylem structures** - Plant vascular systems

## Key Features

- **Hierarchical graph construction** from segmentation images
- **Interactive curation** with custom viewer tools
- **Comprehensive measurements**: branch length, angles, generation, diameter, tissue thickness
- **Scalable processing** with parallelized, lazy operations for large datasets
- **Multi-format support** including `.zarr` for out-of-core computation

## Table of Contents

- [Installation](#installation)
- [Workflow Overview](#workflow-overview)
- [Examples](#examples)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

![SkelePlex Pipeline](docs/images/skeleplex_figure.png)

## Installation

### Basic Installation

```bash
pip install "git+https://github.com/kevinyamauchi/skeleplex-v2.git#egg=skeleplex-v2[cellier,viz]"
```


### System Requirements

- Python 3.11+ (3.12+ for the viewer application)
- For large datasets: Sufficient RAM or `.zarr` format support
- For the neural networks access to a GPU is recommended
- For interactive viewing: OpenGL-compatible graphics


## Workflow Overview

### 1. Segmentation Input

Start with a semantic segmentation where `1` represents foreground (network) and `0` represents background. SkelePlex supports various formats including `.zarr` for out-of-memory computations.

<p align="center">
<img src="docs/images/e13_segmentation.png" alt="Input Segmentation" width="200"/>
<br><i>Input: Binary segmentation of biological network</i>
</p>

### 2. Distance Field Computation

Transform the segmentation into a normalized distance field using a spherical max distance kernel. Each voxel value represents the normalized distance to the background.

<p align="center">
<img src="docs/images/E13_distance_field.png" alt="Distance Field" width="200"/>
<br><i>Distance field representation</i>
</p>

### 3. Skeleton Prediction

Apply the multiscale SkelePlex UNet to predict skeleton centerlines from the distance field image.

<p align="center">
<img src="docs/images/E13_rescaled_from_scale1.png" alt="Predicted Skeleton" width="200"/>
<br><i>AI-predicted skeleton centerlines</i>
</p>

### 4. Graph Construction & Curation

- Construct hierarchical graph with nodes at branch points
- Fit splines to connecting edges
- Use the interactive viewer for manual curation and quality control

<p align="center">
<img src="docs/images/E13_final_skeleton_fused_scales.png" alt="Final Graph" width="200"/>
<br><i>Final curated skeleton graph</i>
</p>

### 5. Data Analysis

There are many ways to analyse the graphs. We have implemented:
 - Measuring of branch length, diameter, branch angle and number of tips connected to the branch
 
 To measure branch diameters we:

 - Extract orthogonal image sections along each branch
 - Process and filter these image sections
 - Extract the section area and determine branch thickness
 - An automatized way to differentiate between the branch lumen and its surrounding wall thickness if the tissue and not the lumen was segmented

## Examples


We provide a couple of examples on how to use SkelePlex, see the [`examples/`](examples/) directory:

### Segmentation to graph
This is the best starting point.
- [`Create a toy segmentation`](examples/segmentation_to_graph/create_example_data.py) 
- [`Segmentation to graph in memory`](examples/segmentation_to_graph/segmentation_to_graph_memory.py) 
- [`Segmentation to graph lazily`](docs/segmentation_to_image_lazy.md) 


### Viewer app to inspect and curate graphs
- [`Basic viewer usage`](examples/viewer/basic_viewer_usage.py) - Usage of the skeleton viewer
- [`Add your custom widget`](examples/viewer/add_custom_widget.py)

### Measure all sorts of network properties
- [`Extract orthogonal slices along each branch`](examples/measurements/extract_orthogonal_slices.py) - Important to measure branch diameters
- [`Measure the branch diameters`](examples/measurements/extract_orthogonal_slices.py)
- [`Measure more and export`](examples/measurements/measure_and_export.py)

### Train your lumen classifier
If you segmented the tissue of a branch and are interested to measure tissue thickness and the lumen diameter, you can train a lumen classifier to automate that step. 

- [`Create trainings data from orthogonal slices`](examples/measurements/lumen_classifier/create_lumen_classifier_trainings_data.py)
- [`Set up trainings data for training`](examples/measurements/lumen_classifier/create_lumen_classifier_trainings_data.py)
- [`Train your classifier`](examples/measurements/lumen_classifier/train_lumen_classifier.py)



## Development

### Development Installation

1. **Fork and Clone**
   ```bash
   git clone https://github.com/kevinyamauchi/skeleplex-v2.git
   cd skeleplex-v2
   ```

2. **Create Environment** (recommended)
   ```bash
   conda create -n skeleplex python=3.12
   conda activate skeleplex
   ```

3. **Install in Development Mode**
   ```bash
   pip install -e ".[dev-all]"
   ```

4. **Set up Pre-commit Hooks** to apply linting
   ```bash
   pre-commit install
   ```
### Running Tests

```bash
# Run all tests
pytest


# Run specific test module
pytest tests/graph/test_skeleton_graph.py
```

## Technical Notes

### Coordinate System
- All coordinates are transformed to physical units (µm)
- Input images should be isotropic for optimal results

### Platform Differences
For computing signed distances to mesh surfaces:
- **Unix/Linux/macOS**: Uses `pysdf` (better performance)
- **Windows**: Uses `igl` (consistent results)

Both approaches yield equivalent results and only affect synthetic data generation.

## Citation

If you use SkelePlex in your research, please cite:

```bibtex
@article{mederacke2025emergence,
  title={The emergence of the fractal bronchial tree},
  author={Mederacke, Malte and Yamauchi, Kevin A and Doumpas, Nikolaos and Schaumann, Laura and Sperl, Jonathan and Weikert, Thomas and Dane, D Merrill and Hsia, Connie CW and Pradella, Maurice and Bremerich, Jens and others},
  journal={BioRxiv},
  pages={2025--01},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/KevinYamauchi/skeleplex-v2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/KevinYamauchi/skeleplex-v2/discussions)
- **Email**: [malte.mederacke@ethz.ch]

---
