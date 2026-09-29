# batch_analysis

End-to-end batch pipeline from raw image volumes to per-branch measurements:
train a segmentation model, segment a folder of samples, curate the results,
build and curate skeleton graphs, and measure them.

All shared configuration (paths, model checkpoints, patch/inference/measurement
parameters) lives in [`_constants.py`](_constants.py). Edit the values marked
`# ADAPT HERE` before running. Scripts that expose command-line flags fall back
to these constants as defaults.

## Data layout

Everything hangs off a single `PROJECT_ROOT`:

```text
PROJECT_ROOT/
  zarr/            <name>.zarr groups: image (+ optional label), voxel_size_um attr
  patches/         training/ and validation/ HDF5 patches
  graphs/          <name>_graph.json  (raw skeleton graphs)
  graphs_fixed/    curated / directed graphs (review_graphs.py)
  graphs_final/    measured graphs + cached slices (analyze_graphs_batch.py)
  csvs/            per-sample and combined measurement tables
```

## Pipeline

| Step | Script | In → Out |
|------|--------|----------|
| 1 | `prepare_patches.py` | annotated zarr → HDF5 training patches |
| 2 | `train_dynunet.py` | patches → segmentation checkpoint |
| 3 | `segment_batch.py` | zarr images → `segmentation` arrays |
| 4 | `review_segmentations.py` | napari QC / component curation of segmentations |
| 5 | `segmentation_to_graph_batch.py` | segmentation → skeleton → `graphs/<name>_graph.json` |
| 6 | `review_graphs.py` | set origin, keep main component, make directed → `graphs_fixed/` |
| 7 | `analyze_graphs_batch.py` | curated graphs → measurements CSVs |

`inference.py` is a helper module (sliding-window segmentation) imported by
`segment_batch.py` and `review_segmentations.py`; it is not run directly.

Step 5 needs a separate normal-field skeleton-prediction model
(`SKELETON_CHECKPOINT`), distinct from the segmentation model.

## Adapting to a new dataset

- Set `PROJECT_ROOT`, `SEG_CHECKPOINT`, `IMAGE_KEY` / `LABEL_KEY`, and
  `DEFAULT_VOXEL_SIZE_UM` in `_constants.py`.
- For sparse volumes (mostly background), keep `CROP_ENABLED = True`; for dense
  volumes set it to `False` (or pass `--no-crop` to `segment_batch.py`).
- To record experiment metadata (genotype, condition, ...) in the output CSVs,
  edit `sample_metadata()` in `analyze_graphs_batch.py`.
