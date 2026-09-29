# batch_analysis

End-to-end batch pipeline from raw image volumes to per-branch measurements:
convert acquisitions, train a segmentation model, segment a folder of samples,
curate the results, build and curate skeleton graphs, and measure them.

## Configuring a dataset

Defaults live in [`_constants.py`](_constants.py). Rather than editing that file
per dataset, write a small config that assigns only what differs and point
`BATCH_ANALYSIS_CONFIG` at it:

```bash
export BATCH_ANALYSIS_CONFIG=/path/to/my_dataset/config.py
python segment_batch.py --channel ssh
```

Every UPPERCASE name the config defines replaces the default. Paths derived from
`PROJECT_ROOT` are re-derived from the config's `PROJECT_ROOT` unless the config
sets them explicitly. [`configs/example_config.py`](configs/example_config.py)
is a commented starting point.

Scripts that expose command-line flags fall back to the config for defaults, so
a flag only ever changes a single run.

## Data layout

Two layouts are supported, selected with `LAYOUT`. Discovery for both lives in
[`_layout.py`](_layout.py) — scripts never glob the filesystem themselves, which
is what keeps the exclusion rules and the channel priority consistent across
stages.

**`LAYOUT = "flat"`** — one project, one directory per stage:

```text
PROJECT_ROOT/
  zarr/            <name>.zarr groups: image (+ optional label), voxel_size_um attr
  patches_<ch>/    training/ and validation/ HDF5 patches
  graphs/          <name>_graph.json  (raw skeleton graphs)
  graphs_fixed/    curated / directed graphs (review_graphs.py)
  graphs_final/    measured graphs + cached slices (analyze_graphs_batch.py)
  csvs/            per-sample and combined measurement tables
```

**`LAYOUT = "nested"`** — a treatment screen: several `EXPERIMENT_ROOTS`, each
holding condition directories that carry their own derived outputs:

```text
<root>/<condition>/
  *.czi, *.h5                  raw acquisitions
  iso35/*.h5                   isotropic + normalised  (ISO_SUBDIR)
  segmentation_<channel>/<stem>.zarr
  graphs/, graphs_fixed/, graphs_final/
```

Conditions may sit several levels below a root; discovery is always recursive.
Measurement CSVs go to one shared `CSVS_DIR` in both layouts.

## Pipeline

| Step | Script | In → Out |
|------|--------|----------|
| 0a | `czi_to_h5.py` | `.czi` → per-channel HDF5 |
| 0b | `preprocess_to_isotropic.py` | HDF5 → isotropic, normalised HDF5 |
| 1 | `prepare_patches.py` | curated volumes → HDF5 training patches |
| 2 | `train_dynunet.py` | patches → segmentation checkpoint |
| 3 | `segment_batch.py` | images → `segmentation` arrays in zarr |
| 3b | `add_channel_to_zarrs.py` | extra image channels into those zarrs (nested only) |
| 4 | `review_segmentations.py` | napari QC, component curation, bad-quality rejection |
| 5 | `segmentation_to_graph_batch.py` | segmentation → skeleton → `graphs/<name>_graph.json` |
| 6 | `review_graphs.py` | set origin, keep main component, make directed → `graphs_fixed/` |
| 7 | `analyze_graphs_batch.py` | curated graphs → measurement CSVs |

Steps 0a/0b are only needed when starting from raw CZI acquisitions; a dataset
that already has isotropic zarr containers starts at step 1.

Supporting modules, not run directly:

- `_constants.py` — defaults and per-dataset config loading
- `_layout.py` — what to process and where its outputs go
- `exclusions.py` — which files the pipeline must skip
- `normalization.py` — the intensity normalisation shared by training and inference
- `inference.py` — sliding-window segmentation for a trained checkpoint

Step 5 needs a separate normal-field skeleton-prediction model
(`SKELETON_CHECKPOINT`), distinct from the segmentation models.

## Channels

A sample can be imaged on several channels (e.g. `ssh` and `dapi`). Each is
segmented by its own checkpoint into its own `segmentation_<channel>/` folder.

The steps that want ONE result per acquisition — review, graph building,
analysis — take the first channel in `CHANNEL_PRIORITY` that has a segmentation.
Without that, a sample segmented on both channels would contribute two graphs of
the same acquisition and double-count in the results.

`SEG_CHECKPOINTS` and `TRAIN_CROP` are keyed by channel and must be changed
together: DynUNet uses InstanceNorm, whose statistics span the whole sliding
window, so inferring at a window size other than the one a checkpoint was
trained at shifts every normalisation statistic in the network and degrades the
segmentation silently. `inference_kwargs(channel)` builds the roi_size from
`TRAIN_CROP`, so the two cannot drift apart.

## Excluding samples

Samples curated out live in directories named `not_processed`, `bad_quality` and
similar, and [`exclusions.py`](exclusions.py) is the single place that decides
what those names are. Every discovery path runs through it.

The names on disk are inconsistent (`no processed`, with a space and no "t"), so
every path part is normalised before comparing — a literal
`"not_processed" in path.parts` matches nothing and lets rejected samples through.
That has gone wrong twice; see the module docstring.

"Mark bad quality" in `review_segmentations.py` moves both the zarr **and** the
preprocessed HDF5 it came from, because moving the zarr alone would let the next
`segment_batch.py` run regenerate it and silently undo the rejection.

## Adapting to a new dataset

- Copy `configs/example_config.py`, set `LAYOUT`, `PROJECT_ROOT` (or
  `EXPERIMENT_ROOTS`), `CHANNELS`, `SEG_CHECKPOINTS` / `TRAIN_CROP` and
  `DEFAULT_VOXEL_SIZE_UM`.
- For sparse volumes (mostly background) keep `CROP_ENABLED = True`; for volumes
  already framed around the sample set it to `False` (or pass `--no-crop`).
- List your hand-curated training volumes in `PATCH_SOURCES`. Model output does
  not belong there — it teaches the new network the old one's mistakes.
- To record experiment metadata (genotype, condition, timepoint) in the CSVs,
  set `SAMPLE_METADATA` to a function of `(stem, sample)`; see
  `analyze_graphs_batch.sample_metadata` for a worked example.
