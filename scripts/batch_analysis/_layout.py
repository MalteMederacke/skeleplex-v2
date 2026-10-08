"""Where the pipeline's files live on disk, for both supported layouts.

Every script asks this module what to process instead of globbing itself. That
is what lets one code base serve two very different directory structures, and
it means the exclusion rules (``exclusions.is_excluded``) are applied in exactly
one place rather than re-implemented per stage.

Layouts
-------
``LAYOUT = "flat"`` — one project, one directory per stage::

    PROJECT_ROOT/
      zarr/          <name>.zarr containers: image (+ label), voxel_size_um attr
      patches/       training/ and validation/ HDF5 patches
      graphs/        <name>_graph.json
      graphs_fixed/  curated / directed graphs
      graphs_final/  measured graphs + cached slices
      csvs/          per-sample and combined measurement tables

``LAYOUT = "nested"`` — a treatment screen: several experiment roots, each with
condition directories that carry their own derived outputs::

    <root>/<condition>/
      *.czi, *.h5              raw acquisitions
      iso35/*.h5               isotropic + normalised (ISO_SUBDIR)
      segmentation_<channel>/<stem>.zarr
      graphs/, graphs_fixed/, graphs_final/

In the nested layout the raw image is read from the iso35 HDF5 and the
segmentation is written to a separate zarr; in the flat layout both live in the
same container. `Sample` hides that difference: ``source``/``source_key`` say
where the image comes from, ``zarr_path`` where the segmentation goes.

Conditions may be nested several levels below a root, so discovery is always
recursive — a depth-1 glob silently misses whole experiments.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _constants import (  # noqa: E402
    CHANNEL_PRIORITY,
    CHANNELS,
    CSVS_DIR,
    EXPERIMENT_ROOTS,
    GRAPHS_DIR,
    GRAPHS_FINAL_DIR,
    GRAPHS_FIXED_DIR,
    IMAGE_KEY,
    ISO_SUBDIR,
    LAYOUT,
    PROJECT_ROOT,
    ZARR_DIR,
)
from exclusions import is_excluded  # noqa: E402

GRAPH_SUFFIX = "_graph.json"


@dataclass(frozen=True)
class Sample:
    """One sample, and everywhere its files live.

    Attributes
    ----------
    stem : str
        Sample name, without extension or ``_graph`` suffix.
    channel : str
        Image channel this sample was segmented from.
    source : Path
        File the raw image is read from — an HDF5 in the nested layout, the
        zarr container itself in the flat layout.
    source_key : str
        Dataset key for the image inside ``source``.
    zarr_path : Path
        Zarr container holding the segmentation.
    sample_dir : Path
        Directory owning this sample's derived outputs (graphs, ...). The
        project root in the flat layout, the condition directory in the nested
        one.
    """

    stem: str
    channel: str
    source: Path
    source_key: str
    zarr_path: Path
    sample_dir: Path

    @property
    def name(self):
        """Human-readable identifier, unique across conditions."""
        if LAYOUT == "flat":
            return self.stem
        return f"{self.sample_dir.name}/{self.stem}"


# ── layout helpers ───────────────────────────────────────────────────────────

def _check_layout():
    if LAYOUT not in ("flat", "nested"):
        raise ValueError(f"LAYOUT must be 'flat' or 'nested', got {LAYOUT!r}")


def experiment_roots():
    """Validated experiment roots (nested layout)."""
    roots = [Path(r) for r in EXPERIMENT_ROOTS]
    if not roots:
        raise ValueError(
            "LAYOUT = 'nested' requires EXPERIMENT_ROOTS to be set in the config"
        )
    missing = [r for r in roots if not r.is_dir()]
    if missing:
        raise FileNotFoundError(
            "EXPERIMENT_ROOTS entries do not exist: "
            + ", ".join(str(m) for m in missing)
        )
    return roots


def source_roots():
    """Roots searched for raw acquisitions by the pre-processing stages.

    The experiment roots in the nested layout, the project root in the flat one.
    """
    _check_layout()
    return experiment_roots() if LAYOUT == "nested" else [PROJECT_ROOT]


def rel(path):
    """Path as a short string for logging, relative to the project when possible."""
    path = Path(path)
    bases = [PROJECT_ROOT] if LAYOUT == "flat" else [Path(r) for r in EXPERIMENT_ROOTS]
    for base in bases:
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def seg_dir_name(channel):
    """Name of the folder holding segmentation zarrs for ``channel``."""
    return f"segmentation_{channel}"


# ── discovery: inputs to segmentation ────────────────────────────────────────

def find_inputs(channel):
    """Samples with a raw image for ``channel``, ready to be segmented.

    Parameters
    ----------
    channel : str
        Channel name, one of CHANNELS.

    Returns
    -------
    list of Sample
        Sorted by source path; curated-out paths already removed.
    """
    _check_layout()
    if channel not in CHANNELS:
        raise KeyError(f"unknown channel {channel!r}; CHANNELS = {CHANNELS}")

    if LAYOUT == "flat":
        samples = []
        for zarr_path in sorted(ZARR_DIR.glob("*.zarr")):
            if not zarr_path.is_dir() or is_excluded(zarr_path):
                continue
            samples.append(
                Sample(
                    stem=zarr_path.stem,
                    channel=channel,
                    source=zarr_path,
                    source_key=IMAGE_KEY,
                    zarr_path=zarr_path,
                    sample_dir=PROJECT_ROOT,
                )
            )
        return samples

    import h5py

    samples = []
    for root in experiment_roots():
        for h5_path in sorted(root.glob(f"**/{ISO_SUBDIR}/*.h5")):
            if is_excluded(h5_path):
                continue
            # Reading the header only; nothing large is loaded here.
            with h5py.File(h5_path, "r") as f:
                if channel not in f:
                    continue
            condition_dir = h5_path.parent.parent
            samples.append(
                Sample(
                    stem=h5_path.stem,
                    channel=channel,
                    source=h5_path,
                    source_key=channel,
                    zarr_path=(
                        condition_dir / seg_dir_name(channel) / f"{h5_path.stem}.zarr"
                    ),
                    sample_dir=condition_dir,
                )
            )
    return samples


# ── discovery: existing segmentations ────────────────────────────────────────

def find_segmented(channels=None):
    """One segmented sample per acquisition, best available channel first.

    A sample segmented on several channels appears once, using the first
    channel in ``channels`` that has a zarr. That keeps the review, graph and
    analysis stages from processing the same acquisition twice.

    Parameters
    ----------
    channels : sequence of str, optional
        Channel preference order. Defaults to CHANNEL_PRIORITY.

    Returns
    -------
    list of Sample
        Sorted by zarr path; curated-out paths already removed.
    """
    _check_layout()
    channels = tuple(channels) if channels is not None else tuple(CHANNEL_PRIORITY)

    if LAYOUT == "flat":
        samples = []
        for zarr_path in sorted(ZARR_DIR.glob("*.zarr")):
            if not zarr_path.is_dir() or is_excluded(zarr_path):
                continue
            samples.append(
                Sample(
                    stem=zarr_path.stem,
                    channel=channels[0],
                    source=zarr_path,
                    source_key=IMAGE_KEY,
                    zarr_path=zarr_path,
                    sample_dir=PROJECT_ROOT,
                )
            )
        return samples

    by_key = {}
    for channel in channels:
        for root in experiment_roots():
            pattern = f"**/{seg_dir_name(channel)}/*.zarr"
            for zarr_path in sorted(root.glob(pattern)):
                if not zarr_path.is_dir() or is_excluded(zarr_path):
                    continue
                condition_dir = zarr_path.parent.parent
                key = (condition_dir, zarr_path.stem)
                if key in by_key:
                    continue  # a higher-priority channel already claimed it
                by_key[key] = Sample(
                    stem=zarr_path.stem,
                    channel=channel,
                    source=iso_h5_for(zarr_path),
                    source_key=channel,
                    zarr_path=zarr_path,
                    sample_dir=condition_dir,
                )
    return sorted(by_key.values(), key=lambda s: s.zarr_path)


def iso_h5_for(zarr_path):
    """The preprocessed HDF5 a segmentation zarr was produced from.

    Only meaningful in the nested layout; in the flat layout the zarr is the
    source and is returned unchanged.
    """
    zarr_path = Path(zarr_path)
    if LAYOUT == "flat":
        return zarr_path
    return zarr_path.parent.parent / ISO_SUBDIR / f"{zarr_path.stem}.h5"


def image_key_in(store, channel):
    """Array key holding the image inside a segmentation zarr.

    Two conventions exist on disk: ``segment_batch.py`` writes the image under
    the channel name, while older containers (and containers written before
    ``add_channel_to_zarrs.py`` ran) call it ``image``. Handle both rather than
    assuming, so a container written by either version still opens.
    """
    for key in (channel, IMAGE_KEY, *CHANNELS):
        if key in store:
            return key
    raise KeyError(
        f"no image array in zarr; has {sorted(store.array_keys())}"
    )


# ── derived output directories ───────────────────────────────────────────────

def graphs_dir(sample):
    """Directory for this sample's raw skeleton graph."""
    if LAYOUT == "flat":
        return GRAPHS_DIR
    return sample.sample_dir / "graphs"


def graphs_fixed_dir(sample):
    """Directory for this sample's curated / directed graph."""
    if LAYOUT == "flat":
        return GRAPHS_FIXED_DIR
    return sample.sample_dir / "graphs_fixed"


def graphs_final_dir(sample):
    """Directory for this sample's measured graph and cached slices."""
    if LAYOUT == "flat":
        return GRAPHS_FINAL_DIR
    return sample.sample_dir / "graphs_final"


def csvs_dir():
    """Directory for measurement tables. Shared across conditions in both layouts."""
    return CSVS_DIR


def graph_path(sample):
    """Path of this sample's raw graph JSON."""
    return graphs_dir(sample) / f"{sample.stem}{GRAPH_SUFFIX}"


# ── discovery: graphs ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraphRef:
    """A graph to load, and the sample it belongs to.

    ``path`` is the curated copy in graphs_fixed/ when one exists, otherwise the
    raw graph — so downstream stages always read the best available version.
    """

    path: Path
    sample: Sample
    curated: bool


def find_graphs(channels=None):
    """Every graph on disk, preferring the curated copy of each.

    Returns
    -------
    list of GraphRef
        Sorted by graph path. Samples without a graph are omitted; samples
        whose segmentation zarr has disappeared are still returned, so the
        caller can report them rather than silently dropping them.
    """
    refs = []
    for sample in find_segmented(channels=channels):
        raw = graph_path(sample)
        if not raw.exists():
            continue
        fixed = graphs_fixed_dir(sample) / raw.name
        use_fixed = fixed.exists()
        refs.append(
            GraphRef(
                path=fixed if use_fixed else raw,
                sample=sample,
                curated=use_fixed,
            )
        )
    return sorted(refs, key=lambda r: r.path)
