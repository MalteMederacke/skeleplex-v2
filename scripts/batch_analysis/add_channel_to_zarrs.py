"""Add extra image channels to existing segmentation zarrs, in place.

Segmentation is run per channel, so a ``segmentation_ssh`` zarr holds only the
SSH image. Reviewing a sample is much easier with its other channels beside it,
and ``analyze_graphs_batch.py`` can sample slices from whichever channel shows
the lumen best. This copies those channels in from the preprocessed HDF5 the
zarr was segmented from.

Also normalises the legacy layout: containers written by older versions stored
the image under ``image`` rather than under its channel name, which makes a
container ambiguous once a second channel is added. Those are renamed.

Resulting container::

    <condition>/segmentation_ssh/<stem>.zarr/
      ssh/           float32  (renamed from 'image' if needed)
      dapi/          float32  (copied from <condition>/<ISO_SUBDIR>/<stem>.h5)
      segmentation/  uint8
      .zattrs: {voxel_size_um: [z, y, x]}

Idempotent: channels already present are left alone, so re-running is free.
Only meaningful in the nested layout — in the flat layout every channel already
lives in the same container.
"""

import argparse
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent))
from _constants import CHANNEL_PRIORITY, CHANNELS, IMAGE_KEY, LAYOUT  # noqa: E402
from _layout import find_segmented, iso_h5_for, rel  # noqa: E402


def _chunks(shape, base=(64, 256, 256)):
    return tuple(min(b, s) for b, s in zip(base, shape))


def rename_legacy_image(store, channel):
    """Rename a legacy ``image`` array to its channel name. True if renamed."""
    if channel in store:
        return False
    if IMAGE_KEY not in store:
        raise KeyError(
            f"container has neither '{channel}' nor '{IMAGE_KEY}'; "
            f"has {sorted(store.array_keys())}"
        )
    src = store[IMAGE_KEY]
    store.create_array(channel, data=src[:], chunks=src.chunks, overwrite=True)
    del store[IMAGE_KEY]
    return True


def add_channels(sample, extra_channels):
    """Copy ``extra_channels`` from the sample's HDF5 into its zarr container."""
    store = zarr.open(str(sample.zarr_path), mode="r+")

    if rename_legacy_image(store, sample.channel):
        print(f"  renamed '{IMAGE_KEY}' -> '{sample.channel}'")

    missing = [ch for ch in extra_channels if ch not in store]
    if not missing:
        print(f"  all of {list(extra_channels)} already present")
        return

    h5_path = iso_h5_for(sample.zarr_path)
    if not Path(h5_path).exists():
        raise FileNotFoundError(f"source HDF5 not found: {h5_path}")

    ref_shape = tuple(store[sample.channel].shape)
    with h5py.File(h5_path, "r") as f:
        for ch in missing:
            if ch not in f:
                print(f"  [skip] no '{ch}' channel in {Path(h5_path).name}")
                continue
            data = f[ch][:].astype(np.float32)
            if data.shape != ref_shape:
                # Not fatal: the arrays are viewed side by side, and a mismatch
                # is worth seeing rather than silently dropping.
                print(
                    f"  WARN: {ch} shape {data.shape} != "
                    f"{sample.channel} {ref_shape}; adding anyway"
                )
            store.create_array(ch, data=data, chunks=_chunks(data.shape))
            print(f"  added '{ch}'")


def main(into=None, extra_channels=None) -> None:
    """Add extra channels to every segmentation zarr of the ``into`` channel.

    Parameters
    ----------
    into : str, optional
        Which segmentation_<channel> containers to augment. Defaults to the
        first entry of CHANNEL_PRIORITY.
    extra_channels : sequence of str, optional
        Channels to copy in. Defaults to every configured channel except
        ``into``.
    """
    if LAYOUT != "nested":
        print(
            "LAYOUT is 'flat': every channel already lives in the same "
            "container, so there is nothing to add."
        )
        return

    into = into or CHANNEL_PRIORITY[0]
    if extra_channels is None:
        extra_channels = tuple(ch for ch in CHANNELS if ch != into)
    if not extra_channels:
        print(f"No channels to add besides {into!r}.")
        return

    samples = [s for s in find_segmented(channels=(into,)) if s.channel == into]
    if not samples:
        print(f"No segmentation_{into} zarrs found.")
        sys.exit(1)

    print(
        f"Found {len(samples)} segmentation_{into} zarrs; "
        f"adding {list(extra_channels)}.\n"
    )
    errors = []
    for sample in samples:
        print(rel(sample.zarr_path))
        try:
            add_channels(sample, extra_channels)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            errors.append((sample, e))

    print(f"\nDone. {len(samples) - len(errors)}/{len(samples)} containers updated.")
    if errors:
        print("Errors:")
        for s, e in errors:
            print(f"  {rel(s.zarr_path)}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--into", choices=list(CHANNELS),
        help="Which segmentation_<channel> containers to augment.",
    )
    parser.add_argument(
        "--add", action="append", dest="extra_channels", choices=list(CHANNELS),
        help="Channel to copy in (repeatable). Defaults to all others.",
    )
    args = parser.parse_args()
    main(into=args.into, extra_channels=args.extra_channels)
