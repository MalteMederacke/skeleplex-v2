"""Convert CZI z-stacks to per-channel HDF5.

Input:  <root>/**/*.czi
Output: <stem>.h5 next to each .czi, one dataset per recognised channel, each
        carrying a ``voxel_size_um`` (z, y, x) attribute.

Channels are named by illumination wavelength through CZI_CHANNEL_MAP in
``_constants.py`` (405 nm -> dapi, 488 nm -> ssh, ...). A channel whose
wavelength matches nothing is skipped rather than guessed at, so an unexpected
acquisition setting shows up as a missing channel instead of mislabelled data.

Files that already have a .h5 are skipped, so a run only picks up new
acquisitions. Anything under a curated-out directory is skipped entirely
(see ``exclusions.py``).
"""

import argparse
import sys
import traceback
from pathlib import Path

import czifile
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _constants import (  # noqa: E402
    CZI_CHANNEL_MAP,
    CZI_WAVELENGTH_TOLERANCE_NM,
)
from _layout import rel, source_roots  # noqa: E402
from exclusions import is_excluded  # noqa: E402


def get_channel_wavelengths(meta):
    """Illumination wavelength (nm, rounded) per channel, None where absent."""
    img_doc = meta.get("ImageDocument", {})
    meta2 = img_doc.get("Metadata", {})
    dims = meta2.get("Information", {}).get("Image", {}).get("Dimensions", {})
    ch_list = dims.get("Channels", {}).get("Channel", [])
    if isinstance(ch_list, dict):
        ch_list = [ch_list]
    wavelengths = []
    for ch in ch_list:
        wl = ch.get("IlluminationWavelength", {}).get("SinglePeak")
        wavelengths.append(round(wl) if wl is not None else None)
    return wavelengths


def get_voxel_size_um(meta):
    """Voxel size as (z, y, x) in micrometres; entries are None when missing."""
    img_doc = meta.get("ImageDocument", {})
    meta2 = img_doc.get("Metadata", {})
    distances = meta2.get("Scaling", {}).get("Items", {}).get("Distance", [])
    voxel = {}
    for d in distances:
        axis = d.get("Id")
        value_m = d.get("Value")
        if axis and value_m is not None:
            voxel[axis] = value_m * 1e6  # m -> um
    return (voxel.get("Z"), voxel.get("Y"), voxel.get("X"))


def match_key(wavelength, channel_map, tolerance_nm):
    """Channel name for an illumination wavelength, or None if unrecognised."""
    if wavelength is None:
        return None
    for ref, key in channel_map.items():
        if abs(wavelength - ref) <= tolerance_nm:
            return key
    return None


def process_czi(czi_path, channel_map, tolerance_nm):
    """Write one .h5 next to ``czi_path``, one dataset per recognised channel."""
    out_path = czi_path.with_suffix(".h5")

    with czifile.CziFile(str(czi_path)) as czi:
        meta = czi.metadata(raw=False)
        wavelengths = get_channel_wavelengths(meta)
        voxel_zyx = get_voxel_size_um(meta)

        # axes are e.g. 'BVIHRSCTZYX0'; squeeze to (I, C, Z, Y, X), then
        # average over I (the illumination directions of a light sheet).
        data = czi.asarray()
        axes = czi.axes

        has_I = "I" in axes
        keep = {"I", "C", "Z", "Y", "X"} if has_I else {"C", "Z", "Y", "X"}
        squeeze_axes = tuple(i for i, ch in enumerate(axes) if ch not in keep)
        data = data.squeeze(axis=squeeze_axes)
        remaining = [ch for ch in axes if ch in keep]

        if has_I:
            i_ax = remaining.index("I")
            data = data.mean(axis=i_ax, keepdims=False)
            remaining.pop(i_ax)

        c_ax = remaining.index("C")
        n_channels = data.shape[c_ax]

        with h5py.File(out_path, "w") as f:
            written = []
            for c_idx in range(n_channels):
                wl = wavelengths[c_idx] if c_idx < len(wavelengths) else None
                key = match_key(wl, channel_map, tolerance_nm)
                if key is None:
                    print(f"  [skip] channel {c_idx} wavelength={wl} nm — no mapping")
                    continue

                vol = np.take(data, c_idx, axis=c_ax)  # (Z, Y, X)
                ds = f.create_dataset(
                    key, data=vol, compression="gzip", compression_opts=4
                )
                if all(v is not None for v in voxel_zyx):
                    ds.attrs["voxel_size_um"] = np.array(voxel_zyx, dtype=np.float64)
                written.append(f"{key}(lambda={wl}nm)")

            print(f"  saved: {', '.join(written)} -> {out_path.name}")


def find_czi_files(roots):
    """Unconverted .czi files under ``roots``, excluding curated-out paths."""
    czi_files = []
    for root in roots:
        for path in sorted(Path(root).glob("**/*.czi")):
            if is_excluded(path):
                continue
            if path.with_suffix(".h5").exists():
                continue
            czi_files.append(path)
    return sorted(czi_files)


def main(roots=None, channel_map=None, tolerance_nm=None) -> None:
    """Convert every unconverted CZI under ``roots`` to HDF5.

    Parameters
    ----------
    roots : sequence of Path, optional
        Directories searched recursively. Defaults to the layout's source roots.
    channel_map : dict, optional
        Illumination wavelength (nm) -> channel name. Defaults to
        CZI_CHANNEL_MAP.
    tolerance_nm : int, optional
        Wavelength match tolerance. Defaults to CZI_WAVELENGTH_TOLERANCE_NM.
    """
    roots = list(roots) if roots else source_roots()
    channel_map = channel_map if channel_map is not None else CZI_CHANNEL_MAP
    tolerance_nm = (
        tolerance_nm if tolerance_nm is not None else CZI_WAVELENGTH_TOLERANCE_NM
    )

    czi_files = find_czi_files(roots)
    if not czi_files:
        print("No .czi files to convert (all already have a .h5).")
        return

    print(f"Found {len(czi_files)} CZI files.\n")
    errors = []
    for path in czi_files:
        print(rel(path))
        try:
            process_czi(path, channel_map, tolerance_nm)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            errors.append((path, e))

    print(f"\nDone. {len(czi_files) - len(errors)}/{len(czi_files)} files converted.")
    if errors:
        print("Errors:")
        for p, e in errors:
            print(f"  {rel(p)}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, action="append", dest="roots",
        help="Directory to search (repeatable). Defaults to the configured roots.",
    )
    args = parser.parse_args()
    main(roots=args.roots)
