"""Resample per-channel HDF5 files to an isotropic voxel size and normalise.

Input:  <root>/**/<stem>.h5            datasets per channel, attr voxel_size_um [z,y,x]
Output: <root>/**/<ISO_SUBDIR>/<stem>.h5   same datasets, float32 0-1,
        attr voxel_size_um [t, t, t] for t = TARGET_VOXEL_SIZE_UM

The target voxel size must match the data the segmentation checkpoints were
trained on, and normalisation goes through ``normalization.normalize_image`` —
the same function ``prepare_patches.py`` applies — so inference sees the
intensity distribution the network was trained on.

Resampling runs chunked along z so a large stack does not have to fit in GPU
memory at once. Files that already have an output are skipped unless
``--overwrite`` (or PREPROCESS_OVERWRITE=1) is given; re-run with it after
changing the normalisation or the target voxel size.
"""

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _constants import (  # noqa: E402
    ISO_SUBDIR,
    PREPROCESS_CHANNELS,
    TARGET_VOXEL_SIZE_UM,
)
from _layout import rel, source_roots  # noqa: E402
from exclusions import is_excluded  # noqa: E402
from normalization import normalize_image  # noqa: E402


def _zoom_backend(backend):
    """Return (zoom_fn, sync_fn) for ``backend``.

    Both backends run the same chunked loop, so switching does not change the
    output — only where the interpolation happens and how much host memory the
    run needs.
    """
    if backend == "gpu":
        import cupy as cp
        import cupyx.scipy.ndimage

        def zoom(slab, scale_factors, order):
            slab_gpu = cp.asarray(slab)
            zoomed_gpu = cupyx.scipy.ndimage.zoom(slab_gpu, scale_factors, order=order)
            out = cp.asnumpy(zoomed_gpu)
            del slab_gpu, zoomed_gpu
            return out

        def sync():
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        return zoom, sync

    from scipy.ndimage import zoom as scipy_zoom

    def zoom(slab, scale_factors, order):
        return scipy_zoom(slab, scale_factors, order=order)

    return zoom, lambda: None


def resize_image(image, current_voxel_size, target_voxel_size, backend="gpu", order=1):
    """Resample a (Z, Y, X) volume to ``target_voxel_size``, chunked along z.

    Chunking bounds peak memory: only a z-slab is interpolated at a time. Each
    slab is padded in input space so the interpolation kernel sees context
    across chunk borders, then trimmed back in output space so consecutive
    chunks tile the output z axis contiguously without gaps or seams.
    """
    zoom, sync = _zoom_backend(backend)

    scale_factors = np.array(current_voxel_size) / np.array(target_voxel_size)
    z_scale = scale_factors[0]

    n_z = image.shape[0]
    # Use round() to match the output shape convention of scipy/cupyx zoom,
    # which sizes each axis as int(round(in * scale)). Y/X are zoomed in full
    # every chunk, so their size is constant and matches each zoomed slab.
    out_z = int(round(n_z * z_scale))
    out_y = int(round(image.shape[1] * scale_factors[1]))
    out_x = int(round(image.shape[2] * scale_factors[2]))
    output = np.zeros((out_z, out_y, out_x), dtype=np.float32)

    z_chunk = 32
    # Input-space overlap so the zoom interpolation kernel at chunk edges sees
    # enough context. Boundaries are computed in output space with round() so
    # consecutive chunks tile the output z axis contiguously.
    z_overlap = int(np.ceil(2 / z_scale)) + 2

    for z_start in range(0, n_z, z_chunk):
        z_end = min(z_start + z_chunk, n_z)

        # Padded input range (clamped to volume bounds)
        pad_start = max(0, z_start - z_overlap)
        pad_end = min(n_z, z_end + z_overlap)

        # Where the padded slab's zoom begins in global output space
        out_pad_start = int(round(pad_start * z_scale))

        # Valid output range (no overlap) — the last chunk reaches exactly out_z
        out_valid_start = int(round(z_start * z_scale))
        out_valid_end = out_z if z_end == n_z else int(round(z_end * z_scale))

        slab = image[pad_start:pad_end].astype(np.float32)
        zoomed = zoom(slab, scale_factors, order)

        # Trim to the valid region in output space, clamping the copy length to
        # what the zoomed slab actually provides and what the output can hold,
        # so a 1-row rounding difference at a boundary can never overflow.
        trim_start = max(0, out_valid_start - out_pad_start)
        n_rows = min(
            out_valid_end - out_valid_start,
            zoomed.shape[0] - trim_start,
            out_z - out_valid_start,
        )
        output[out_valid_start:out_valid_start + n_rows] = (
            zoomed[trim_start:trim_start + n_rows]
        )

        del zoomed, slab
        sync()

    return output


def out_path_for(in_path, iso_subdir):
    """Where the resampled copy of ``in_path`` goes."""
    in_path = Path(in_path)
    return in_path.parent / iso_subdir / in_path.name


def process_h5(in_path, out_path, channels, target_um, backend):
    """Resample and normalise every known channel of one HDF5 file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, sync = _zoom_backend(backend)

    with h5py.File(in_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
        for ch in channels:
            if ch not in f_in:
                continue

            ds_in = f_in[ch]
            voxel_size_um = ds_in.attrs.get("voxel_size_um")
            if voxel_size_um is None:
                print(f"  [skip] {ch}: no voxel_size_um attribute")
                continue

            image = ds_in[:]  # (Z, Y, X)
            resampled = resize_image(
                image,
                list(voxel_size_um),
                [target_um] * 3,
                backend=backend,
                order=1,
            )

            del image
            gc.collect()
            sync()

            normalized = normalize_image(resampled)
            del resampled

            ds_out = f_out.create_dataset(
                ch, data=normalized, compression="gzip", compression_opts=4
            )
            ds_out.attrs["voxel_size_um"] = np.array(
                [target_um] * 3, dtype=np.float64
            )
            del normalized

        written = [ch for ch in channels if ch in f_out]
        print(f"  saved {written} -> {out_path.name}")


def find_h5_files(roots, iso_subdir, overwrite):
    """Unprocessed HDF5 files under ``roots``, excluding curated-out paths.

    Files already inside ``iso_subdir`` are skipped — resampling an already
    resampled file would compound the interpolation.
    """
    todo = []
    for root in roots:
        for path in sorted(Path(root).glob("**/*.h5")):
            if is_excluded(path) or iso_subdir in path.parts:
                continue
            if overwrite or not out_path_for(path, iso_subdir).exists():
                todo.append(path)
    return sorted(todo)


def main(
    roots=None,
    target_um: float = TARGET_VOXEL_SIZE_UM,
    channels=PREPROCESS_CHANNELS,
    iso_subdir: str = ISO_SUBDIR,
    backend: str = "gpu",
    overwrite: bool = False,
) -> None:
    """Resample every unprocessed HDF5 under ``roots`` to isotropic voxels.

    Parameters
    ----------
    roots : sequence of Path, optional
        Directories searched recursively. Defaults to the layout's source roots.
    target_um : float
        Target isotropic voxel size in micrometres.
    channels : sequence of str
        Channel datasets carried through resampling and normalisation.
    iso_subdir : str
        Sub-directory each output is written into, next to its input.
    backend : str
        ``"gpu"`` (cupy) or ``"cpu"`` (scipy) interpolation.
    overwrite : bool
        Re-process files that already have an output.
    """
    roots = list(roots) if roots else source_roots()

    h5_files = find_h5_files(roots, iso_subdir, overwrite)
    if not h5_files:
        print(
            f"No .h5 files to process (overwrite={overwrite}; pass --overwrite "
            f"to redo files that already have an {iso_subdir} version)."
        )
        return

    print(f"Found {len(h5_files)} h5 files. backend={backend}\n")
    errors = []
    for path in h5_files:
        print(rel(path))
        try:
            process_h5(
                path, out_path_for(path, iso_subdir), channels, target_um, backend
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            errors.append((path, e))

    print(f"\nDone. {len(h5_files) - len(errors)}/{len(h5_files)} files processed.")
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
    parser.add_argument("--target-um", type=float, default=TARGET_VOXEL_SIZE_UM)
    parser.add_argument("--iso-subdir", default=ISO_SUBDIR)
    parser.add_argument(
        "--backend", choices=["cpu", "gpu"], default="gpu",
        help="Interpolation backend: cupy on the GPU, or scipy on the CPU.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        default=os.environ.get("PREPROCESS_OVERWRITE", "0") == "1",
        help="Re-process files that already have an output (PREPROCESS_OVERWRITE=1).",
    )
    args = parser.parse_args()

    main(
        roots=args.roots,
        target_um=args.target_um,
        iso_subdir=args.iso_subdir,
        backend=args.backend,
        overwrite=args.overwrite,
    )
