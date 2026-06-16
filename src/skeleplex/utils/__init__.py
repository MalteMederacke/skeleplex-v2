"""Utilities for operating on the skeleton."""

from skeleplex.utils._chunked import (
    calculate_expanded_slice,
    get_boundary_slices,
    iteratively_process_chunks_3d,
)
from skeleplex.utils._geometry import line_segments_in_aabb, points_in_aabb
from skeleplex.utils._inference_slurm import (
    build_sbatch_command,
    infer_on_chunk,
    initialize_parallel_inference,
)

# Names backed by torch-dependent modules. These are imported lazily (PEP 562)
# so that importing skeleplex.utils — e.g. for the numpy-only geometry helpers
# used by the viewer app — does not require torch to be installed.
_LAZY_IMPORTS = {
    "TTAWrapper": "skeleplex.utils._tta",
    "Identity": "skeleplex.utils._tta_augmentations",
    "FlipZ": "skeleplex.utils._tta_augmentations",
    "FlipY": "skeleplex.utils._tta_augmentations",
    "FlipX": "skeleplex.utils._tta_augmentations",
    "Rot90ZY": "skeleplex.utils._tta_augmentations",
    "Rot180ZY": "skeleplex.utils._tta_augmentations",
    "Rot270ZY": "skeleplex.utils._tta_augmentations",
    "Rot90ZX": "skeleplex.utils._tta_augmentations",
    "Rot180ZX": "skeleplex.utils._tta_augmentations",
    "Rot270ZX": "skeleplex.utils._tta_augmentations",
    "Rot90YX": "skeleplex.utils._tta_augmentations",
    "Rot180YX": "skeleplex.utils._tta_augmentations",
    "Rot270YX": "skeleplex.utils._tta_augmentations",
}


def __getattr__(name: str):
    """Lazily import torch-dependent names on first access (PEP 562)."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    """Include the lazily-imported names in dir() and tab-completion."""
    return [*globals().keys(), *_LAZY_IMPORTS]


__all__ = [
    "FlipX",
    "FlipY",
    "FlipZ",
    "Identity",
    "Rot90YX",
    "Rot90ZX",
    "Rot90ZY",
    "Rot180YX",
    "Rot180ZX",
    "Rot180ZY",
    "Rot270YX",
    "Rot270ZX",
    "Rot270ZY",
    "TTAWrapper",
    "build_sbatch_command",
    "calculate_expanded_slice",
    "get_boundary_slices",
    "infer_on_chunk",
    "initialize_parallel_inference",
    "iteratively_process_chunks_3d",
    "line_segments_in_aabb",
    "points_in_aabb",
]
