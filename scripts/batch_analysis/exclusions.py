"""Single source of truth for which files the pipeline must skip.

Folders holding files that should NOT be processed are named inconsistently on
disk ("not_processed", "no processed", "bad_quality", ...), so every path part
is normalised before comparing. Importing this instead of re-implementing the
check keeps every stage -- czi_to_h5, preprocess_to_isotropic, segment_batch,
segmentation_to_graph_batch, review_segmentations, analyze_graphs_batch -- in
agreement about what was curated out.

A literal `"not_processed" in Path(p).parts` does NOT work: one such folder on
disk is actually "no processed" (a space, and no "t"), so such a check silently
matches nothing and the rejected samples flow straight through to the results.
It has gone wrong twice this way -- once letting 16 raw files / 244 GB through
czi_to_h5 and preprocess_to_isotropic, and once letting the "no processed"
segmentation copies double segmentation_to_graph_batch's work list from 17 to
34. Import from here rather than re-implementing the check.

``review_segmentations.py`` moves rejected samples into BAD_QUALITY_DIR, which
is one of the names below, so a rejection made there is honoured by every later
stage without any further bookkeeping.
"""

from pathlib import Path

# Directory name review_segmentations.py moves rejected samples into. It must
# stay one of the names _EXCLUDED_DIRS normalises to.
BAD_QUALITY_DIR = "bad_quality"

_EXCLUDED_DIRS = {"not_processed", "no_processed", "notprocessed", "noprocessed",
                  "bad_quality", "badquality"}


def is_excluded(path):
    """True if any directory in the path marks the file as not-to-be-processed."""
    for part in Path(path).parts:
        norm = "".join(c if c.isalnum() else "_" for c in part.strip().lower())
        while "__" in norm:
            norm = norm.replace("__", "_")
        if norm.strip("_") in _EXCLUDED_DIRS:
            return True
    return False
