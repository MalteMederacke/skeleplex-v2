"""Backfill batch completion markers from existing SLURM log files.

Usage:
    python backfill_markers.py --step 1_2 --log-dir /path/to/logs --csv csvs/step_1_2.csv

Reads log files matching ``<STEP>_fusion_p_*.out`` and writes a marker for every
batch whose log ends with "Batch N finished ...".  Logs that were cancelled or
truncated are left without a marker so check_failed_chunks.py will flag them.
"""

import argparse
import re
import sys
from pathlib import Path

CSV_DIR = Path(__file__).parent / "csvs"


def backfill(step: str, log_dir: Path, csv_path: Path) -> None:
    marker_dir = csv_path.parent / "markers"
    marker_dir.mkdir(exist_ok=True)
    stem = csv_path.stem

    pattern = f"{step}_fusion_p_*.out"
    log_files = sorted(log_dir.glob(pattern))
    if not log_files:
        print(f"No log files found matching {log_dir / pattern}")
        sys.exit(1)

    finished_re = re.compile(r"^Batch (\d+) finished")

    written = 0
    skipped = 0
    for log_file in log_files:
        batch_id = None
        try:
            text = log_file.read_text(errors="replace")
        except OSError as exc:
            print(f"  SKIP {log_file.name}: {exc}")
            skipped += 1
            continue

        for line in text.splitlines():
            m = finished_re.match(line.strip())
            if m:
                batch_id = int(m.group(1))
                break

        if batch_id is None:
            skipped += 1
            continue

        marker = marker_dir / f"{stem}_batch_{batch_id}.done"
        if not marker.exists():
            marker.touch()
            written += 1

    total = len(log_files)
    print(f"Log files found : {total}")
    print(f"Markers written : {written}")
    print(f"Incomplete/skip : {skipped}")
    print(f"Marker dir      : {marker_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        required=True,
        choices=["1_1", "1_2", "1_3", "2_2", "2_3", "2_4"],
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Directory containing the SLURM .out log files",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Input CSV (default: csvs/step_<STEP>.csv)",
    )
    args = parser.parse_args()

    csv_path = args.csv or CSV_DIR / f"step_{args.step}.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    backfill(args.step, args.log_dir, csv_path)
