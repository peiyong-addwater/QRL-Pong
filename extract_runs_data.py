from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tbparse import SummaryReader


RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("runs_scalars_csv")


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    """Yield sub-directories inside the TensorBoard runs folder."""
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir():
            yield candidate


def _has_event_files(run_dir: Path) -> bool:
    """Quick check to skip runs that do not contain TensorBoard event files."""
    return any(run_dir.glob("events.out.tfevents.*"))


def collect_scalars(run_dir: Path, pivot: bool) -> pd.DataFrame:
    reader = SummaryReader(run_dir, pivot=pivot)
    df = reader.scalars.copy()
    df["run"] = run_dir.name
    return df


def export_runs(root: Path, output_dir: Path, pivot: bool) -> List[Path]:
    written_files: List[Path] = []
    for run_dir in _iter_run_dirs(root):
        if not _has_event_files(run_dir):
            continue
        df = collect_scalars(run_dir, pivot=pivot)
        target = output_dir / f"{run_dir.name}.csv"
        df.to_csv(target, index=False)
        written_files.append(target)
    if not written_files:
        raise RuntimeError(f"No TensorBoard event files found under {root}")
    return written_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate TensorBoard scalars from all runs."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Root directory containing multiple TensorBoard run folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to store one CSV per run.",
    )
    parser.add_argument(
        "--pivot",
        action="store_true",
        help="Return wide-format scalars (each tag as a column).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = export_runs(args.runs_dir, output_dir=args.output_dir, pivot=args.pivot)
    print(f"Wrote {len(written)} CSV files into {args.output_dir}")


if __name__ == "__main__":
    main()
