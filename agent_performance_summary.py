"""Summarize final episodic returns and lengths across all runs.

This script scans the CSVs exported from TensorBoard (`runs_scalars_csv`)
and, for each model type (classical parameter count or quantum variant +
layer count), averages the value of `0-Episodic-Stats/episodic_return`
and `0-Episodic-Stats/episodic_length` at the last recorded step for that run.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# For each csv, there are four columns: step, tag, value, run
# For `tag`, there are the following unique values:
# '0-Episodic-Stats/episodic_length' '0-Episodic-Stats/episodic_return'
# '1-Training-Losses/approx_kl' '1-Training-Losses/clipfrac'
# '1-Training-Losses/entropy' '1-Training-Losses/explained_variance'
# '1-Training-Losses/policy_loss' '1-Training-Losses/value_loss'
# '2-Training-Stats/SPS' '2-Training-Stats/learning_rate'

DEFAULT_CSV_DIR = Path("runs_scalars_csv")
DEFAULT_OUTPUT = Path("final_step_episodic_return_summary.csv")
TARGET_TAG_RETURN = "0-Episodic-Stats/episodic_return"
TARGET_TAG_LENGTH = "0-Episodic-Stats/episodic_length"

# Example filenames:
# - Pong1PCB2L128P__seed_0_1763339523.csv
# - Pong1PQFM_XObs_entangled_trainable_rzz_QLayers_4___seed_3_1761181471.csv
CLASSICAL_RE = re.compile(r"^Pong1PCB2L(?P<params>\d+)P__seed_\d+_\d+$")
QUANTUM_RE = re.compile(
    r"^Pong1PQFM_XObs_(?P<variant>entangled_trainable_rzz|entangled|separable)_"
    r"QLayers_(?P<layers>\d+)___seed_\d+_\d+$"
)
MODEL_SORT_ORDER = {
    "classical": 0,
    "quantum_separable": 1,
    "quantum_entangled": 2,
    "quantum_entangled_trainable_rzz": 3,
}


def parse_model_id(stem: str) -> Tuple[str, int] | None:
    """Return (model_family, depth) where depth is params or layer count."""
    classical = CLASSICAL_RE.match(stem)
    if classical:
        params = int(classical.group("params"))
        return "classical", params
    quantum = QUANTUM_RE.match(stem)
    if quantum:
        variant = quantum.group("variant")
        layers = int(quantum.group("layers"))
        return f"quantum_{variant}", layers
    return None


def final_value_for_tag(csv_file: Path, tag: str) -> float | None:
    """Return the value at the largest step for the requested tag."""
    df = pd.read_csv(csv_file, usecols=["step", "tag", "value"])
    tag_df = df[df["tag"] == tag]
    if tag_df.empty:
        return None
    # Take the entry with the maximum step; ties fall back to last occurrence.
    max_step = tag_df["step"].max()
    last_row = tag_df[tag_df["step"] == max_step].tail(1)
    return float(last_row["value"].iloc[0])


def collect_final_values(csv_dir: Path, tags: List[str]) -> Tuple[Dict[str, Dict[Tuple[str, int], List[float]]], List[str]]:
    """Gather per-run final values grouped by model id for each tag."""
    grouped: Dict[str, Dict[Tuple[str, int], List[float]]] = {tag: defaultdict(list) for tag in tags}
    skipped: List[str] = []
    for csv_file in sorted(csv_dir.glob("*.csv")):
        model_id = parse_model_id(csv_file.stem)
        if model_id is None:
            skipped.append(csv_file.name)
            continue
        per_file_values = {}
        for tag in tags:
            value = final_value_for_tag(csv_file, tag)
            if value is None:
                per_file_values = None
                break
            per_file_values[tag] = value
        if per_file_values is None:
            skipped.append(csv_file.name)
            continue
        for tag, value in per_file_values.items():
            grouped[tag][model_id].append(value)
    return grouped, skipped


def build_summary(
    grouped_returns: Dict[Tuple[str, int], List[float]],
    grouped_lengths: Dict[Tuple[str, int], List[float]],
) -> pd.DataFrame:
    """Create a summary dataframe with mean/std/min/max across seeds for returns and lengths."""
    records = []
    all_model_ids = set(grouped_returns.keys()) | set(grouped_lengths.keys())
    for (model_family, depth) in sorted(
        all_model_ids, key=lambda item: (MODEL_SORT_ORDER.get(item[0], 99), item[1])
    ):
        ret_vals = grouped_returns.get((model_family, depth), [])
        len_vals = grouped_lengths.get((model_family, depth), [])
        records.append(
            {
                "model_family": model_family,
                "layers_or_params": depth,
                "runs_return": len(ret_vals),
                "mean_final_return": float(np.mean(ret_vals)) if ret_vals else np.nan,
                "std_final_return": float(np.std(ret_vals)) if ret_vals else np.nan,
                "max_final_return": float(np.max(ret_vals)) if ret_vals else np.nan,
                "min_final_return": float(np.min(ret_vals)) if ret_vals else np.nan,
                #"runs_length": len(len_vals),
                "mean_final_length": float(np.mean(len_vals)) if len_vals else np.nan,
                "std_final_length": float(np.std(len_vals)) if len_vals else np.nan,
                "max_final_length": float(np.max(len_vals)) if len_vals else np.nan,
                "min_final_length": float(np.min(len_vals)) if len_vals else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No matching runs found.")
        return
    display = df.copy()
    for col in [
        "mean_final_return",
        "std_final_return",
        "max_final_return",
        "min_final_return",
        "mean_final_length",
        "std_final_length",
        "max_final_length",
        "min_final_length",
    ]:
        display[col] = display[col].map(lambda v: f"{v:.2f}")
    print(display.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average the final recorded episodic return and length for each model configuration "
            "across all TensorBoard run CSVs."
        )
    )
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR, help="Directory containing run CSV files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to write the summary CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    grouped, skipped = collect_final_values(args.csv_dir, [TARGET_TAG_RETURN, TARGET_TAG_LENGTH])
    summary_df = build_summary(grouped[TARGET_TAG_RETURN], grouped[TARGET_TAG_LENGTH])
    summary_df.to_csv(args.output, index=False)
    print_summary(summary_df)
    if skipped:
        print(f"\nSkipped {len(skipped)} files (unrecognized name or missing tag).")


if __name__ == "__main__":
    main()
