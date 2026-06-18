import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


CLASS_RE = re.compile(r"^class_(\d+)$")


def class_columns(df: pd.DataFrame) -> List[str]:
    cols = [col for col in df.columns if CLASS_RE.match(col)]
    return sorted(cols, key=lambda col: int(col.split("_")[1]))


def smooth_group(group: pd.DataFrame, class_cols: List[str], window: int, min_appear: int) -> pd.DataFrame:
    group = group.sort_values("image").reset_index(drop=True)
    out = group.copy()
    values = group[class_cols].astype(int)

    for idx in range(len(group)):
        start = max(0, idx - window + 1)
        frame_window = values.iloc[start : idx + 1]
        for col in class_cols:
            appear = int((frame_window[col] > 0).sum())
            out.loc[idx, col] = int(frame_window[col].max()) if appear >= min_appear else 0

    out["total_count"] = out[class_cols].astype(int).sum(axis=1)
    return out


def fuse_camera_counts(df: pd.DataFrame, class_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (event_id, model), group in df.groupby(["event_id", "model"], sort=True):
        row: Dict[str, object] = {
            "event_id": event_id,
            "model": model,
            "num_cameras": group["camera"].nunique(),
        }
        total = 0
        for col in class_cols:
            value = int(group[col].astype(int).max())
            row[col] = value
            total += value
        row["fused_total_count"] = total
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply temporal appearance filtering to per-frame class counts.")
    parser.add_argument("--input", required=True, help="per_image_counts.csv from inventory_pipeline.py")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-appear", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    cols = class_columns(df)
    if not cols:
        raise ValueError("No class_N columns found.")

    smoothed_parts = []
    for _, group in df.groupby(["camera", "model"], sort=False):
        smoothed_parts.append(smooth_group(group, cols, args.window, args.min_appear))
    smoothed = pd.concat(smoothed_parts, ignore_index=True)
    fused = fuse_camera_counts(smoothed, cols)

    smoothed.to_csv(output_dir / "per_image_counts_temporal.csv", index=False, encoding="utf-8-sig")
    fused.to_csv(output_dir / "camera_fused_counts_temporal.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    print(output_dir / "per_image_counts_temporal.csv")
    print(output_dir / "camera_fused_counts_temporal.csv")


if __name__ == "__main__":
    main()
