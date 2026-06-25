import argparse
import csv
import re
import time
from pathlib import Path
from typing import List

import pandas as pd


CLASS_RE = re.compile(r"^class_(\d+)$")


def class_columns(df: pd.DataFrame) -> List[str]:
    cols = [col for col in df.columns if CLASS_RE.match(col)]
    return sorted(cols, key=lambda col: int(col.split("_")[1]))


def smooth_group(group: pd.DataFrame, class_cols: List[str], window: int, min_appear: int) -> pd.DataFrame:
    group = group.sort_values("image").reset_index(drop=True)
    out = group.copy()
    values = group[class_cols].astype(int)

    appear = (values > 0).rolling(window=window, min_periods=1).sum()
    window_max = values.rolling(window=window, min_periods=1).max()
    out[class_cols] = window_max.where(appear >= min_appear, 0).astype(int)

    out["total_count"] = out[class_cols].astype(int).sum(axis=1)
    return out


def parse_camera_exclusions(value: str) -> set:
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def fuse_camera_counts(df: pd.DataFrame, class_cols: List[str], exclude_cameras: set = None) -> pd.DataFrame:
    exclude_cameras = exclude_cameras or set()
    if exclude_cameras:
        df = df[~df["camera"].astype(str).str.lower().isin(exclude_cameras)].copy()
    grouped = df.groupby(["event_id", "model"], sort=True)
    fused = grouped[class_cols].max().astype(int).reset_index()
    camera_counts = grouped["camera"].nunique().reset_index(name="num_cameras")
    fused = fused.merge(camera_counts, on=["event_id", "model"], how="left")
    ordered_cols = ["event_id", "model", "num_cameras", *class_cols]
    fused = fused[ordered_cols]
    fused["fused_total_count"] = fused[class_cols].sum(axis=1).astype(int)
    return fused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply temporal appearance filtering to per-frame class counts.")
    parser.add_argument("--input", required=True, help="per_image_counts.csv from inventory_pipeline.py")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-appear", type=int, default=5)
    parser.add_argument(
        "--exclude-count-cameras",
        default="",
        help="Comma-separated cameras to exclude from camera-fused temporal counting, e.g. cam2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
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
    fused = fuse_camera_counts(smoothed, cols, parse_camera_exclusions(args.exclude_count_cameras))

    smoothed.to_csv(output_dir / "per_image_counts_temporal.csv", index=False, encoding="utf-8-sig")
    fused.to_csv(output_dir / "camera_fused_counts_temporal.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    print(output_dir / "per_image_counts_temporal.csv")
    print(output_dir / "camera_fused_counts_temporal.csv")
    print(f"temporal_filter_seconds={time.perf_counter() - start:.6f}")


if __name__ == "__main__":
    main()
