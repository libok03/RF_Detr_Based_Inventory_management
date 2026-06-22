import argparse
import re
import time
from pathlib import Path
from typing import List

import pandas as pd


CLASS_RE = re.compile(r"^class_(\d+)$")


def class_columns(df: pd.DataFrame) -> List[str]:
    cols = [col for col in df.columns if CLASS_RE.match(col)]
    return sorted(cols, key=lambda col: int(col.split("_")[1]))


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def infer_params(run_dir: Path) -> dict:
    text = str(run_dir).lower()
    params = {"run_name": run_dir.parent.name if run_dir.name == "inventory" else run_dir.name}
    conf_match = re.search(r"conf(?:_|)(\d+)", text)
    nms_match = re.search(r"nms(?:_|)(\d+)", text)
    if conf_match:
        params["rf_conf"] = int(conf_match.group(1)) / 100
    if nms_match:
        params["nms_iou"] = int(nms_match.group(1)) / 100
    return params


def smooth_group(group: pd.DataFrame, class_cols: List[str], window: int, min_appear: int) -> pd.DataFrame:
    group = group.sort_values("image").reset_index(drop=True)
    out = group.copy()
    values = group[class_cols].astype(int)
    appear = (values > 0).rolling(window=window, min_periods=1).sum()
    window_max = values.rolling(window=window, min_periods=1).max()
    out[class_cols] = window_max.where(appear >= min_appear, 0).astype(int)
    out["total_count"] = out[class_cols].sum(axis=1).astype(int)
    return out


def fuse_camera_counts(df: pd.DataFrame, class_cols: List[str]) -> pd.DataFrame:
    grouped = df.groupby(["event_id", "model"], sort=True)
    fused = grouped[class_cols].max().astype(int).reset_index()
    camera_counts = grouped["camera"].nunique().reset_index(name="num_cameras")
    fused = fused.merge(camera_counts, on=["event_id", "model"], how="left")
    fused["fused_total_count"] = fused[class_cols].sum(axis=1).astype(int)
    return fused[["event_id", "model", "num_cameras", *class_cols, "fused_total_count"]]


def summarize_fused(fused: pd.DataFrame, prefix: str) -> dict:
    totals = fused["fused_total_count"].astype(int)
    return {
        f"{prefix}_mean_total": float(totals.mean()),
        f"{prefix}_median_total": float(totals.median()),
        f"{prefix}_max_total": int(totals.max()),
        f"{prefix}_nonzero_frames": int((totals > 0).sum()),
        f"{prefix}_zero_frames": int((totals == 0).sum()),
        f"{prefix}_mean_abs_delta": float(totals.diff().abs().fillna(0).mean()),
    }


def load_run(run_dir: Path, cap_one: bool) -> tuple[pd.DataFrame, List[str]]:
    input_path = run_dir / "per_image_counts.csv"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    df = pd.read_csv(input_path)
    class_cols = class_columns(df)
    if not class_cols:
        raise ValueError(f"No class columns found in {input_path}")
    df[class_cols] = df[class_cols].astype(int)
    if cap_one:
        df[class_cols] = df[class_cols].clip(upper=1)
        df["total_count"] = df[class_cols].sum(axis=1).astype(int)
    return df, class_cols


def analyze_run(run_dir: Path, windows: List[int], mins: List[int], fps: float, cap_one: bool, save_temporal: bool) -> List[dict]:
    start = time.perf_counter()
    df, class_cols = load_run(run_dir, cap_one)
    raw_fused = fuse_camera_counts(df, class_cols)
    params = infer_params(run_dir)
    unique_events = raw_fused["event_id"].nunique()
    video_seconds = unique_events / fps if fps > 0 else 0.0
    latency_seconds = float(df["latency_ms"].astype(float).sum() / 1000.0) if "latency_ms" in df else 0.0
    mean_latency_ms = float(df["latency_ms"].astype(float).mean()) if "latency_ms" in df else 0.0

    rows = []
    for window in windows:
        for min_appear in mins:
            if min_appear > window:
                continue
            smoothed_parts = [
                smooth_group(group, class_cols, window, min_appear)
                for _, group in df.groupby(["camera", "model"], sort=False)
            ]
            smoothed = pd.concat(smoothed_parts, ignore_index=True)
            temporal_fused = fuse_camera_counts(smoothed, class_cols)
            if save_temporal:
                suffix = f"binary_w{window}_m{min_appear}"
                smoothed.to_csv(run_dir / f"per_image_counts_{suffix}.csv", index=False, encoding="utf-8-sig")
                temporal_fused.to_csv(run_dir / f"camera_fused_counts_{suffix}.csv", index=False, encoding="utf-8-sig")

            row = {
                **params,
                "run_dir": str(run_dir),
                "cap_one": cap_one,
                "window": window,
                "min_appear": min_appear,
                "frames": int(len(df)),
                "events": int(unique_events),
                "fps": fps,
                "video_seconds": float(video_seconds),
                "inference_seconds": latency_seconds,
                "mean_latency_ms": mean_latency_ms,
                "inference_rtf": float(latency_seconds / video_seconds) if video_seconds > 0 else 0.0,
            }
            row.update(summarize_fused(raw_fused, "raw"))
            row.update(summarize_fused(temporal_fused, "temporal"))
            row["postprocess_seconds"] = float(time.perf_counter() - start)
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RF-DETR conf/NMS runs with binary class-count assumption and temporal windows."
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Inventory output directories containing per_image_counts.csv")
    parser.add_argument("--output", default="binary_temporal_sweep_summary.csv")
    parser.add_argument("--windows", default="5", help="Comma-separated window values, e.g. 5,8,10")
    parser.add_argument("--mins", default="5", help="Comma-separated min_appear values, e.g. 3,4,5")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--no-cap-one", dest="cap_one", action="store_false")
    parser.add_argument("--save-temporal", action="store_true")
    parser.set_defaults(cap_one=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = parse_int_list(args.windows)
    mins = parse_int_list(args.mins)
    all_rows = []
    for run in args.runs:
        all_rows.extend(analyze_run(Path(run), windows, mins, args.fps, args.cap_one, args.save_temporal))
    out = pd.DataFrame(all_rows)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(args.output)


if __name__ == "__main__":
    main()
