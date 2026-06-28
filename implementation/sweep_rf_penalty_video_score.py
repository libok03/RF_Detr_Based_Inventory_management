import argparse
import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore", message=r".*Pandas requires version.*bottleneck.*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd

try:
    from temporal_filter import class_columns, fuse_camera_counts, parse_camera_exclusions, smooth_group
    from train_rf_class_thresholds import build_threshold_table, choose_thresholds_with_rf, parse_thresholds
except ImportError:
    from implementation.temporal_filter import class_columns, fuse_camera_counts, parse_camera_exclusions, smooth_group
    from implementation.train_rf_class_thresholds import build_threshold_table, choose_thresholds_with_rf, parse_thresholds


def parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def load_valid_cache(valid_cache: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    det_path = valid_cache / "valid_low_conf_detections.csv"
    gt_path = valid_cache / "valid_ground_truth.csv"
    if not det_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"Missing valid cache files under {valid_cache}")
    return pd.read_csv(det_path), pd.read_csv(gt_path)


def build_or_load_base_table(
    valid_cache: Path,
    output_dir: Path,
    threshold_spec: str,
    num_classes: int,
    iou_match: float,
) -> pd.DataFrame:
    cache_path = output_dir / "threshold_candidates_base.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    det_df, gt_df = load_valid_cache(valid_cache)
    args = SimpleNamespace(
        num_classes=num_classes,
        iou_match=iou_match,
        fp_penalty=0.0,
        fn_penalty=0.0,
    )
    table = build_threshold_table(det_df, gt_df, parse_thresholds(threshold_spec), args)
    table.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return table


def score_threshold_table(table: pd.DataFrame, fp_penalty: float, fn_penalty: float) -> pd.DataFrame:
    out = table.copy()
    max_price = max(float(out["price"].max()), 1e-9)
    price_weight = out["price"].astype(float) / max_price
    out["score"] = (
        out["f1"].astype(float)
        - fp_penalty * price_weight * out["over_rate"].astype(float)
        - fn_penalty * price_weight * out["under_rate"].astype(float)
    )
    return out


def choose_thresholds(table: pd.DataFrame, args) -> tuple[Dict[int, float], pd.DataFrame]:
    thresholds, chosen, _, _ = choose_thresholds_with_rf(table, args)
    return {int(k): float(v) for k, v in thresholds.items()}, chosen


def apply_thresholds_to_video_detections(det_df: pd.DataFrame, thresholds: Dict[int, float]) -> pd.DataFrame:
    cls = det_df["class_id"].astype(int)
    conf = det_df["confidence"].astype(float)
    keep = conf >= cls.map(lambda class_id: thresholds.get(int(class_id), 1.0)).astype(float)
    return det_df[keep].copy()


def build_per_image_counts(base_counts: pd.DataFrame, det_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    class_cols = [f"class_{class_id}" for class_id in range(num_classes)]
    meta_cols = ["image", "event_id", "camera", "model"]
    rows = base_counts[meta_cols].copy()
    rows["total_count"] = 0
    for col in class_cols:
        rows[col] = 0

    if not det_df.empty:
        counts = (
            det_df.assign(class_col=det_df["class_id"].astype(int).map(lambda x: f"class_{x}"))
            .groupby(["image", "class_col"], sort=False)
            .size()
            .unstack(fill_value=0)
        )
        rows = rows.set_index("image")
        for col in counts.columns:
            if col in rows.columns:
                rows.loc[counts.index, col] = counts[col].astype(int)
        rows = rows.reset_index()

    rows["total_count"] = rows[class_cols].astype(int).sum(axis=1)
    return rows[["image", "event_id", "camera", "model", "total_count", *class_cols]]


def apply_temporal_and_fuse(
    per_image: pd.DataFrame,
    window: int,
    min_appear: int,
    exclude_count_cameras: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = class_columns(per_image)
    parts = []
    for _, group in per_image.groupby(["camera", "model"], sort=False):
        parts.append(smooth_group(group, cols, window, min_appear))
    temporal = pd.concat(parts, ignore_index=True)
    fused = fuse_camera_counts(temporal, cols, parse_camera_exclusions(exclude_count_cameras))
    return temporal, fused


def score_video(
    fused: pd.DataFrame,
    coverage_weight: float,
    overcount_class_weight: float,
    overcount_frame_weight: float,
    overcount_excess_weight: float,
) -> Dict[str, float]:
    class_cols = class_columns(fused)
    counts = fused[class_cols].astype(int)
    max_counts = counts.max(axis=0)
    present_classes = int((max_counts > 0).sum())
    overcount_classes = int((max_counts > 1).sum())
    overcount_event_class_pairs = int((counts > 1).sum().sum())
    overcount_excess = int((counts - 1).clip(lower=0).sum().sum())
    max_fused_total = int(fused["fused_total_count"].max()) if not fused.empty else 0
    mean_fused_total = float(fused["fused_total_count"].mean()) if not fused.empty else 0.0
    video_score = (
        coverage_weight * present_classes
        - overcount_class_weight * overcount_classes
        - overcount_frame_weight * overcount_event_class_pairs
        - overcount_excess_weight * overcount_excess
    )
    return {
        "video_score": float(video_score),
        "present_classes": present_classes,
        "overcount_classes": overcount_classes,
        "overcount_event_class_pairs": overcount_event_class_pairs,
        "overcount_excess": overcount_excess,
        "max_fused_total": max_fused_total,
        "mean_fused_total": mean_fused_total,
    }


def write_threshold_json(path: Path, thresholds: Dict[int, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({str(k): round(float(v), 4) for k, v in sorted(thresholds.items())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep RF threshold penalties and score them on cached video detections.")
    parser.add_argument("--valid-cache", required=True, help="Directory with valid_low_conf_detections.csv and valid_ground_truth.csv.")
    parser.add_argument("--video-cache-dir", required=True, help="Inventory output directory with low-conf detections.csv and per_image_counts.csv.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fp-penalties", default="0.10,0.20,0.30,0.40,0.50")
    parser.add_argument("--fn-penalties", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--thresholds", default="0.30:0.90:0.05")
    parser.add_argument("--iou-match", type=float, default=0.50)
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument("--rf-trees", type=int, default=400)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-appear", type=int, default=5)
    parser.add_argument("--exclude-count-cameras", default="cam2")
    parser.add_argument("--coverage-weight", type=float, default=1.0)
    parser.add_argument("--overcount-class-weight", type=float, default=1.0)
    parser.add_argument("--overcount-frame-weight", type=float, default=0.02)
    parser.add_argument("--overcount-excess-weight", type=float, default=0.05)
    parser.add_argument("--save-all-thresholds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_cache_dir)

    base_table = build_or_load_base_table(
        Path(args.valid_cache),
        output_dir,
        args.thresholds,
        args.num_classes,
        args.iou_match,
    )
    video_dets = pd.read_csv(video_dir / "detections.csv")
    base_counts = pd.read_csv(video_dir / "per_image_counts.csv")

    rf_args = SimpleNamespace(
        rf_trees=args.rf_trees,
        seed=args.seed,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
    )

    summary_rows = []
    threshold_dir = output_dir / "thresholds"
    for fp_penalty in parse_float_list(args.fp_penalties):
        for fn_penalty in parse_float_list(args.fn_penalties):
            scored_table = score_threshold_table(base_table, fp_penalty, fn_penalty)
            thresholds, chosen = choose_thresholds(scored_table, rf_args)
            filtered = apply_thresholds_to_video_detections(video_dets, thresholds)
            per_image = build_per_image_counts(base_counts, filtered, args.num_classes)
            _, fused = apply_temporal_and_fuse(per_image, args.window, args.min_appear, args.exclude_count_cameras)
            metrics = score_video(
                fused,
                args.coverage_weight,
                args.overcount_class_weight,
                args.overcount_frame_weight,
                args.overcount_excess_weight,
            )
            row = {
                "fp_penalty": fp_penalty,
                "fn_penalty": fn_penalty,
                **metrics,
                "mean_threshold": float(sum(thresholds.values()) / max(len(thresholds), 1)),
                "min_threshold": float(min(thresholds.values())),
                "max_threshold": float(max(thresholds.values())),
                "kept_detections": int(len(filtered)),
            }
            summary_rows.append(row)

            tag = f"fp{fp_penalty:.2f}_fn{fn_penalty:.2f}".replace(".", "")
            chosen.to_csv(output_dir / f"chosen_{tag}.csv", index=False, encoding="utf-8-sig")
            if args.save_all_thresholds:
                write_threshold_json(threshold_dir / f"{tag}.json", thresholds)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["video_score", "present_classes", "overcount_excess"],
        ascending=[False, False, True],
    )
    summary.to_csv(output_dir / "penalty_video_score_summary.csv", index=False, encoding="utf-8-sig")

    best = summary.iloc[0]
    best_table = score_threshold_table(base_table, float(best.fp_penalty), float(best.fn_penalty))
    best_thresholds, best_chosen = choose_thresholds(best_table, rf_args)
    write_threshold_json(output_dir / "best_rf_class_thresholds.json", best_thresholds)
    best_chosen.to_csv(output_dir / "best_rf_class_thresholds.csv", index=False, encoding="utf-8-sig")

    print(summary.head(20).to_string(index=False))
    print("saved:", output_dir / "penalty_video_score_summary.csv")
    print("best:", output_dir / "best_rf_class_thresholds.json")


if __name__ == "__main__":
    main()
