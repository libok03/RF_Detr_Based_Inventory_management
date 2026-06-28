import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore", message=r".*Pandas requires version.*bottleneck.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*A new version of Albumentations is available.*")
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from torch.jit import TracerWarning

    warnings.filterwarnings("ignore", category=TracerWarning)
except Exception:
    pass

import cv2
import numpy as np
import pandas as pd

try:
    from detectors import RFDETRDetector
    from inventory_pipeline import (
        containment_duplicate_suppression,
        nms_by_class,
        same_class_duplicate_suppression,
    )
    from label_map import LABEL_MAP, class_name, class_price
except ImportError:
    from implementation.detectors import RFDETRDetector
    from implementation.inventory_pipeline import (
        containment_duplicate_suppression,
        nms_by_class,
        same_class_duplicate_suppression,
    )
    from implementation.label_map import LABEL_MAP, class_name, class_price


def yolo_to_xyxy(line: str, w: int, h: int):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    cx, cy, bw, bh = map(float, parts[1:5])
    return cls, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + bb - inter + 1e-9)


def load_gt(label_path: Path, w: int, h: int) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parsed = yolo_to_xyxy(line, w, h)
        if parsed is not None:
            out.append(parsed)
    return out


def match_counts(pred_rows: List[dict], gt_rows: List[Tuple[int, float, float, float, float]], num_classes: int, iou_thr: float) -> Dict[int, dict]:
    pred_by_class = {cls: [] for cls in range(num_classes)}
    gt_by_class = {cls: [] for cls in range(num_classes)}
    for pred in pred_rows:
        cls = int(pred["class_id"])
        if 0 <= cls < num_classes:
            pred_by_class[cls].append(pred)
    for gt in gt_rows:
        cls = int(gt[0])
        if 0 <= cls < num_classes:
            gt_by_class[cls].append(gt)

    stats = {}
    for cls in range(num_classes):
        preds = pred_by_class[cls]
        gts = gt_by_class[cls]
        candidates = []
        for pi, p in enumerate(preds):
            for gi, g in enumerate(gts):
                ov = iou((p["x1"], p["y1"], p["x2"], p["y2"]), g[1:5])
                if ov >= iou_thr:
                    candidates.append((ov, pi, gi))
        candidates.sort(reverse=True, key=lambda x: x[0])
        used_p, used_g = set(), set()
        for _, pi, gi in candidates:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
        tp = len(used_p)
        fp = len(preds) - tp
        fn = len(gts) - len(used_g)
        stats[cls] = {"tp": tp, "fp": fp, "fn": fn, "gt": len(gts), "pred": len(preds)}
    return stats


def postprocess(dets, args):
    out = [d for d in dets if d.confidence >= args.min_conf]
    if args.single_nms:
        out = nms_by_class(out, args.nms_iou)
    if args.duplicate_center_threshold > 0:
        out = same_class_duplicate_suppression(out, args.duplicate_center_threshold, args.duplicate_conf_ratio)
    if args.containment_threshold > 0:
        out = containment_duplicate_suppression(out, args.containment_threshold, args.containment_conf_ratio)
    return out


def collect_detections(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid = Path(args.valid_root)
    image_dir = valid / "images"
    label_dir = valid / "labels"
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.max_images > 0:
        images = images[: args.max_images]

    detector = RFDETRDetector(args.model_path, variant=args.rf_variant, optimize_batch_size=1)
    det_rows = []
    gt_rows = []

    for idx, img_path in enumerate(images, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dets = postprocess(detector.predict_rgb(rgb, args.min_conf), args)
        gts = load_gt(label_dir / f"{img_path.stem}.txt", w, h)

        for d in dets:
            area = max(0.0, d.x2 - d.x1) * max(0.0, d.y2 - d.y1) / max(float(w * h), 1.0)
            det_rows.append(
                {
                    "image": img_path.name,
                    "class_id": d.class_id,
                    "confidence": d.confidence,
                    "x1": d.x1,
                    "y1": d.y1,
                    "x2": d.x2,
                    "y2": d.y2,
                    "area_norm": area,
                }
            )
        for cls, x1, y1, x2, y2 in gts:
            gt_rows.append({"image": img_path.name, "class_id": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        if idx % 500 == 0:
            print(f"[{idx}/{len(images)}] detections={len(det_rows)} gt={len(gt_rows)}", flush=True)

    return pd.DataFrame(det_rows), pd.DataFrame(gt_rows)


def build_threshold_table(det_df: pd.DataFrame, gt_df: pd.DataFrame, thresholds: List[float], args) -> pd.DataFrame:
    images = sorted(set(gt_df["image"].tolist()) | set(det_df["image"].tolist()))
    gt_by_image = {
        image: [
            (int(r.class_id), float(r.x1), float(r.y1), float(r.x2), float(r.y2))
            for r in group.itertuples(index=False)
        ]
        for image, group in gt_df.groupby("image", sort=False)
    }
    max_price = max(class_price(c) for c in range(args.num_classes))
    rows = []
    for threshold in thresholds:
        det_t = det_df[det_df["confidence"] >= threshold].copy()
        det_by_image = {
            image: group.to_dict("records")
            for image, group in det_t.groupby("image", sort=False)
        }
        totals = {cls: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for cls in range(args.num_classes)}
        for image in images:
            pred_rows = det_by_image.get(image, [])
            gt_rows = gt_by_image.get(image, [])
            stats = match_counts(pred_rows, gt_rows, args.num_classes, args.iou_match)
            for cls, s in stats.items():
                for key in totals[cls]:
                    totals[cls][key] += s[key]
        det_class_stats = det_t.groupby("class_id").agg(
            confidence=("confidence", "mean"),
            area_norm=("area_norm", "mean"),
        )
        for cls, s in totals.items():
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            over = fp / (s["gt"] + 1e-9)
            under = fn / (s["gt"] + 1e-9)
            price = class_price(cls)
            price_weight = price / max_price
            score = f1 - args.fp_penalty * price_weight * over - args.fn_penalty * price_weight * under
            if cls in det_class_stats.index:
                mean_conf = float(det_class_stats.loc[cls, "confidence"])
                mean_area = float(det_class_stats.loc[cls, "area_norm"])
            else:
                mean_conf = 0.0
                mean_area = 0.0
            rows.append(
                {
                    "class_id": cls,
                    "item_name": class_name(cls),
                    "price": price,
                    "threshold": threshold,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "gt_count": s["gt"],
                    "pred_count": s["pred"],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "over_rate": over,
                    "under_rate": under,
                    "mean_conf_kept": mean_conf,
                    "mean_area_kept": mean_area,
                    "score": score,
                }
            )
    return pd.DataFrame(rows)


def choose_thresholds_with_rf(table: pd.DataFrame, args):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn") from exc

    feature_cols = [
        "class_id",
        "price",
        "threshold",
        "gt_count",
        "pred_count",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "over_rate",
        "under_rate",
        "mean_conf_kept",
        "mean_area_kept",
    ]
    train = table.copy()
    model = RandomForestRegressor(
        n_estimators=args.rf_trees,
        random_state=args.seed,
        min_samples_leaf=args.rf_min_samples_leaf,
        n_jobs=-1,
    )
    model.fit(train[feature_cols], train["score"])
    train["rf_score"] = model.predict(train[feature_cols])

    chosen = []
    for cls, group in train.groupby("class_id"):
        row = group.sort_values(["rf_score", "score", "f1"], ascending=False).iloc[0]
        chosen.append(row)
    chosen_df = pd.DataFrame(chosen).sort_values("class_id")
    thresholds = {str(int(r.class_id)): round(float(r.threshold), 4) for r in chosen_df.itertuples(index=False)}
    importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    return thresholds, chosen_df, importance, train


def parse_thresholds(spec: str) -> List[float]:
    if "," in spec:
        return [float(x) for x in spec.split(",") if x.strip()]
    start, stop, step = [float(x) for x in spec.split(":")]
    values = []
    cur = start
    while cur <= stop + 1e-9:
        values.append(round(cur, 4))
        cur += step
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Random-Forest-guided per-class RF-DETR confidence thresholds.")
    parser.add_argument("--valid-root", required=True, help="YOLO valid directory containing images/ and labels/.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rf-variant", default="large")
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--thresholds", default="0.30:0.90:0.05", help="start:stop:step or comma list.")
    parser.add_argument("--iou-match", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.40)
    parser.add_argument("--single-nms", action="store_true", default=True)
    parser.add_argument("--duplicate-center-threshold", type=float, default=0.85)
    parser.add_argument("--duplicate-conf-ratio", type=float, default=0.70)
    parser.add_argument("--containment-threshold", type=float, default=0.70)
    parser.add_argument("--containment-conf-ratio", type=float, default=0.95)
    parser.add_argument("--fp-penalty", type=float, default=0.20)
    parser.add_argument("--fn-penalty", type=float, default=0.10)
    parser.add_argument("--rf-trees", type=int, default=400)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--detections-cache", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.detections_cache and Path(args.detections_cache).exists():
        det_df = pd.read_csv(Path(args.detections_cache) / "valid_low_conf_detections.csv")
        gt_df = pd.read_csv(Path(args.detections_cache) / "valid_ground_truth.csv")
    else:
        det_df, gt_df = collect_detections(args)
        det_df.to_csv(output_dir / "valid_low_conf_detections.csv", index=False, encoding="utf-8-sig")
        gt_df.to_csv(output_dir / "valid_ground_truth.csv", index=False, encoding="utf-8-sig")

    table = build_threshold_table(det_df, gt_df, parse_thresholds(args.thresholds), args)
    thresholds, chosen, importance, scored = choose_thresholds_with_rf(table, args)

    table.to_csv(output_dir / "threshold_candidates.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(output_dir / "threshold_candidates_with_rf_score.csv", index=False, encoding="utf-8-sig")
    chosen.to_csv(output_dir / "rf_class_thresholds.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(output_dir / "rf_feature_importance.csv", index=False, encoding="utf-8-sig")
    (output_dir / "rf_class_thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved:", output_dir / "rf_class_thresholds.json")
    print(chosen[["class_id", "item_name", "price", "threshold", "precision", "recall", "f1", "fp", "fn", "score", "rf_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
