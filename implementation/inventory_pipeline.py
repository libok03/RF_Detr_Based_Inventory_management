import argparse
import csv
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
try:
    from detectors import Detection, RFDETRDetector, YOLODetector, iou, iter_images, weighted_boxes_fusion
    from label_map import class_name
except ImportError:
    from implementation.detectors import Detection, RFDETRDetector, YOLODetector, iou, iter_images, weighted_boxes_fusion
    from implementation.label_map import class_name


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CLASSES = 60
CAM_RE = re.compile(r"(?:^|_)(cam\d+)(?:_|$)", re.IGNORECASE)


def nms_by_class(detections: Sequence[Detection], iou_threshold: float) -> List[Detection]:
    kept: List[Detection] = []
    for class_id in sorted({det.class_id for det in detections}):
        candidates = sorted(
            [det for det in detections if det.class_id == class_id],
            key=lambda det: det.confidence,
            reverse=True,
        )
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            candidates = [det for det in candidates if iou(best.xyxy, det.xyxy) < iou_threshold]
    return sorted(kept, key=lambda det: (det.class_id, -det.confidence))


def same_class_duplicate_suppression(
    detections: Sequence[Detection],
    center_threshold: float,
    confidence_ratio_threshold: float,
) -> List[Detection]:
    if center_threshold <= 0:
        return list(detections)

    kept: List[Detection] = []
    for class_id in sorted({det.class_id for det in detections}):
        candidates = sorted(
            [det for det in detections if det.class_id == class_id],
            key=lambda det: det.confidence,
            reverse=True,
        )
        class_kept: List[Detection] = []
        for det in candidates:
            det_cx = (det.x1 + det.x2) * 0.5
            det_cy = (det.y1 + det.y2) * 0.5
            det_w = max(1.0, det.x2 - det.x1)
            det_h = max(1.0, det.y2 - det.y1)
            duplicate = False
            for prev in class_kept:
                prev_cx = (prev.x1 + prev.x2) * 0.5
                prev_cy = (prev.y1 + prev.y2) * 0.5
                prev_w = max(1.0, prev.x2 - prev.x1)
                prev_h = max(1.0, prev.y2 - prev.y1)
                dx = det_cx - prev_cx
                dy = det_cy - prev_cy
                distance = (dx * dx + dy * dy) ** 0.5
                scale = max(det_w, det_h, prev_w, prev_h)
                confidence_ratio = det.confidence / max(prev.confidence, 1e-9)
                if distance <= center_threshold * scale and confidence_ratio < confidence_ratio_threshold:
                    duplicate = True
                    break
            if not duplicate:
                class_kept.append(det)
        kept.extend(class_kept)
    return sorted(kept, key=lambda det: (det.class_id, -det.confidence))


def soft_nms_by_class(
    detections: Sequence[Detection],
    iou_threshold: float,
    min_score: float,
) -> List[Detection]:
    kept: List[Detection] = []
    for class_id in sorted({det.class_id for det in detections}):
        candidates = [det for det in detections if det.class_id == class_id]
        while candidates:
            candidates.sort(key=lambda det: det.confidence, reverse=True)
            best = candidates.pop(0)
            kept.append(best)
            decayed = []
            for det in candidates:
                overlap = iou(best.xyxy, det.xyxy)
                if overlap >= iou_threshold:
                    det = Detection(
                        class_id=det.class_id,
                        confidence=det.confidence * (1.0 - overlap),
                        x1=det.x1,
                        y1=det.y1,
                        x2=det.x2,
                        y2=det.y2,
                        image_w=det.image_w,
                        image_h=det.image_h,
                        source=det.source,
                    )
                if det.confidence >= min_score:
                    decayed.append(det)
            candidates = decayed
    return sorted(kept, key=lambda det: (det.class_id, -det.confidence))


class ModelRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = args.model

        self.yolo = None
        self.rf = None

        if self.model in {"yolo11n", "ensemble_wbf", "ensemble_nms", "ensemble_soft_nms"}:
            self.yolo = YOLODetector(args.yolo_model, device=args.device)

        if self.model in {
            "rf_detr_small_aug",
            "rf_detr_large",
            "rf_detr_large_aug",
            "ensemble_wbf",
            "ensemble_nms",
            "ensemble_soft_nms",
        }:
            rf_model, rf_variant = self._resolve_rf_model()
            self.rf = RFDETRDetector(rf_model, variant=rf_variant)

    def _resolve_rf_model(self) -> Tuple[str, str]:
        if self.model == "rf_detr_small_aug":
            return self.args.rf_small_aug_model, "small"
        if self.model == "rf_detr_large":
            return self.args.rf_large_model, "large"
        return self.args.rf_large_aug_model, "large"

    def predict(self, image_path: Path) -> List[Detection]:
        if self.model == "yolo11n":
            return self.yolo.predict(image_path, self.args.yolo_conf)

        if self.model.startswith("rf_detr_"):
            dets = self.rf.predict(image_path, self.args.rf_conf)
            if self.args.single_nms:
                dets = nms_by_class(dets, self.args.nms_iou)
            if self.args.duplicate_center_threshold > 0:
                dets = same_class_duplicate_suppression(
                    dets,
                    self.args.duplicate_center_threshold,
                    self.args.duplicate_conf_ratio,
                )
            return dets

        rf_dets = self.rf.predict(image_path, self.args.rf_conf)
        yolo_dets = self.yolo.predict(image_path, self.args.yolo_conf)
        dets = rf_dets + yolo_dets
        weights = {"rfdetr": self.args.rf_weight, "yolo": self.args.yolo_weight}

        if self.model == "ensemble_wbf":
            dets = weighted_boxes_fusion(dets, self.args.nms_iou, weights)
            return same_class_duplicate_suppression(
                dets,
                self.args.duplicate_center_threshold,
                self.args.duplicate_conf_ratio,
            )
        if self.model == "ensemble_nms":
            dets = nms_by_class(dets, self.args.nms_iou)
            return same_class_duplicate_suppression(
                dets,
                self.args.duplicate_center_threshold,
                self.args.duplicate_conf_ratio,
            )
        if self.model == "ensemble_soft_nms":
            dets = soft_nms_by_class(dets, self.args.nms_iou, self.args.soft_nms_min_score)
            return same_class_duplicate_suppression(
                dets,
                self.args.duplicate_center_threshold,
                self.args.duplicate_conf_ratio,
            )

        raise ValueError(f"Unknown model: {self.model}")


def parse_camera(filename: str) -> str:
    match = CAM_RE.search(filename)
    return match.group(1).lower() if match else "unknown"


def parse_event_id(path: Path) -> str:
    name = path.stem
    cam = parse_camera(path.name)
    if cam == "unknown":
        return name
    return re.sub(rf"(^|_){cam}(_|$)", lambda m: m.group(1) if m.group(2) == "" else m.group(1), name, count=1, flags=re.IGNORECASE).strip("_")


def count_classes(detections: Iterable[Detection], num_classes: int) -> Dict[int, int]:
    counts = {class_id: 0 for class_id in range(num_classes)}
    for det in detections:
        if 0 <= det.class_id < num_classes:
            counts[det.class_id] += 1
        else:
            counts[det.class_id] = counts.get(det.class_id, 0) + 1
    return counts


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_long_count_csv(path: Path, rows: List[Dict[str, object]], num_classes: int) -> None:
    long_rows = []
    for row in rows:
        for class_id in range(num_classes):
            count = int(row.get(f"class_{class_id}", 0))
            if count <= 0:
                continue
            long_rows.append(
                {
                    "event_id": row.get("event_id", ""),
                    "image": row.get("image", ""),
                    "camera": row.get("camera", ""),
                    "model": row.get("model", ""),
                    "class_id": class_id,
                    "item_name": class_name(class_id),
                    "count": count,
                }
            )
    write_csv(path, long_rows, ["event_id", "image", "camera", "model", "class_id", "item_name", "count"])


def write_long_fused_csv(path: Path, rows: List[Dict[str, object]], num_classes: int) -> None:
    long_rows = []
    for row in rows:
        for class_id in range(num_classes):
            count = int(row.get(f"class_{class_id}", 0))
            if count <= 0:
                continue
            long_rows.append(
                {
                    "event_id": row.get("event_id", ""),
                    "model": row.get("model", ""),
                    "num_cameras": row.get("num_cameras", ""),
                    "class_id": class_id,
                    "item_name": class_name(class_id),
                    "fused_count": count,
                }
            )
    write_csv(path, long_rows, ["event_id", "model", "num_cameras", "class_id", "item_name", "fused_count"])


def run(args: argparse.Namespace) -> None:
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    images = list(iter_images(source, args.recursive))
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise FileNotFoundError(f"No images found: {source}")

    runner = ModelRunner(args)

    per_image_rows: List[Dict[str, object]] = []
    detection_rows: List[Dict[str, object]] = []
    details = []

    for idx, image_path in enumerate(images, start=1):
        start = time.perf_counter()
        detections = runner.predict(image_path)
        latency_ms = (time.perf_counter() - start) * 1000.0
        counts = count_classes(detections, args.num_classes)
        event_id = parse_event_id(image_path)
        camera = parse_camera(image_path.name)
        total_count = sum(counts.values())

        logger.info("[%d/%d] %s count=%d latency=%.2fms", idx, len(images), image_path.name, total_count, latency_ms)

        row = {
            "image": image_path.name,
            "event_id": event_id,
            "camera": camera,
            "model": args.model,
            "total_count": total_count,
            "latency_ms": f"{latency_ms:.3f}",
        }
        row.update({f"class_{class_id}": counts.get(class_id, 0) for class_id in range(args.num_classes)})
        per_image_rows.append(row)

        for det in detections:
            detection_rows.append(
                {
                    "image": image_path.name,
                    "event_id": event_id,
                    "camera": camera,
                    "model": args.model,
                    "class_id": det.class_id,
                    "confidence": f"{det.confidence:.6f}",
                    "x1": f"{det.x1:.2f}",
                    "y1": f"{det.y1:.2f}",
                    "x2": f"{det.x2:.2f}",
                    "y2": f"{det.y2:.2f}",
                    "source": det.source,
                }
            )

        details.append(
            {
                "image": str(image_path),
                "event_id": event_id,
                "camera": camera,
                "model": args.model,
                "latency_ms": latency_ms,
                "class_counts": {str(k): v for k, v in counts.items() if v},
                "detections": [asdict(det) for det in detections],
            }
        )

    class_cols = [f"class_{class_id}" for class_id in range(args.num_classes)]
    write_csv(
        output_dir / "per_image_counts.csv",
        per_image_rows,
        ["image", "event_id", "camera", "model", "total_count", "latency_ms"] + class_cols,
    )
    write_csv(
        output_dir / "detections.csv",
        detection_rows,
        ["image", "event_id", "camera", "model", "class_id", "confidence", "x1", "y1", "x2", "y2", "source"],
    )

    fused_rows = fuse_camera_counts(per_image_rows, args.num_classes)
    write_csv(
        output_dir / "camera_fused_counts.csv",
        fused_rows,
        ["event_id", "model", "num_cameras", "fused_total_count"] + class_cols,
    )
    write_long_count_csv(output_dir / "per_image_counts_long.csv", per_image_rows, args.num_classes)
    write_long_fused_csv(output_dir / "camera_fused_counts_long.csv", fused_rows, args.num_classes)

    with (output_dir / "detections.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    logger.info("Saved implementation outputs to %s", output_dir.resolve())


def fuse_camera_counts(rows: List[Dict[str, object]], num_classes: int) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["event_id"]), str(row["model"])), []).append(row)

    fused = []
    for (event_id, model), group in sorted(grouped.items()):
        out = {
            "event_id": event_id,
            "model": model,
            "num_cameras": len({row["camera"] for row in group}),
        }
        fused_total = 0
        for class_id in range(num_classes):
            value = max(int(row[f"class_{class_id}"]) for row in group)
            out[f"class_{class_id}"] = value
            fused_total += value
        out["fused_total_count"] = fused_total
        fused.append(out)
    return fused


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unmanned-store detection, count, and camera fusion pipeline.")
    parser.add_argument("--source", required=True, help="Image/frame file or directory")
    parser.add_argument("--output-dir", default="implementation_outputs/inventory")
    parser.add_argument(
        "--model",
        default="yolo11n",
        choices=[
            "yolo11n",
            "rf_detr_small_aug",
            "rf_detr_large",
            "rf_detr_large_aug",
            "ensemble_wbf",
            "ensemble_nms",
            "ensemble_soft_nms",
        ],
    )
    parser.add_argument("--yolo-model", default=str(ROOT / "server_eval_package" / "weights" / "best.pt"))
    parser.add_argument("--rf-small-aug-model", default=str(ROOT / "server_eval_package" / "weights" / "rf-detr_small_aug.pth"))
    parser.add_argument("--rf-large-model", default=str(ROOT / "server_eval_package" / "weights" / "rf-detr_large.pth"))
    parser.add_argument("--rf-large-aug-model", default=str(ROOT / "server_eval_package" / "weights" / "rf-detr_large_aug.pt"))
    parser.add_argument("--device", default=None, help="YOLO device, e.g. 0 or cpu")
    parser.add_argument("--rf-conf", type=float, default=0.10)
    parser.add_argument("--yolo-conf", type=float, default=0.60)
    parser.add_argument("--nms-iou", type=float, default=0.55)
    parser.add_argument("--single-nms", action="store_true", help="Apply class-wise NMS to single RF-DETR outputs")
    parser.add_argument("--duplicate-center-threshold", type=float, default=0.85)
    parser.add_argument("--duplicate-conf-ratio", type=float, default=0.65)
    parser.add_argument("--soft-nms-min-score", type=float, default=0.001)
    parser.add_argument("--rf-weight", type=float, default=1.0)
    parser.add_argument("--yolo-weight", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_CLASSES)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
