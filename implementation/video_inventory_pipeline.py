import argparse
import csv
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[1]
try:
    from detectors import Detection, RFDETRDetector
    from inventory_pipeline import (
        count_classes,
        containment_duplicate_suppression,
        fuse_camera_counts,
        nms_by_class,
        parse_camera_exclusions,
        same_class_duplicate_suppression,
        write_csv,
        write_long_count_csv,
        write_long_fused_csv,
    )
except ImportError:
    from implementation.detectors import Detection, RFDETRDetector
    from implementation.inventory_pipeline import (
        count_classes,
        containment_duplicate_suppression,
        fuse_camera_counts,
        nms_by_class,
        parse_camera_exclusions,
        same_class_duplicate_suppression,
        write_csv,
        write_long_count_csv,
        write_long_fused_csv,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
CAM_RE = re.compile(r"^cam\d+$", re.IGNORECASE)


def iter_videos(source: Path, recursive: bool) -> Iterable[Path]:
    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        yield source
        return
    pattern = "**/*" if recursive else "*"
    for path in sorted(source.glob(pattern)):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


def camera_name(video_path: Path) -> str:
    parent = video_path.parent.name
    if CAM_RE.match(parent):
        return parent.lower()
    match = re.search(r"(cam\d+)", video_path.stem, re.IGNORECASE)
    return match.group(1).lower() if match else "unknown"


def event_name(video_path: Path) -> str:
    return video_path.stem


def group_videos(source: Path, recursive: bool) -> Dict[str, List[Tuple[str, Path]]]:
    grouped: Dict[str, List[Tuple[str, Path]]] = {}
    for video in iter_videos(source, recursive):
        grouped.setdefault(event_name(video), []).append((camera_name(video), video))
    return {
        event: sorted(items, key=lambda item: item[0])
        for event, items in sorted(grouped.items())
    }


def postprocess_rf(detections: Sequence[Detection], args: argparse.Namespace) -> List[Detection]:
    dets = list(detections)
    if args.single_nms:
        dets = nms_by_class(dets, args.nms_iou)
    if args.duplicate_center_threshold > 0:
        dets = same_class_duplicate_suppression(
            dets,
            args.duplicate_center_threshold,
            args.duplicate_conf_ratio,
        )
    if args.containment_threshold > 0:
        dets = containment_duplicate_suppression(
            dets,
            args.containment_threshold,
            args.containment_conf_ratio,
        )
    return dets


def append_rows(
    *,
    image_name: str,
    event_id: str,
    camera: str,
    model: str,
    detections: List[Detection],
    latency_ms: float,
    num_classes: int,
    per_image_rows: List[Dict[str, object]],
    detection_rows: List[Dict[str, object]],
    details: List[Dict[str, object]],
) -> None:
    counts = count_classes(detections, num_classes)
    total_count = sum(counts.values())
    row = {
        "image": image_name,
        "event_id": event_id,
        "camera": camera,
        "model": model,
        "total_count": total_count,
        "latency_ms": f"{latency_ms:.3f}",
    }
    row.update({f"class_{class_id}": counts.get(class_id, 0) for class_id in range(num_classes)})
    per_image_rows.append(row)

    for det in detections:
        detection_rows.append(
            {
                "image": image_name,
                "event_id": event_id,
                "camera": camera,
                "model": model,
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
            "image": image_name,
            "event_id": event_id,
            "camera": camera,
            "model": model,
            "latency_ms": latency_ms,
            "class_counts": {str(k): v for k, v in counts.items() if v},
            "detections": [asdict(det) for det in detections],
        }
    )


def run(args: argparse.Namespace) -> None:
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model not in {"rf_detr_small_aug", "rf_detr_large", "rf_detr_large_aug"}:
        raise ValueError("Direct video batch inference currently supports RF-DETR single models only.")

    groups = group_videos(source, args.recursive)
    if not groups:
        raise FileNotFoundError(f"No videos found: {source}")

    model_path, variant = {
        "rf_detr_small_aug": (args.rf_small_aug_model, "small"),
        "rf_detr_large": (args.rf_large_model, "large"),
        "rf_detr_large_aug": (args.rf_large_aug_model, "large"),
    }[args.model]
    max_batch = max(len(items) for items in groups.values())
    detector = RFDETRDetector(model_path, variant=variant, optimize_batch_size=max_batch)

    per_image_rows: List[Dict[str, object]] = []
    detection_rows: List[Dict[str, object]] = []
    details: List[Dict[str, object]] = []
    total_sampled = 0

    for event, videos in groups.items():
        caps = []
        try:
            for camera, path in videos:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open video: {path}")
                caps.append((camera, path, cap))

            frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) for _, _, cap in caps]
            total_frames = min(frame_counts) if frame_counts else 0
            if args.max_frames > 0:
                total_frames = min(total_frames, args.max_frames)
            logger.info("Event %s: %d cameras, %d frames, stride=%d", event, len(caps), total_frames, args.frame_stride)

            for frame_idx in range(total_frames):
                frames_rgb = []
                cameras = []
                valid = True
                for camera, _, cap in caps:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        valid = False
                        break
                    if frame_idx % args.frame_stride == 0:
                        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                        cameras.append(camera)
                if not valid:
                    break
                if frame_idx % args.frame_stride != 0:
                    continue

                start = time.perf_counter()
                batch_dets = detector.predict_batch_rgb(frames_rgb, args.rf_conf)
                batch_latency_ms = (time.perf_counter() - start) * 1000.0
                per_image_latency_ms = batch_latency_ms / max(len(batch_dets), 1)
                event_id = f"{event}_frame_{frame_idx:06d}"
                total_sampled += len(batch_dets)

                for camera, dets in zip(cameras, batch_dets):
                    dets = postprocess_rf(dets, args)
                    image_name = f"{camera}_{event}_frame_{frame_idx:06d}.jpg"
                    append_rows(
                        image_name=image_name,
                        event_id=event_id,
                        camera=camera,
                        model=args.model,
                        detections=dets,
                        latency_ms=per_image_latency_ms,
                        num_classes=args.num_classes,
                        per_image_rows=per_image_rows,
                        detection_rows=detection_rows,
                        details=details,
                    )

                if total_sampled <= max_batch or frame_idx % (args.frame_stride * 100) == 0:
                    logger.info(
                        "%s frame=%06d batch=%.2fms per_image=%.2fms",
                        event,
                        frame_idx,
                        batch_latency_ms,
                        per_image_latency_ms,
                    )
        finally:
            for _, _, cap in caps:
                cap.release()

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
    fused_rows = fuse_camera_counts(
        per_image_rows,
        args.num_classes,
        parse_camera_exclusions(args.exclude_count_cameras),
    )
    write_csv(
        output_dir / "camera_fused_counts.csv",
        fused_rows,
        ["event_id", "model", "num_cameras", "fused_total_count"] + class_cols,
    )
    write_long_count_csv(output_dir / "per_image_counts_long.csv", per_image_rows, args.num_classes)
    write_long_fused_csv(output_dir / "camera_fused_counts_long.csv", fused_rows, args.num_classes)
    with (output_dir / "detections.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    logger.info("Saved direct video outputs to %s", output_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct multi-camera video RF-DETR inventory pipeline.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", default="implementation_outputs/inventory")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--model", default="rf_detr_large_aug", choices=["rf_detr_small_aug", "rf_detr_large", "rf_detr_large_aug"])
    parser.add_argument("--rf-small-aug-model", default=str(ROOT / "server_eval_package" / "weights" / "rf-detr_small_aug.pth"))
    parser.add_argument("--rf-large-model", default=str(ROOT / "server_eval_package" / "weights" / "rf-detr_large.pth"))
    parser.add_argument("--rf-large-aug-model", default=str(ROOT / "server_eval_package" / "weights" / "rf-detr_large_aug.pt"))
    parser.add_argument("--rf-conf", type=float, default=0.60)
    parser.add_argument("--nms-iou", type=float, default=0.40)
    parser.add_argument("--single-nms", action="store_true", default=True)
    parser.add_argument("--no-single-nms", dest="single_nms", action="store_false")
    parser.add_argument("--duplicate-center-threshold", type=float, default=0.85)
    parser.add_argument("--duplicate-conf-ratio", type=float, default=0.70)
    parser.add_argument("--containment-threshold", type=float, default=0.0)
    parser.add_argument("--containment-conf-ratio", type=float, default=0.95)
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument(
        "--exclude-count-cameras",
        default="",
        help="Comma-separated cameras to exclude from camera-fused counting, e.g. cam2.",
    )
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
