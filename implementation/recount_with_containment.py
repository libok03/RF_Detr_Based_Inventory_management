import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from detectors import Detection
    from inventory_pipeline import (
        containment_duplicate_suppression,
        fuse_camera_counts,
        write_csv,
        write_long_count_csv,
        write_long_fused_csv,
    )
except ImportError:
    from implementation.detectors import Detection
    from implementation.inventory_pipeline import (
        containment_duplicate_suppression,
        fuse_camera_counts,
        write_csv,
        write_long_count_csv,
        write_long_fused_csv,
    )


def class_columns(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("class_")], key=lambda c: int(c.split("_")[1]))


def row_to_detection(row: pd.Series) -> Detection:
    return Detection(
        class_id=int(row["class_id"]),
        confidence=float(row["confidence"]),
        x1=float(row["x1"]),
        y1=float(row["y1"]),
        x2=float(row["x2"]),
        y2=float(row["y2"]),
        image_w=max(int(float(row["x2"])) + 1, 1),
        image_h=max(int(float(row["y2"])) + 1, 1),
        source=str(row.get("source", "rfdetr")),
    )


def count_classes(detections: List[Detection], num_classes: int) -> Dict[int, int]:
    counts = {class_id: 0 for class_id in range(num_classes)}
    for det in detections:
        if 0 <= det.class_id < num_classes:
            counts[det.class_id] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Recount existing detections after containment-based duplicate suppression.")
    parser.add_argument("--input-dir", required=True, help="Inventory output directory containing detections.csv and per_image_counts.csv.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--containment-threshold", type=float, default=0.70)
    parser.add_argument("--containment-conf-ratio", type=float, default=0.95)
    parser.add_argument("--exclude-count-cameras", default="")
    parser.add_argument("--num-classes", type=int, default=60)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per = pd.read_csv(input_dir / "per_image_counts.csv")
    det = pd.read_csv(input_dir / "detections.csv")
    class_cols = [f"class_{i}" for i in range(args.num_classes)]

    grouped: Dict[str, List[Detection]] = {}
    metadata: Dict[str, Tuple[str, str, str]] = {}
    filtered_rows = []

    for _, row in det.iterrows():
        image = str(row["image"])
        grouped.setdefault(image, []).append(row_to_detection(row))
        metadata[image] = (str(row["event_id"]), str(row["camera"]), str(row["model"]))

    filtered_by_image: Dict[str, List[Detection]] = {}
    for image, detections in grouped.items():
        filtered = containment_duplicate_suppression(
            detections,
            args.containment_threshold,
            args.containment_conf_ratio,
        )
        filtered_by_image[image] = filtered
        event_id, camera, model = metadata[image]
        for d in filtered:
            filtered_rows.append(
                {
                    "image": image,
                    "event_id": event_id,
                    "camera": camera,
                    "model": model,
                    "class_id": d.class_id,
                    "confidence": f"{d.confidence:.6f}",
                    "x1": f"{d.x1:.2f}",
                    "y1": f"{d.y1:.2f}",
                    "x2": f"{d.x2:.2f}",
                    "y2": f"{d.y2:.2f}",
                    "source": d.source,
                }
            )

    per_rows = []
    for _, row in per.iterrows():
        image = str(row["image"])
        filtered = filtered_by_image.get(image, [])
        counts = count_classes(filtered, args.num_classes)
        out = {
            "image": image,
            "event_id": row["event_id"],
            "camera": row["camera"],
            "model": row["model"],
            "total_count": sum(counts.values()),
            "latency_ms": row.get("latency_ms", ""),
        }
        out.update({col: counts[int(col.split("_")[1])] for col in class_cols})
        per_rows.append(out)

    write_csv(output_dir / "detections.csv", filtered_rows, ["image", "event_id", "camera", "model", "class_id", "confidence", "x1", "y1", "x2", "y2", "source"])
    write_csv(output_dir / "per_image_counts.csv", per_rows, ["image", "event_id", "camera", "model", "total_count", "latency_ms", *class_cols])
    fused_rows = fuse_camera_counts(
        per_rows,
        args.num_classes,
        {x.strip().lower() for x in args.exclude_count_cameras.split(",") if x.strip()},
    )
    write_csv(output_dir / "camera_fused_counts.csv", fused_rows, ["event_id", "model", "num_cameras", "fused_total_count", *class_cols])
    write_long_count_csv(output_dir / "per_image_counts_long.csv", per_rows, args.num_classes)
    write_long_fused_csv(output_dir / "camera_fused_counts_long.csv", fused_rows, args.num_classes)

    before = len(det)
    after = len(filtered_rows)
    summary = pd.DataFrame(
        [
            {
                "before_detections": before,
                "after_detections": after,
                "removed_detections": before - after,
                "containment_threshold": args.containment_threshold,
                "containment_conf_ratio": args.containment_conf_ratio,
            }
        ]
    )
    summary.to_csv(output_dir / "containment_summary.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
