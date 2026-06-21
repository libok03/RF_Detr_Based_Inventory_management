import argparse
import logging
import subprocess
import sys
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd):
    logger.info("Running: %s", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)


def has_video_ext(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}


def source_contains_videos(source: Path, recursive: bool) -> bool:
    if source.is_file():
        return has_video_ext(source)
    pattern = "**/*" if recursive else "*"
    return any(path.is_file() and has_video_ext(path) for path in source.glob(pattern))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run frame extraction, detection/counting, camera fusion, and temporal filtering in one command."
    )
    parser.add_argument("--source", required=True, help="Video file/directory or already extracted frame directory")
    parser.add_argument("--output-root", default="implementation_outputs/sample_allcams_rf_detr_large_aug_stride1_conf045")
    parser.add_argument("--recursive", action="store_true")

    parser.add_argument("--skip-frame-extract", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--resize-width", type=int, default=0)

    parser.add_argument(
        "--model",
        default="rf_detr_large_aug",
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
    parser.add_argument("--yolo-model", default="weights/best.pt")
    parser.add_argument("--rf-small-aug-model", default="weights/rf-detr_small_aug.pth")
    parser.add_argument("--rf-large-model", default="weights/rf-detr_large.pth")
    parser.add_argument("--rf-large-aug-model", default="weights/rf-detr_large_aug.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--rf-conf", type=float, default=0.45)
    parser.add_argument("--yolo-conf", type=float, default=0.60)
    parser.add_argument("--nms-iou", type=float, default=0.55)
    parser.add_argument("--single-nms", action="store_true", default=True)
    parser.add_argument("--no-single-nms", dest="single_nms", action="store_false")
    parser.add_argument("--soft-nms-min-score", type=float, default=0.001)
    parser.add_argument("--rf-weight", type=float, default=1.0)
    parser.add_argument("--yolo-weight", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument("--max-images", type=int, default=0)

    parser.add_argument("--skip-temporal-filter", action="store_true")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-appear", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source = Path(args.source)
    output_root = Path(args.output_root)
    frames_dir = output_root / "frames"
    inventory_dir = output_root / "inventory"

    if not source.exists():
        raise FileNotFoundError(source)

    use_frame_extract = not args.skip_frame_extract and source_contains_videos(source, args.recursive)
    if use_frame_extract:
        frame_cmd = [
            sys.executable,
            str(repo_root / "implementation" / "frame_extract.py"),
            "--source",
            str(source),
            "--output-dir",
            str(frames_dir),
            "--stride",
            str(args.frame_stride),
            "--resize-width",
            str(args.resize_width),
        ]
        if args.recursive:
            frame_cmd.append("--recursive")
        run_command(frame_cmd)
        detect_source = frames_dir
        detect_recursive = True
    else:
        logger.info("Skipping frame extraction; treating source as image/frame input.")
        detect_source = source
        detect_recursive = args.recursive

    detect_cmd = [
        sys.executable,
        str(repo_root / "implementation" / "inventory_pipeline.py"),
        "--source",
        str(detect_source),
        "--output-dir",
        str(inventory_dir),
        "--model",
        args.model,
        "--yolo-model",
        args.yolo_model,
        "--rf-small-aug-model",
        args.rf_small_aug_model,
        "--rf-large-model",
        args.rf_large_model,
        "--rf-large-aug-model",
        args.rf_large_aug_model,
        "--rf-conf",
        str(args.rf_conf),
        "--yolo-conf",
        str(args.yolo_conf),
        "--nms-iou",
        str(args.nms_iou),
        "--soft-nms-min-score",
        str(args.soft_nms_min_score),
        "--rf-weight",
        str(args.rf_weight),
        "--yolo-weight",
        str(args.yolo_weight),
        "--num-classes",
        str(args.num_classes),
        "--max-images",
        str(args.max_images),
    ]
    if args.device is not None:
        detect_cmd.extend(["--device", args.device])
    if args.single_nms:
        detect_cmd.append("--single-nms")
    if detect_recursive:
        detect_cmd.append("--recursive")
    run_command(detect_cmd)

    if not args.skip_temporal_filter:
        temporal_cmd = [
            sys.executable,
            str(repo_root / "implementation" / "temporal_filter.py"),
            "--input",
            str(inventory_dir / "per_image_counts.csv"),
            "--window",
            str(args.window),
            "--min-appear",
            str(args.min_appear),
        ]
        run_command(temporal_cmd)

    logger.info("Full pipeline finished. Outputs: %s", inventory_dir.resolve())


if __name__ == "__main__":
    main()
