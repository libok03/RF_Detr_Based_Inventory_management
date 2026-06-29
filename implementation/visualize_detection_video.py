import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2

try:
    from label_map import class_name
except ImportError:
    from implementation.label_map import class_name


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
CAM_RE = re.compile(r"cam\d+", re.IGNORECASE)
FRAME_RE = re.compile(r"(.+)_frame_(\d+)\.jpg$", re.IGNORECASE)


def iter_videos(source: Path, recursive: bool) -> Iterable[Path]:
    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        yield source
        return
    pattern = "**/*" if recursive else "*"
    for path in sorted(source.glob(pattern)):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


def camera_name(path: Path) -> str:
    parent = path.parent.name
    if CAM_RE.fullmatch(parent):
        return parent.lower()
    match = CAM_RE.search(path.stem)
    return match.group(0).lower() if match else "unknown"


def event_name(path: Path) -> str:
    return path.stem


def group_videos(source: Path, recursive: bool) -> Dict[str, List[Tuple[str, Path]]]:
    grouped: Dict[str, List[Tuple[str, Path]]] = {}
    for video in iter_videos(source, recursive):
        grouped.setdefault(event_name(video), []).append((camera_name(video), video))
    return {event: sorted(items, key=lambda item: item[0]) for event, items in sorted(grouped.items())}


def parse_detection_image(image: str) -> Tuple[str, str, int]:
    match = FRAME_RE.match(image)
    if not match:
        return "unknown", Path(image).stem, -1
    prefix = match.group(1)
    frame_idx = int(match.group(2))
    cam_match = CAM_RE.search(prefix)
    camera = cam_match.group(0).lower() if cam_match else "unknown"
    event = prefix
    if camera != "unknown":
        event = re.sub(rf"(^|_){camera}(_|$)", lambda m: m.group(1) if m.group(2) == "" else m.group(1), prefix, count=1, flags=re.IGNORECASE).strip("_")
    return camera, event, frame_idx


def load_detections(path: Path) -> Dict[Tuple[str, str, int], List[dict]]:
    by_frame: Dict[Tuple[str, str, int], List[dict]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            camera, event, frame_idx = parse_detection_image(row.get("image", ""))
            if frame_idx < 0:
                continue
            by_frame.setdefault((event, camera, frame_idx), []).append(row)
    return by_frame


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    palette = [
        (40, 40, 255),
        (40, 180, 40),
        (255, 120, 20),
        (220, 40, 220),
        (40, 220, 220),
        (220, 220, 40),
        (120, 80, 255),
        (80, 220, 120),
    ]
    return palette[class_id % len(palette)]


def draw_detections(frame, detections: List[dict], title: str, show_conf: bool) -> None:
    cv2.putText(frame, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    for det in detections:
        class_id = int(float(det["class_id"]))
        conf = float(det.get("confidence", 0.0) or 0.0)
        x1 = int(float(det["x1"]))
        y1 = int(float(det["y1"]))
        x2 = int(float(det["x2"]))
        y2 = int(float(det["y2"]))
        color = color_for_class(class_id)
        label = f"c{class_id} {class_name(class_id)}"
        if show_conf:
            label = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        y_text = max(18, y1 - 6)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y_text - th - 4), (x1 + tw + 4, y_text + 2), color, -1)
        cv2.putText(frame, label, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def write_annotated_video(
    *,
    event: str,
    camera: str,
    video_path: Path,
    detections: Dict[Tuple[str, str, int], List[dict]],
    output_dir: Path,
    hold_detections: bool,
    show_conf: bool,
    max_frames: int,
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{camera}_{event}_detections.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    last_dets: List[dict] = []

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        current = detections.get((event, camera, frame_idx))
        if current is not None:
            last_dets = current
        draw_detections(
            frame,
            last_dets if hold_detections else (current or []),
            f"{camera} | {event} | frame {frame_idx:06d}",
            show_conf,
        )
        writer.write(frame)

    cap.release()
    writer.release()
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render detection CSV results back onto source videos.")
    parser.add_argument("--source", required=True, help="Video file or directory containing cam folders.")
    parser.add_argument("--detections", required=True, help="detections.csv from the inventory pipeline.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--no-hold-detections", dest="hold_detections", action="store_false")
    parser.set_defaults(hold_detections=True)
    parser.add_argument("--hide-conf", dest="show_conf", action="store_false")
    parser.set_defaults(show_conf=True)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    detections_path = Path(args.detections)
    output_dir = Path(args.output_dir) if args.output_dir else detections_path.parent / "visualized_videos"

    detections = load_detections(detections_path)
    groups = group_videos(source, args.recursive)
    if not groups:
        raise FileNotFoundError(f"No videos found: {source}")

    for event, videos in groups.items():
        for camera, video_path in videos:
            out = write_annotated_video(
                event=event,
                camera=camera,
                video_path=video_path,
                detections=detections,
                output_dir=output_dir,
                hold_detections=args.hold_detections,
                show_conf=args.show_conf,
                max_frames=args.max_frames,
            )
            print(out)


if __name__ == "__main__":
    main()
