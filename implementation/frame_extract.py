import argparse
import logging
from pathlib import Path

import cv2


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def iter_videos(source: Path, recursive: bool):
    if source.is_file() and source.suffix.lower() in VIDEO_EXTS:
        yield source
        return

    pattern = "**/*" if recursive else "*"
    for path in sorted(source.glob(pattern)):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


def make_video_key(video_path: Path) -> str:
    parent = video_path.parent.name
    if parent.lower().startswith("cam"):
        return f"{parent}_{video_path.stem}"
    return video_path.stem


def extract_video(video_path: Path, output_dir: Path, stride: int, resize_width: int = 0) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return 0

    video_key = make_video_key(video_path)
    stem_dir = output_dir / video_key
    stem_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride == 0:
            if resize_width > 0:
                h, w = frame.shape[:2]
                scale = resize_width / float(w)
                frame = cv2.resize(frame, (resize_width, int(h * scale)), interpolation=cv2.INTER_AREA)

            out_path = stem_dir / f"{video_key}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    logger.info("Extracted %d frames from %s", saved, video_path.name)
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="Extract sampled frames from mp4 videos.")
    parser.add_argument("--source", required=True, help="Video file or directory")
    parser.add_argument("--output-dir", default="implementation_outputs/frames")
    parser.add_argument("--stride", type=int, default=30, help="Save every Nth frame")
    parser.add_argument("--resize-width", type=int, default=0, help="Optional output width; 0 keeps original")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output_dir)
    videos = list(iter_videos(source, args.recursive))
    if not videos:
        raise FileNotFoundError(f"No videos found: {source}")

    total = 0
    for video in videos:
        total += extract_video(video, output_dir, max(1, args.stride), args.resize_width)
    logger.info("Done. Total saved frames: %d", total)


if __name__ == "__main__":
    main()
