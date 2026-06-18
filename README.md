# RF-DETR Based Inventory Management

Unmanned-store inventory estimation pipeline using YOLO11n, RF-DETR, optional RF-DETR+YOLO ensemble, temporal filtering, and class-wise max camera fusion.

This repository intentionally contains only implementation code and lightweight documentation. It does not include datasets, model weights, generated outputs, or full report files.

## Pipeline

1. Extract frames from mp4 videos.
2. Run object detection on each frame.
3. Convert bounding boxes to 60-class count vectors.
4. Optionally apply temporal appearance filtering with `window` and `min_appear`.
5. Fuse cam1~cam5 counts using class-wise max fusion.
6. Save detection, per-frame count, and fused inventory CSV files.

## Main Files

- `implementation/frame_extract.py`: mp4 to sampled frame extraction.
- `implementation/detectors.py`: YOLO/RF-DETR detector wrappers and box utilities.
- `implementation/inventory_pipeline.py`: detection, class count, and camera fusion.
- `implementation/temporal_filter.py`: sliding-window temporal count filtering.
- `implementation/README.md`: run commands and output descriptions.
- `docs/report_implementation_summary.md`: implementation notes extracted from the report draft.

## Quick Start

```bash
pip install -r requirements.txt
python implementation/inventory_pipeline.py \
  --source /path/to/frames \
  --model yolo11n \
  --yolo-model /path/to/best.pt \
  --yolo-conf 0.60 \
  --device 0 \
  --recursive \
  --output-dir implementation_outputs/yolo11n_inventory
```

For continuous video sequences, apply temporal filtering after inference:

```bash
python implementation/temporal_filter.py \
  --input implementation_outputs/yolo11n_inventory/per_image_counts.csv \
  --window 8 \
  --min-appear 6
```

## Notes

Model weights should be placed locally or on the server and passed through command-line arguments. Large files are excluded from this repository.
