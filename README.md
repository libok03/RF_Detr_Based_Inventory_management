# RF-DETR Based Inventory Management

Unmanned-store inventory estimation pipeline using YOLO11n, RF-DETR, optional RF-DETR+YOLO ensemble, temporal filtering, and class-wise max camera fusion.

This repository intentionally contains only implementation code and lightweight documentation. It does not include datasets, model weights, generated outputs, or full report files.

## Pipeline

1. Extract frames from mp4 videos.
2. Run object detection on each frame.
3. Convert bounding boxes to 60-class count vectors.
4. Fuse cam1~cam5 counts using class-wise max fusion.
5. Optionally apply temporal appearance filtering with `window` and `min_appear`.
6. Save detection, per-frame count, temporal count, and fused inventory CSV files.

## Main Files

- `implementation/run_full_pipeline.py`: one-command runner from frame extraction to temporal filtering.
- `implementation/frame_extract.py`: mp4 to sampled frame extraction.
- `implementation/detectors.py`: YOLO/RF-DETR detector wrappers and box utilities.
- `implementation/inventory_pipeline.py`: detection, class count, and camera fusion.
- `implementation/temporal_filter.py`: sliding-window temporal count filtering.
- `implementation/README.md`: run commands and output descriptions.
- `docs/report_implementation_summary.md`: implementation notes extracted from the report draft.

## Quick Start

```bash
pip install -r requirements.txt
python implementation/run_full_pipeline.py \
  --source /path/to/4.TestVideo_Sample \
  --model rf_detr_large_aug \
  --rf-large-aug-model weights/rf-detr_large_aug.pt \
  --rf-conf 0.45 \
  --single-nms \
  --nms-iou 0.55 \
  --device 0 \
  --recursive \
  --frame-stride 1 \
  --output-root implementation_outputs/sample_allcams_rf_detr_large_aug_stride1_conf045
```

If `--source` contains mp4 videos, frames are extracted first. If it points to already extracted images, frame extraction is skipped automatically.

The current default settings in `run_full_pipeline.py` are RF-DETR Large Aug, `rf_conf=0.45`, `single_nms=True`, `nms_iou=0.55`, and `frame_stride=1`.

## Separate Steps

Run detection/counting only:

```bash
python implementation/inventory_pipeline.py \
  --source /path/to/frames \
  --model rf_detr_large_aug \
  --rf-large-aug-model weights/rf-detr_large_aug.pt \
  --rf-conf 0.45 \
  --single-nms \
  --nms-iou 0.55 \
  --device 0 \
  --recursive \
  --output-dir implementation_outputs/rf_detr_large_aug_inventory
```

Apply temporal filtering after inference:

```bash
python implementation/temporal_filter.py \
  --input implementation_outputs/rf_detr_large_aug_inventory/per_image_counts.csv \
  --window 8 \
  --min-appear 6
```

## Notes

Model weights should be placed locally or on the server and passed through command-line arguments. Large files are excluded from this repository.
