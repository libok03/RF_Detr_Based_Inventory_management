# Implementation Pipeline

This folder is for the actual unmanned-store implementation, separated from the validation/report scripts.

## 1. Extract Frames

```bash
python implementation/frame_extract.py \
  --source /path/to/videos \
  --output-dir implementation_outputs/frames \
  --stride 1 \
  --recursive
```

`--stride 1` saves every frame. Increase the stride only when reducing compute is more important than frame-level tracking.

## One-Command Full Pipeline

Run frame extraction, detection/counting, camera fusion, and temporal filtering together:

```bash
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

If `--source` contains mp4 files, frames are extracted first. If `--source` is already an image/frame directory, frame extraction is skipped automatically.

## 2. Run Detection, Count, and Camera Fusion

YOLO11n single model:

```bash
python implementation/inventory_pipeline.py \
  --source implementation_outputs/frames \
  --model yolo11n \
  --yolo-model weights/best.pt \
  --yolo-conf 0.60 \
  --device 0 \
  --recursive \
  --output-dir implementation_outputs/yolo11n_inventory
```

RF-DETR Large with single-model NMS:

```bash
python implementation/inventory_pipeline.py \
  --source implementation_outputs/frames \
  --model rf_detr_large \
  --rf-large-model weights/rf-detr_large.pth \
  --rf-conf 0.45 \
  --single-nms \
  --nms-iou 0.55 \
  --recursive \
  --output-dir implementation_outputs/rf_detr_large_inventory
```

RF-DETR Large Aug + YOLO11n ensemble:

```bash
python implementation/inventory_pipeline.py \
  --source implementation_outputs/frames \
  --model ensemble_nms \
  --rf-large-aug-model weights/rf-detr_large_aug.pt \
  --yolo-model weights/best.pt \
  --rf-conf 0.10 \
  --yolo-conf 0.60 \
  --nms-iou 0.55 \
  --device 0 \
  --recursive \
  --output-dir implementation_outputs/ensemble_inventory
```

## Outputs

- `per_image_counts.csv`: each image/frame count vector.
- `detections.csv`: bbox-level detection results.
- `camera_fused_counts.csv`: class-wise max fusion across cam1~cam5 by event id.
- `detections.json`: full debug output.

## 3. Optional Temporal Filtering

For continuous frame sequences, apply temporal appearance filtering after inference:

```bash
python implementation/temporal_filter.py \
  --input implementation_outputs/yolo11n_inventory/per_image_counts.csv \
  --window 8 \
  --min-appear 6
```

Additional outputs:

- `per_image_counts_temporal.csv`: temporally filtered frame-level count vector.
- `camera_fused_counts_temporal.csv`: camera-fused counts after temporal filtering.

## Current Recommended Thresholds

Current RF-DETR default run:

- RF-DETR Large Aug: `rf_conf=0.45`, `single_nms=True`, `nms_iou=0.55`.
- Frame extraction: `frame_stride=1`.
- Temporal filtering: `window=8`, `min_appear=6`.

The current `inventory_pipeline.py` applies image-level count and camera max fusion. Temporal `window/min_appear` smoothing should be applied on top of frame-level outputs when continuous video sequences are used.
