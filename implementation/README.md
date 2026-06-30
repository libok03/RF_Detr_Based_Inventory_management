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

### Final Baseline

Current baseline for the sample multi-camera video submission run:

```bash
python implementation/run_full_pipeline.py \
  --source ~/Dataset/4.TestVideo_Sample \
  --direct-video-inference \
  --model rf_detr_large_aug \
  --rf-large-aug-model ~/RF-DETR/output_aug_v5_b/checkpoint_best_ema.pth \
  --class-thresholds implementation_outputs/rf_penalty_video_sweep_v5_b_with_cam2/best_rf_class_thresholds.json \
  --rf-conf 0.60 \
  --single-nms \
  --nms-iou 0.40 \
  --duplicate-center-threshold 0.85 \
  --duplicate-conf-ratio 0.70 \
  --containment-threshold 0.70 \
  --containment-conf-ratio 0.95 \
  --recursive \
  --frame-stride 3 \
  --video-batch-frames 1 \
  --window 5 \
  --min-appear 5 \
  --output-root implementation_outputs/final_submission_v5_b_stride3
```

This run reads mp4 files directly and performs RF-DETR inference, class-wise thresholding, NMS, duplicate suppression, temporal filtering, camera max fusion, and submission CSV generation in one command.

### Generic Example

```bash
python implementation/run_full_pipeline.py \
  --source /path/to/4.TestVideo_Sample \
  --model rf_detr_large_aug \
  --rf-large-aug-model weights/rf-detr_large_aug.pt \
  --rf-conf 0.45 \
  --single-nms \
  --nms-iou 0.55 \
  --duplicate-center-threshold 0.85 \
  --duplicate-conf-ratio 0.65 \
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
  --duplicate-center-threshold 0.85 \
  --duplicate-conf-ratio 0.65 \
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
  --window 5 \
  --min-appear 5
```

Additional outputs:

- `per_image_counts_temporal.csv`: temporally filtered frame-level count vector.
- `camera_fused_counts_temporal.csv`: camera-fused counts after temporal filtering.

## Competition Submission CSV

`run_full_pipeline.py` also writes submission-format CSV files in the inventory output folder:

- `competition_submission.csv`: long format with `event_number`, `event_id`, `action`, `item_name`, `class_id`, `quantity_after_event`, `item_price`, and `total_inventory_value`.
- `competition_submission_kr.csv`: Korean long format with the required fields: item name, event number, purchase/return action, quantity after event, and total inventory value.
- `competition_submission_kr_cp949.csv`: Korean CSV encoded for Windows Excel.
- `competition_submission_kr.tsv`: Korean tab-separated file for spreadsheet import.
- `competition_submission_wide.csv`: one row per event with all product quantities as columns.

`action` is inferred by comparing the current fused inventory vector with the previous event vector:

- `purchase`: total inventory decreased.
- `return`: total inventory increased.
- `no_change`: total inventory did not change.
- `initial`: first event row.

To regenerate submission files from an existing fused count CSV:

```bash
python implementation/make_submission_csv.py \
  --input implementation_outputs/sample_allcams_v5_b/inventory/camera_fused_counts_temporal.csv \
  --output-dir implementation_outputs/sample_allcams_v5_b/inventory \
  --include-zero-items
```

## Detection Visualization Videos

To render `detections.csv` back onto the original camera videos:

```bash
python implementation/visualize_detection_video.py \
  --source /path/to/4.TestVideo_Sample \
  --detections implementation_outputs/final_submission_v5_b_stride3/inventory/detections.csv \
  --output-dir implementation_outputs/final_submission_v5_b_stride3/inventory/visualized_videos \
  --recursive
```

Or enable it at the end of the full pipeline:

```bash
python implementation/run_full_pipeline.py \
  ... \
  --make-visualization-video
```

The visualized mp4 files are for inspection and presentation. Leave `--make-visualization-video` off when measuring official RTF.

## Current Recommended Thresholds

Current final baseline:

- RF-DETR Large Aug v5_b checkpoint: `~/RF-DETR/output_aug_v5_b/checkpoint_best_ema.pth`.
- Per-class RF confidence thresholds: `implementation_outputs/rf_penalty_video_sweep_v5_b_with_cam2/best_rf_class_thresholds.json`.
- RF confidence floor: `rf_conf=0.60`.
- Single-model NMS: `nms_iou=0.40`.
- Same-class duplicate suppression: `duplicate_center_threshold=0.85`, `duplicate_conf_ratio=0.70`.
- Containment duplicate suppression: `containment_threshold=0.70`, `containment_conf_ratio=0.95`.
- Direct video inference: `direct_video_inference=True`, `video_batch_frames=1`.
- Frame sampling: `frame_stride=3`.
- Temporal filtering: `window=5`, `min_appear=5`.

The current pipeline applies image-level count extraction, camera max fusion, and temporal `window/min_appear` smoothing. Same-class duplicate suppression reduces split detections of one product, while containment duplicate suppression removes lower-confidence boxes mostly contained by a higher-confidence same-class box.
