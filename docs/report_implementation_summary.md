# Report-Derived Implementation Summary

This document keeps only the implementation-relevant content from the report draft. The full Word/PDF report is intentionally not included in the repository.

## Problem Scope

The system estimates unmanned-store inventory from multi-camera shelf videos. The practical implementation target is not only object detection, but class-wise product count estimation and inventory state estimation.

## Input Structure

- Input videos are mp4 files captured by cam1~cam5.
- Frames are extracted before detection.
- Each frame is processed independently by the selected detector.
- File names containing `cam1`~`cam5` are grouped by event id for camera fusion.

## Detection Models

The implementation supports:

- YOLO11n single-model inference.
- RF-DETR Small_Aug, RF-DETR Large, and RF-DETR Large_Aug single-model inference.
- RF-DETR + YOLO11n ensemble using WBF, NMS, or Soft-NMS.

Model weights are not committed to the repository. They should be passed through CLI arguments.

## Count Vector Conversion

Post-processed detections are converted into a 60-class count vector. For each image and each class `c`, the count is the number of bounding boxes predicted as class `c`. Classes not detected in the frame are filled with zero.

## Temporal Filtering

Single-frame predictions can fluctuate due to lighting, occlusion, hand movement, and camera noise. The implementation provides sliding-window temporal filtering:

- `window`: number of recent frames considered.
- `min_appear`: minimum number of appearances required inside the window.

Recommended stable setting from GPU sweep results:

- `conf=0.60`
- `window=8`
- `min_appear=6`

A100 low-FN setting:

- `conf=0.60`
- `window=4`
- `min_appear=3`

A6000 low-FN setting:

- `conf=0.60`
- `window=8`
- `min_appear=6`

## Class-Wise Max Camera Fusion

The system does not directly merge bounding box coordinates across cameras because cam1~cam5 have different viewpoints. Instead, each camera output is converted into a class-wise count vector first.

For class `c`, camera `k` count is denoted as `n_{k,c}`. The final camera-fused count is:

```text
N_c = max(n_1c, n_2c, n_3c, n_4c, n_5c)
```

This design reduces under-counting from occlusion because a class hidden in one camera may be visible in another.

## Outputs

The implementation writes:

- `detections.csv`: bounding-box level predictions.
- `per_image_counts.csv`: per-frame class-wise count vectors.
- `camera_fused_counts.csv`: class-wise max-fused inventory counts.
- `detections.json`: detailed debug output.
- `per_image_counts_temporal.csv` and `camera_fused_counts_temporal.csv` when temporal filtering is applied.

## Important Limitation

Max camera fusion can reduce under-counting but can amplify false positives, because the maximum count across cameras is selected. Therefore confidence thresholding, NMS, and temporal filtering are important for stable inventory estimation.
