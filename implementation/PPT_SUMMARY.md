# PPT Summary: RF-DETR Based Inventory Management

## 1. Final Baseline Command

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

## 2. System Overview

- Input: 5 synchronized camera videos for each purchase/return event.
- Detector: RF-DETR Large Aug v5_b checkpoint.
- Processing unit: video frames are read directly from mp4 files and treated as image frames.
- Main output: event-level inventory CSV containing item name, event number, purchase/return action, quantity after event, and total inventory value.
- Debug outputs: per-frame counts, bbox detections, camera-fused counts, temporal counts, visualized detection videos.

## 3. AI Model Structure

- RF-DETR is used as a frame-level object detector, not a native video model.
- Each sampled frame is passed through RF-DETR, then post-processing converts bbox detections into class-wise count vectors.
- The pipeline uses class-wise confidence thresholds learned from validation detections.
- Final inventory estimation is produced by combining detection, duplicate suppression, temporal filtering, and multi-camera fusion.

## 4. Data Augmentation Strategy

- Final selected checkpoint: `output_aug_v5_b/checkpoint_best_ema.pth`.
- Multiple augmentation variants were compared: v5_a, v5_b, and v5_c.
- v5_b was selected because it offered the best practical balance between class coverage and over-detection behavior on sample videos.
- Background/scale/edge-oriented augmentation variants were explored to improve robustness against shelf background, camera angle, and object boundary ambiguity.

## 5. Multi-Camera Strategy

- The same event is grouped across cam0~cam4.
- Each camera produces a 60-dimensional class count vector.
- Camera fusion is max fusion by class:

```text
fused_count[class_id] = max(cam0_count, cam1_count, cam2_count, cam3_count, cam4_count)
```

- This is used because occlusion and viewpoint limitations can cause one camera to miss objects visible from another camera.
- Cam2 was analyzed separately because it produced more over-detection in several frames, but the final baseline includes cam2 after threshold and suppression tuning.

## 6. Preprocessing and Video Handling

- Initial implementation extracted frames to disk.
- Final implementation supports direct video inference to avoid frame image write overhead.
- Current baseline samples every 3 frames using `frame_stride=3`.
- `video_batch_frames=1` means one sampled time index is processed at a time across 5 cameras.
- Larger batch groups were tested, but very large optimized batches were unstable in the server environment.

## 7. Detection Post-Processing

- Single-model NMS:
  - `nms_iou=0.40`
  - Removes heavily overlapping boxes of the same class.
- Same-class duplicate suppression:
  - `duplicate_center_threshold=0.85`
  - `duplicate_conf_ratio=0.70`
  - Removes lower-confidence duplicate boxes whose centers are very close to a stronger same-class box.
- Containment duplicate suppression:
  - `containment_threshold=0.70`
  - `containment_conf_ratio=0.95`
  - Removes lower-confidence same-class boxes mostly contained inside a stronger box.
- These steps address cases where one product is split into two boxes or one large product receives nested boxes.

## 8. Class-Wise Threshold Optimization

- A single confidence threshold was insufficient because some classes over-detected while others needed lower confidence to preserve recall.
- Validation detections were cached at low confidence.
- A Random Forest based threshold selection procedure was used to estimate per-class confidence thresholds.
- Penalty sweep explored FP/FN trade-offs:
  - `fp_penalty`: cost for false positives.
  - `fn_penalty`: cost for false negatives.
- Selected setting from video-oriented sweep:
  - `fp_penalty=0.70`
  - `fn_penalty=0.05`
- Best threshold JSON:
  - `implementation_outputs/rf_penalty_video_sweep_v5_b_with_cam2/best_rf_class_thresholds.json`

## 9. Temporal Filtering

- Raw frame detections can flicker due to reflection, partial occlusion, blur, or detector uncertainty.
- Temporal rule:
  - `window=5`
  - `min_appear=5`
- A class count is kept only when it consistently appears across the sliding window.
- This reduces one-frame false positives while preserving persistent products.
- Temporal filtering is applied before final camera-fused count output.

## 10. Occlusion and Failure Case Handling

- Occlusion issue:
  - Some products are hidden behind others from a single camera.
  - Multi-camera max fusion recovers objects visible from another viewpoint.
- Similar packaging issue:
  - Product classes with similar wrappers can cause class confusion.
  - Class-wise thresholds and validation error analysis were used to reduce high-risk false positives.
- Reflection/lighting issue:
  - Temporal filtering reduces short-lived reflective false detections.
- Large/nested boxes:
  - Containment suppression handles lower-confidence nested boxes.

## 11. Event and Inventory Tracking

- Each fused frame/event row contains class-wise inventory counts after processing.
- Action is inferred by comparing the current fused inventory vector with the previous event vector:
  - total inventory decreased: purchase.
  - total inventory increased: return.
  - total inventory unchanged: no_change.
  - first row: initial.
- Total inventory value is calculated using the class price map.
- Submission CSV outputs:
  - `competition_submission.csv`
  - `competition_submission_kr.csv`
  - `competition_submission_kr_cp949.csv`
  - `competition_submission_kr.tsv`
  - `competition_submission_wide.csv`

## 12. Visualization Outputs

- `visualize_detection_video.py` renders `detections.csv` back onto the original mp4 videos.
- It produces camera-wise annotated videos:

```text
visualized_videos/cam0_Sample_1_detections.mp4
visualized_videos/cam1_Sample_1_detections.mp4
...
```

- Visualized videos are for debugging and presentation, not for official RTF measurement.

## 13. Performance and RTF Notes

- Sample A100 run:
  - total: 439.37 s
  - inference/count: 436.99 s
  - temporal: 0.74 s
  - submission CSV: 1.52 s
  - video duration: 238.97 s
  - RTF: 1.838615
- Bottleneck is RF-DETR inference, not temporal filtering or CSV generation.
- A6000 was observed to run this RF-DETR configuration faster than A100 in this environment.
- For official speed evaluation, the final RTF should be measured on the actual target GPU environment.

## 14. Main Lessons Learned

- RF-DETR gave strong detection quality, but false positives in specific classes required class-wise thresholding.
- Direct video reading reduced unnecessary disk I/O compared with frame extraction.
- Temporal filtering made count vectors more stable with minimal processing cost.
- Camera max fusion improved robustness against single-camera occlusion.
- Debugging with per-class overcount summaries and visualized videos was essential for identifying failure classes.

## 15. Suggested Presentation Flow

1. Problem definition: inventory recognition from 5-camera purchase/return videos.
2. Dataset and class structure: 60 product classes with price map.
3. Model: RF-DETR frame detector.
4. Pipeline: video input, RF-DETR detection, count vector, temporal filtering, camera fusion, submission CSV.
5. Multi-camera fusion: why max fusion helps occlusion.
6. Error analysis: over-detection classes and failure examples.
7. Improvements: NMS, same-class suppression, containment suppression, RF class thresholds.
8. Final outputs: submission CSV and detection visualization videos.
9. Performance: RTF breakdown and bottleneck analysis.
10. Conclusion: robust frame-level detection plus lightweight temporal/multi-camera post-processing.
