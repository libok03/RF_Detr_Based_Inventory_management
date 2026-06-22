#!/usr/bin/env bash
set -euo pipefail

SOURCE="${SOURCE:-$HOME/Dataset/4.TestVideo_Sample}"
RF_MODEL="${RF_MODEL:-$HOME/RF-DETR/output_aug_v3_clean_dataset/checkpoint_best_ema.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-implementation_outputs/rf_conf_nms_binary_sweep}"
DEVICE="${DEVICE:-0}"
FPS="${FPS:-25}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
DUP_CENTER="${DUP_CENTER:-0.85}"
DUP_RATIO="${DUP_RATIO:-0.70}"

CONFS="${CONFS:-0.35 0.45 0.55}"
NMS_IOUS="${NMS_IOUS:-0.35 0.55 0.75}"
WINDOWS="${WINDOWS:-5}"
MINS="${MINS:-5}"

mkdir -p "$OUTPUT_ROOT"

FRAMES_DIR="$OUTPUT_ROOT/frames"
if [ ! -d "$FRAMES_DIR" ] || [ -z "$(find "$FRAMES_DIR" -type f -name '*.jpg' -print -quit)" ]; then
  python implementation/frame_extract.py \
    --source "$SOURCE" \
    --output-dir "$FRAMES_DIR" \
    --stride "$FRAME_STRIDE" \
    --recursive
fi

RUN_DIRS=()
for CONF in $CONFS; do
  for NMS in $NMS_IOUS; do
    CONF_TAG="$(printf '%03d' "$(python - <<PY
print(round(float("$CONF") * 100))
PY
)")"
    NMS_TAG="$(printf '%03d' "$(python - <<PY
print(round(float("$NMS") * 100))
PY
)")"
    RUN_ROOT="$OUTPUT_ROOT/rf_conf${CONF_TAG}_nms${NMS_TAG}"
    INV_DIR="$RUN_ROOT/inventory"
    RUN_DIRS+=("$INV_DIR")

    if [ -f "$INV_DIR/per_image_counts.csv" ]; then
      echo "skip existing $INV_DIR"
      continue
    fi

    python implementation/inventory_pipeline.py \
      --source "$FRAMES_DIR" \
      --output-dir "$INV_DIR" \
      --model rf_detr_large_aug \
      --rf-large-aug-model "$RF_MODEL" \
      --rf-conf "$CONF" \
      --single-nms \
      --nms-iou "$NMS" \
      --duplicate-center-threshold "$DUP_CENTER" \
      --duplicate-conf-ratio "$DUP_RATIO" \
      --device "$DEVICE" \
      --recursive
  done
done

python implementation/sweep_binary_temporal.py \
  --runs "${RUN_DIRS[@]}" \
  --windows "$WINDOWS" \
  --mins "$MINS" \
  --fps "$FPS" \
  --output "$OUTPUT_ROOT/binary_temporal_sweep_summary.csv" \
  --save-temporal

echo "$OUTPUT_ROOT/binary_temporal_sweep_summary.csv"
