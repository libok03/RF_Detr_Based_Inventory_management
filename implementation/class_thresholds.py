import json
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from detectors import Detection
except ImportError:
    from implementation.detectors import Detection


def load_class_thresholds(path: str) -> Dict[int, float]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): float(v) for k, v in data.items()}


def threshold_floor(default_conf: float, thresholds: Dict[int, float]) -> float:
    if not thresholds:
        return default_conf
    return min(default_conf, min(thresholds.values()))


def apply_class_thresholds(
    detections: Iterable[Detection],
    thresholds: Dict[int, float],
    default_conf: float,
) -> List[Detection]:
    if not thresholds:
        return [det for det in detections if det.confidence >= default_conf]
    return [
        det
        for det in detections
        if det.confidence >= thresholds.get(det.class_id, default_conf)
    ]
