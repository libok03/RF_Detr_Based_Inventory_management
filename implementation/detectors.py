import logging
import os
import warnings
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

warnings.filterwarnings("ignore", category=FutureWarning, module=r"deprecate\..*")
warnings.filterwarnings("ignore", message=r".*A new version of Albumentations is available.*")
warnings.filterwarnings("ignore", message=r".*Converting a tensor to a Python boolean might cause the trace to be incorrect.*")
warnings.filterwarnings("ignore", message=r".*`loss_type=None` was set in the config.*")
warnings.filterwarnings("ignore", message=r".*`use_return_dict` is deprecated.*")
try:
    from torch.jit import TracerWarning

    warnings.filterwarnings("ignore", category=TracerWarning)
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("rf-detr").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("albumentations").setLevel(logging.ERROR)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@contextmanager
def quiet_stderr(enabled: bool = True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull, redirect_stderr(devnull):
        yield


@dataclass
class Detection:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_w: int
    image_h: int
    source: str

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def clipped(self) -> "Detection":
        return Detection(
            class_id=self.class_id,
            confidence=self.confidence,
            x1=float(np.clip(self.x1, 0, self.image_w - 1)),
            y1=float(np.clip(self.y1, 0, self.image_h - 1)),
            x2=float(np.clip(self.x2, 0, self.image_w - 1)),
            y2=float(np.clip(self.y2, 0, self.image_h - 1)),
            image_w=self.image_w,
            image_h=self.image_h,
            source=self.source,
        )


def iter_images(source: Path, recursive: bool) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() in IMAGE_EXTS:
            yield source
        return

    pattern = "**/*" if recursive else "*"
    for path in sorted(source.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def fuse_cluster(cluster: List[Detection], model_weights: Dict[str, float]) -> Detection:
    weights = np.array(
        [det.confidence * model_weights.get(det.source, 1.0) for det in cluster],
        dtype=np.float32,
    )
    boxes = np.array([det.xyxy for det in cluster], dtype=np.float32)
    fused_box = boxes.mean(axis=0) if float(weights.sum()) <= 0 else np.average(boxes, axis=0, weights=weights)
    confidence = max(det.confidence for det in cluster)
    first = cluster[0]
    return Detection(
        class_id=first.class_id,
        confidence=float(confidence),
        x1=float(fused_box[0]),
        y1=float(fused_box[1]),
        x2=float(fused_box[2]),
        y2=float(fused_box[3]),
        image_w=first.image_w,
        image_h=first.image_h,
        source="ensemble",
    ).clipped()


def weighted_boxes_fusion(
    detections: List[Detection],
    iou_threshold: float,
    model_weights: Dict[str, float],
) -> List[Detection]:
    fused: List[Detection] = []
    for class_id in sorted({det.class_id for det in detections}):
        class_dets = [det for det in detections if det.class_id == class_id]
        class_dets.sort(
            key=lambda det: det.confidence * model_weights.get(det.source, 1.0),
            reverse=True,
        )
        clusters: List[List[Detection]] = []

        for det in class_dets:
            best_idx = -1
            best_iou = 0.0
            for idx, cluster in enumerate(clusters):
                overlap = iou(det.xyxy, fuse_cluster(cluster, model_weights).xyxy)
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx

            if best_idx >= 0 and best_iou >= iou_threshold:
                clusters[best_idx].append(det)
            else:
                clusters.append([det])

        fused.extend(fuse_cluster(cluster, model_weights) for cluster in clusters)

    return sorted(fused, key=lambda det: (det.class_id, -det.confidence))


class RFDETRDetector:
    def __init__(
        self,
        model_path: str,
        variant: str = "large",
        optimize_batch_size: Optional[int] = None,
        num_classes: int = 60,
    ):
        self.model_path = os.path.abspath(model_path)
        self.variant = variant
        self.optimize_batch_size = optimize_batch_size
        self.num_classes = num_classes
        self.model = self._load_model()
        self._optimize_for_inference()

    def _optimize_for_inference(self) -> None:
        if os.environ.get("RFDETR_OPTIMIZE", "1").lower() in {"0", "false"}:
            return
        optimize = getattr(self.model, "optimize_for_inference", None)
        if callable(optimize):
            logger.debug("Optimizing RF-DETR model for inference")
            quiet_optimize = os.environ.get("RFDETR_QUIET_OPTIMIZE", "1").lower() not in {"0", "false"}
            with warnings.catch_warnings(), quiet_stderr(quiet_optimize):
                warnings.simplefilter("ignore")
                if self.optimize_batch_size:
                    try:
                        optimize(batch_size=self.optimize_batch_size)
                        return
                    except TypeError:
                        logger.debug("RF-DETR optimize_for_inference does not accept batch_size")
                optimize()

    def _load_model(self):
        logger.debug("Loading RF-DETR model: %s", self.model_path)
        class_names = {
            "base": "RFDETRBase",
            "large": "RFDETRLarge",
            "nano": "RFDETRNano",
            "small": "RFDETRSmall",
            "medium": "RFDETRMedium",
        }
        class_name = class_names.get(self.variant.lower(), "RFDETRLarge")

        try:
            import rfdetr

            cls = getattr(rfdetr, class_name)
            try:
                return cls(pretrain_weights=self.model_path, num_classes=self.num_classes)
            except TypeError:
                try:
                    model = cls(num_classes=self.num_classes)
                except TypeError:
                    model = cls()
                if hasattr(model, "load"):
                    model.load(self.model_path)
                    return model
                raise
        except Exception as direct_exc:
            logger.debug("Direct RF-DETR class load failed: %s", direct_exc)

        try:
            from rfdetr import RFDETR

            if hasattr(RFDETR, "from_checkpoint"):
                try:
                    return RFDETR.from_checkpoint(self.model_path, num_classes=self.num_classes)
                except TypeError:
                    return RFDETR.from_checkpoint(self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load RF-DETR model. model={self.model_path}, variant={self.variant}, error={exc}"
            ) from exc

        raise RuntimeError(f"Failed to load RF-DETR model. model={self.model_path}, variant={self.variant}")

    def predict(self, image_path: Path, conf: float) -> List[Detection]:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return []

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.predict_rgb(image_rgb, conf)

    def _prediction_to_detections(self, pred, image_w: int, image_h: int) -> List[Detection]:
        detections = []
        for box, score, class_id in zip(np.asarray(pred.xyxy), np.asarray(pred.confidence), np.asarray(pred.class_id)):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    confidence=float(score),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    image_w=image_w,
                    image_h=image_h,
                    source="rfdetr",
                ).clipped()
            )
        return detections

    def predict_rgb(self, image_rgb: np.ndarray, conf: float) -> List[Detection]:
        h, w = image_rgb.shape[:2]
        pred = self.model.predict(image_rgb, threshold=conf)
        return self._prediction_to_detections(pred, w, h)

    def predict_batch_rgb(self, images_rgb: Sequence[np.ndarray], conf: float) -> List[List[Detection]]:
        if not images_rgb:
            return []
        preds = self.model.predict(list(images_rgb), threshold=conf)
        if len(images_rgb) == 1 and not isinstance(preds, (list, tuple)):
            preds = [preds]
        return [
            self._prediction_to_detections(pred, image.shape[1], image.shape[0])
            for pred, image in zip(preds, images_rgb)
        ]


class YOLODetector:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.model_path = os.path.abspath(model_path)
        self.device = device
        logger.info("Loading YOLO model: %s", self.model_path)
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)

    def predict(self, image_path: Path, conf: float) -> List[Detection]:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return []

        h, w = image_bgr.shape[:2]
        kwargs = {"conf": conf, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(str(image_path), **kwargs)[0]
        if result.boxes is None:
            return []

        detections = []
        for box, score, class_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        ):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    confidence=float(score),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    image_w=w,
                    image_h=h,
                    source="yolo",
                ).clipped()
            )
        return detections
