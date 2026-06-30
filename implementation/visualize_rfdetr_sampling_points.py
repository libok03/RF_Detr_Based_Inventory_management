import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from detectors import RFDETRDetector


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def named_module_roots(model):
    seen = set()
    queue = [("root", model)]
    attr_names = [
        "model",
        "module",
        "nn_model",
        "net",
        "network",
        "detector",
        "detr",
        "model_ema",
        "ema",
        "_model",
    ]

    while queue:
        prefix, obj = queue.pop(0)
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if hasattr(obj, "named_modules") and callable(getattr(obj, "named_modules")):
            yield prefix, obj
            continue

        for attr in attr_names:
            if not hasattr(obj, attr):
                continue
            try:
                child = getattr(obj, attr)
            except Exception:
                continue
            if child is not None and id(child) not in seen:
                queue.append((f"{prefix}.{attr}", child))


def find_deformable_attention_modules(model) -> List[Tuple[str, torch.nn.Module]]:
    modules = []
    for root_name, torch_model in named_module_roots(model):
        for name, module in torch_model.named_modules():
            has_offsets = hasattr(module, "sampling_offsets")
            has_weights = hasattr(module, "attention_weights")
            class_name = module.__class__.__name__.lower()
            if has_offsets and has_weights and "deform" in class_name:
                modules.append((f"{root_name}.{name}", module))
    return modules


def compute_sampling_locations(module, inputs):
    if len(inputs) < 5:
        return None

    query = inputs[0]
    reference_points = inputs[1]
    spatial_shapes = inputs[3]
    if query is None or reference_points is None or spatial_shapes is None:
        return None

    with torch.no_grad():
        batch_size, num_queries = query.shape[:2]
        num_heads = getattr(module, "n_heads", getattr(module, "num_heads", None))
        num_levels = getattr(module, "n_levels", getattr(module, "num_levels", None))
        num_points = getattr(module, "n_points", getattr(module, "num_points", None))
        if not all([num_heads, num_levels, num_points]):
            return None

        offsets = module.sampling_offsets(query)
        offsets = offsets.view(batch_size, num_queries, num_heads, num_levels, num_points, 2)

        if reference_points.shape[-1] == 2:
            normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], -1)
            locations = reference_points[:, :, None, :, None, :] + offsets / normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            locations = (
                reference_points[:, :, None, :, None, :2]
                + offsets / num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            return None

        return locations.detach().float().cpu()


def register_sampling_hooks(detector: RFDETRDetector, store: list):
    handles = []
    modules = find_deformable_attention_modules(detector.model)
    if not modules:
        raise RuntimeError("No deformable attention modules with sampling_offsets were found.")

    def make_hook(name):
        def hook(module, inputs, output):
            locations = compute_sampling_locations(module, inputs)
            if locations is not None:
                store.append((name, locations))
        return hook

    for name, module in modules:
        handles.append(module.register_forward_hook(make_hook(name)))
    return handles, modules


def points_from_locations(locations: torch.Tensor, image_w: int, image_h: int, max_points: int) -> np.ndarray:
    arr = locations.numpy()
    arr = arr.reshape(-1, 2)
    arr = arr[np.isfinite(arr).all(axis=1)]
    arr = arr[(arr[:, 0] >= 0) & (arr[:, 0] <= 1) & (arr[:, 1] >= 0) & (arr[:, 1] <= 1)]
    if len(arr) > max_points:
        idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
        arr = arr[idx]
    pts = np.empty_like(arr)
    pts[:, 0] = arr[:, 0] * image_w
    pts[:, 1] = arr[:, 1] * image_h
    return pts


def draw_points(draw: ImageDraw.ImageDraw, points: np.ndarray, color, radius: int = 2):
    for x, y in points:
        draw.rectangle([x - radius, y - radius, x + radius, y + radius], fill=color)


def draw_detections(draw: ImageDraw.ImageDraw, detections, color=(255, 40, 40)):
    for det in detections:
        draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline=color, width=3)
        label = f"c{det.class_id} {det.confidence:.2f}"
        draw.text((det.x1 + 2, max(2, det.y1 - 16)), label, fill=color, font=font(13, True))


def draw_header(draw: ImageDraw.ImageDraw, text: str, xy=(14, 12)):
    f = font(21, True)
    bbox = draw.textbbox(xy, text, font=f)
    draw.rounded_rectangle([bbox[0] - 8, bbox[1] - 6, bbox[2] + 8, bbox[3] + 6], radius=8, fill=(255, 255, 255))
    draw.text(xy, text, fill=(20, 30, 45), font=f)


def make_relation_panel(image: Image.Image, detections):
    panel = image.copy()
    draw = ImageDraw.Draw(panel, "RGBA")
    centers = []
    for det in detections:
        cx = (det.x1 + det.x2) / 2
        cy = (det.y1 + det.y2) / 2
        centers.append((det.class_id, cx, cy, det.confidence, det))
        draw.rectangle([det.x1, det.y1, det.x2, det.y2], outline=(255, 45, 45), width=3)

    centers = sorted(centers, key=lambda row: row[3], reverse=True)[:10]
    for i, (_, x1, y1, _, _) in enumerate(centers):
        for _, x2, y2, _, _ in centers[i + 1 : i + 3]:
            draw.line([x1, y1, x2, y2], fill=(20, 170, 95, 180), width=3)
    for class_id, x, y, conf, _ in centers:
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(20, 170, 95))
        draw.text((x + 6, y - 8), f"c{class_id}", fill=(20, 110, 70), font=font(13, True))

    draw_header(draw, "Object relation cues (not explicit local sampling)")
    return panel


def main():
    parser = argparse.ArgumentParser(description="Visualize RF-DETR deformable attention sampling points on a dataset image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="large", choices=["nano", "small", "medium", "base", "large"])
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--output", default="report_single_figures/figure_rfdetr_sampling_points.png")
    parser.add_argument("--module-index", type=int, default=-1, help="Which deformable attention hook output to visualize. -1 uses the last one.")
    parser.add_argument("--max-points", type=int, default=1800)
    args = parser.parse_args()

    # Hooks do not survive the traced/optimized inference wrapper reliably.
    os.environ["RFDETR_OPTIMIZE"] = "0"

    image_path = Path(args.image)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)

    detector = RFDETRDetector(args.model, variant=args.variant, num_classes=60)
    captures = []
    handles, modules = register_sampling_hooks(detector, captures)
    try:
        detections = detector.predict_rgb(image_rgb, args.conf)
    finally:
        for handle in handles:
            handle.remove()

    if not captures:
        raise RuntimeError(f"Hooks were registered on {len(modules)} modules, but no sampling locations were captured.")

    module_name, locations = captures[args.module_index]
    pts = points_from_locations(locations, image.width, image.height, args.max_points)

    sampling_panel = image.copy()
    draw = ImageDraw.Draw(sampling_panel, "RGBA")
    draw_points(draw, pts, (55, 95, 225, 150), radius=2)
    draw_detections(draw, detections)
    draw_header(draw, "Actual RF-DETR deformable attention samples")
    draw.text((16, 44), f"module: {module_name} | points shown: {len(pts)}", fill=(35, 45, 65), font=font(14))

    relation_panel = make_relation_panel(image, detections)

    gap = 26
    top = 70
    canvas = Image.new("RGB", (image.width * 2 + gap * 3, image.height + top + 28), (246, 248, 252))
    cdraw = ImageDraw.Draw(canvas)
    cdraw.text((gap, 20), "RF-DETR on unmanned-store dataset", fill=(25, 35, 55), font=font(26, True))
    cdraw.text(
        (gap, 50),
        "Deformable attention gives adaptive sampling points; inventory scenes still benefit from explicit object relation reasoning.",
        fill=(75, 85, 105),
        font=font(15),
    )
    canvas.paste(sampling_panel, (gap, top))
    canvas.paste(relation_panel, (image.width + gap * 2, top))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()
