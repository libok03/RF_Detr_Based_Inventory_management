import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from detectors import RFDETRDetector


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def named_module_roots(model):
    seen = set()
    queue = [("root", model)]
    attr_names = ["model", "module", "nn_model", "net", "network", "detector", "detr", "_model"]
    while queue:
        prefix, obj = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
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
            class_name = module.__class__.__name__.lower()
            if hasattr(module, "sampling_offsets") and hasattr(module, "attention_weights") and "deform" in class_name:
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


def register_hooks(detector: RFDETRDetector, captures: list, module_index: int):
    modules = find_deformable_attention_modules(detector.model)
    if not modules:
        raise RuntimeError("No deformable attention modules with sampling_offsets were found.")
    selected = modules[module_index]

    def hook(module, inputs, output):
        locations = compute_sampling_locations(module, inputs)
        if locations is not None:
            captures.append(locations)

    handle = selected[1].register_forward_hook(hook)
    return handle, selected[0]


def points_from_locations(locations: torch.Tensor, image_w: int, image_h: int, max_points: int) -> np.ndarray:
    arr = locations.numpy().reshape(-1, 2)
    arr = arr[np.isfinite(arr).all(axis=1)]
    arr = arr[(arr[:, 0] >= 0) & (arr[:, 0] <= 1) & (arr[:, 1] >= 0) & (arr[:, 1] <= 1)]
    if len(arr) > max_points:
        idx = np.linspace(0, len(arr) - 1, max_points).astype(int)
        arr = arr[idx]
    pts = np.empty_like(arr)
    pts[:, 0] = arr[:, 0] * image_w
    pts[:, 1] = arr[:, 1] * image_h
    return pts


def iter_images(path: Path, recursive: bool):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        yield path
        return
    pattern = "**/*" if recursive else "*"
    for image_path in sorted(path.glob(pattern)):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
            yield image_path


def draw_points_only(image_rgb: np.ndarray, points: np.ndarray, radius: int, alpha: int) -> Image.Image:
    image = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x, y in points:
        draw.rectangle([x - radius, y - radius, x + radius, y + radius], fill=(35, 85, 235, alpha))
    return Image.alpha_composite(image, overlay).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Export plain RF-DETR deformable attention sampling point images.")
    parser.add_argument("--image", default="", help="Single image path.")
    parser.add_argument("--image-dir", default="", help="Directory of images.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="large", choices=["nano", "small", "medium", "base", "large"])
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--output-dir", default="implementation_outputs/rfdetr_sampling_points_plain")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--module-index", type=int, default=-1)
    parser.add_argument("--max-points", type=int, default=1800)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--point-alpha", type=int, default=170)
    args = parser.parse_args()

    os.environ["RFDETR_OPTIMIZE"] = "0"
    detector = RFDETRDetector(args.model, variant=args.variant, num_classes=60)

    if args.image:
        images = [Path(args.image)]
    else:
        images = list(iter_images(Path(args.image_dir), args.recursive))
    images = images[: args.limit]
    if not images:
        raise FileNotFoundError("No images found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(images, start=1):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"skip unreadable: {image_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        captures = []
        handle, module_name = register_hooks(detector, captures, args.module_index)
        try:
            detector.predict_rgb(image_rgb, args.conf)
        finally:
            handle.remove()
        if not captures:
            print(f"skip no capture: {image_path}")
            continue
        points = points_from_locations(captures[-1], image_rgb.shape[1], image_rgb.shape[0], args.max_points)
        out_image = draw_points_only(image_rgb, points, args.point_radius, args.point_alpha)
        out_path = output_dir / f"{idx:02d}_{image_path.stem}_sampling_points.png"
        out_image.save(out_path)
        print(out_path)


if __name__ == "__main__":
    main()
