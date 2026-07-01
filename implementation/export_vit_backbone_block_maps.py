import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from detectors import RFDETRDetector


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


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
        "backbone",
        "encoder",
        "body",
        "features",
        "_model",
    ]
    while queue:
        prefix, obj = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if hasattr(obj, "named_modules") and callable(getattr(obj, "named_modules")):
            yield prefix, obj
            continue

        if isinstance(obj, dict):
            for key, child in obj.items():
                if child is not None and id(child) not in seen:
                    queue.append((f"{prefix}.{key}", child))
            continue

        if isinstance(obj, (list, tuple)):
            for idx, child in enumerate(obj):
                if child is not None and id(child) not in seen:
                    queue.append((f"{prefix}.{idx}", child))
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

        try:
            attrs = vars(obj)
        except TypeError:
            attrs = {}
        for attr, child in attrs.items():
            if attr.startswith("__"):
                continue
            if child is None or id(child) in seen:
                continue
            if isinstance(child, (str, bytes, int, float, bool, Path)):
                continue
            queue.append((f"{prefix}.{attr}", child))


def find_vit_block_modules(model) -> List[Tuple[str, torch.nn.Module]]:
    def is_vit_layer(name: str, module: torch.nn.Module) -> bool:
        lname = name.lower()
        cname = module.__class__.__name__.lower()
        return (
            re.search(r"encoder\.layer\.\d+$", lname) is not None
            or ("dinov2" in cname and "layer" in cname)
            or "withregisterslayer" in cname
            or (
                hasattr(module, "attention")
                and hasattr(module, "mlp")
                and hasattr(module, "norm1")
                and hasattr(module, "norm2")
            )
            or (
                hasattr(module, "attn")
                and hasattr(module, "mlp")
                and (hasattr(module, "norm1") or hasattr(module, "ls1"))
            )
        )

    candidates: List[Tuple[str, torch.nn.Module]] = []
    for root_name, torch_model in named_module_roots(model):
        for name, module in torch_model.named_modules():
            lname = name.lower()
            is_backbone = "backbone" in lname or "dinov2" in lname or "encoder" in lname
            if is_backbone and is_vit_layer(name, module):
                candidates.append((f"{root_name}.{name}", module))
    if candidates:
        return candidates

    # Fallback for wrappers whose module names do not include "backbone".
    for root_name, torch_model in named_module_roots(model):
        for name, module in torch_model.named_modules():
            if is_vit_layer(name, module):
                candidates.append((f"{root_name}.{name}", module))
    return candidates


def list_interesting_modules(model, limit: int = 200):
    rows = []
    for root_name, torch_model in named_module_roots(model):
        for name, module in torch_model.named_modules():
            full_name = f"{root_name}.{name}"
            lname = full_name.lower()
            cname = module.__class__.__name__
            if any(key in lname for key in ["backbone", "dino", "encoder", "block", "attn"]) or any(
                key in cname.lower() for key in ["block", "attn", "dino", "backbone"]
            ):
                rows.append((full_name, cname))
                if len(rows) >= limit:
                    return rows
    return rows


def tensor_from_output(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = tensor_from_output(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = tensor_from_output(item)
            if tensor is not None:
                return tensor
    return None


def infer_grid_from_tokens(num_tokens: int, image_h: int, image_w: int):
    candidates = []
    for drop in [0, 1, 2, 4, 5, 8]:
        n = num_tokens - drop
        if n <= 0:
            continue
        for gh in range(1, int(np.sqrt(n)) + 2):
            if n % gh != 0:
                continue
            gw = n // gh
            ratio = gw / gh
            target = image_w / image_h
            candidates.append((abs(ratio - target), drop, gh, gw))
            candidates.append((abs((gh / gw) - target), drop, gw, gh))
    if not candidates:
        side = int(np.sqrt(num_tokens))
        return 0, side, max(1, num_tokens // max(1, side))
    _, drop, gh, gw = min(candidates, key=lambda row: row[0])
    return drop, gh, gw


def activation_to_map(tensor: torch.Tensor, image_h: int, image_w: int) -> np.ndarray:
    x = tensor.detach().float().cpu()
    if x.ndim == 4:
        # BCHW or BHWC
        if x.shape[1] <= 4096 and x.shape[1] >= x.shape[-1]:
            fmap = x[0].pow(2).mean(dim=0).sqrt().numpy()
        else:
            fmap = x[0].pow(2).mean(dim=-1).sqrt().numpy()
    elif x.ndim == 3:
        # BNC token embeddings.
        tokens = x[0]
        drop, gh, gw = infer_grid_from_tokens(tokens.shape[0], image_h, image_w)
        tokens = tokens[drop : drop + gh * gw]
        fmap = tokens.pow(2).mean(dim=-1).sqrt().reshape(gh, gw).numpy()
    elif x.ndim == 2:
        drop, gh, gw = infer_grid_from_tokens(x.shape[0], image_h, image_w)
        tokens = x[drop : drop + gh * gw]
        fmap = tokens.pow(2).mean(dim=-1).sqrt().reshape(gh, gw).numpy()
    else:
        raise ValueError(f"Unsupported activation shape: {tuple(x.shape)}")

    fmap = fmap - np.nanmin(fmap)
    denom = np.nanmax(fmap) + 1e-8
    fmap = np.clip(fmap / denom, 0, 1)
    fmap = cv2.resize(fmap, (image_w, image_h), interpolation=cv2.INTER_CUBIC)
    return fmap


def overlay_heatmap(image_rgb: np.ndarray, fmap: np.ndarray, alpha: float) -> Image.Image:
    heat = np.uint8(np.clip(fmap, 0, 1) * 255)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = np.uint8(np.clip((1 - alpha) * image_rgb + alpha * heat, 0, 255))
    return Image.fromarray(out)


def iter_images(path: Path, recursive: bool):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        yield path
        return
    pattern = "**/*" if recursive else "*"
    for image_path in sorted(path.glob(pattern)):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
            yield image_path


def pick_blocks(blocks, count: int):
    if len(blocks) <= count:
        return list(range(len(blocks)))
    return np.linspace(0, len(blocks) - 1, count).round().astype(int).tolist()


def main():
    parser = argparse.ArgumentParser(description="Export real ViT/DINO backbone block activation maps from RF-DETR.")
    parser.add_argument("--image", default="")
    parser.add_argument("--image-dir", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="large", choices=["nano", "small", "medium", "base", "large"])
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--output-dir", default="implementation_outputs/vit_backbone_block_maps")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit-images", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.48)
    parser.add_argument("--list-blocks", action="store_true")
    args = parser.parse_args()

    os.environ["RFDETR_OPTIMIZE"] = "0"
    detector = RFDETRDetector(args.model, variant=args.variant, num_classes=60)
    blocks = find_vit_block_modules(detector.model)
    if not blocks:
        print("No ViT/DINO backbone block modules were found. Interesting modules:")
        for name, class_name in list_interesting_modules(detector.model):
            print(f"- {name} [{class_name}]")
        raise RuntimeError("No ViT/DINO backbone block modules were found.")

    if args.list_blocks:
        for i, (name, module) in enumerate(blocks):
            print(f"{i:02d}: {name} [{module.__class__.__name__}]")

    selected_indices = pick_blocks(blocks, args.num_blocks)
    selected = [(i, *blocks[i]) for i in selected_indices]

    if args.image:
        images = [Path(args.image)]
    else:
        images = list(iter_images(Path(args.image_dir), args.recursive))[: args.limit_images]
    if not images:
        raise FileNotFoundError("No images found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"skip unreadable: {image_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        captures = {}
        handles = []

        def make_hook(index):
            def hook(module, inputs, output):
                tensor = tensor_from_output(output)
                if tensor is not None:
                    captures[index] = tensor.detach().cpu()
            return hook

        for block_index, _, module in selected:
            handles.append(module.register_forward_hook(make_hook(block_index)))

        try:
            detector.predict_rgb(image_rgb, args.conf)
        finally:
            for handle in handles:
                handle.remove()

        stem = image_path.stem
        for out_rank, (block_index, block_name, _) in enumerate(selected, start=1):
            if block_index not in captures:
                print(f"missing activation: block {block_index} {block_name}")
                continue
            fmap = activation_to_map(captures[block_index], h, w)
            out_img = overlay_heatmap(image_rgb, fmap, args.alpha)
            out_path = output_dir / f"{stem}_vit_block_{out_rank:02d}_idx{block_index:02d}.png"
            out_img.save(out_path)
            print(out_path)


if __name__ == "__main__":
    main()
