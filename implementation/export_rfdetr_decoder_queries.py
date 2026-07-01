import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

from detectors import RFDETRDetector


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
        "transformer",
        "decoder",
        "model_ema",
        "ema",
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


def is_decoder_layer(name: str, module: torch.nn.Module) -> bool:
    lname = name.lower()
    cname = module.__class__.__name__.lower()
    if any(skip in lname for skip in ["norm", "dropout", "linear", "embedding", "bbox_embed", "class_embed"]):
        return False
    if any(skip in cname for skip in ["norm", "dropout", "linear", "embedding"]):
        return False
    return (
        re.search(r"decoder\.(layers?|layer)\.\d+$", lname) is not None
        or "decoderlayer" in cname
        or ("decoder" in lname and "layer" in cname)
        or ("transformer" in lname and "decoder" in cname and "layer" in cname)
    )


def find_decoder_modules(model) -> List[Tuple[str, torch.nn.Module]]:
    candidates: List[Tuple[str, torch.nn.Module]] = []
    for root_name, torch_model in named_module_roots(model):
        for name, module in torch_model.named_modules():
            if is_decoder_layer(name, module):
                candidates.append((f"{root_name}.{name}", module))

    deduped = []
    seen_ids = set()
    for name, module in candidates:
        if id(module) in seen_ids:
            continue
        seen_ids.add(id(module))
        deduped.append((name, module))
    return deduped


def list_interesting_modules(model, limit: int = 250):
    rows = []
    for root_name, torch_model in named_module_roots(model):
        for name, module in torch_model.named_modules():
            full_name = f"{root_name}.{name}"
            lname = full_name.lower()
            cname = module.__class__.__name__
            if any(key in lname for key in ["decoder", "query", "transformer"]) or any(
                key in cname.lower() for key in ["decoder", "query", "transformer"]
            ):
                rows.append((full_name, cname))
                if len(rows) >= limit:
                    return rows
    return rows


def tensors_from_output(output) -> List[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (list, tuple)):
        tensors: List[torch.Tensor] = []
        for item in output:
            tensors.extend(tensors_from_output(item))
        return tensors
    if isinstance(output, dict):
        tensors = []
        for item in output.values():
            tensors.extend(tensors_from_output(item))
        return tensors
    if hasattr(output, "__dict__"):
        tensors = []
        for item in vars(output).values():
            tensors.extend(tensors_from_output(item))
        return tensors
    return []


def looks_like_query_tensor(tensor: torch.Tensor, num_queries: int) -> bool:
    if tensor.ndim < 3:
        return False
    shape = tuple(tensor.shape)
    return num_queries in shape and max(shape) >= num_queries


def to_query_matrix(tensor: torch.Tensor, num_queries: int) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    arr = np.squeeze(arr)

    if arr.ndim == 4:
        # Common shape: decoder_layers x batch x queries x channels.
        query_axes = [idx for idx, size in enumerate(arr.shape) if size == num_queries]
        if query_axes:
            q_axis = query_axes[-1]
            layer_axis = 0 if q_axis != 0 else 1
            arr = np.take(arr, indices=arr.shape[layer_axis] - 1, axis=layer_axis)
            arr = np.squeeze(arr)

    if arr.ndim == 3:
        query_axes = [idx for idx, size in enumerate(arr.shape) if size == num_queries]
        if not query_axes:
            raise ValueError(f"Cannot find query axis in shape {arr.shape}")
        q_axis = query_axes[-1]
        arr = np.moveaxis(arr, q_axis, 0)
        arr = arr.reshape(num_queries, -1)
    elif arr.ndim == 2:
        if arr.shape[0] != num_queries and arr.shape[1] == num_queries:
            arr = arr.T
        if arr.shape[0] != num_queries:
            raise ValueError(f"Cannot convert tensor with shape {arr.shape} to query matrix")
    else:
        raise ValueError(f"Cannot convert tensor with shape {arr.shape} to query matrix")

    return arr.astype(np.float32, copy=False)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")[:180]


def write_query_summary(path: Path, query_matrix: np.ndarray) -> None:
    norms = np.linalg.norm(query_matrix, axis=1)
    abs_mean = np.mean(np.abs(query_matrix), axis=1)
    means = np.mean(query_matrix, axis=1)
    maxs = np.max(query_matrix, axis=1)
    mins = np.min(query_matrix, axis=1)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "l2_norm", "abs_mean", "mean", "min", "max"])
        for idx in range(query_matrix.shape[0]):
            writer.writerow([idx, norms[idx], abs_mean[idx], means[idx], mins[idx], maxs[idx]])


def write_detection_csv(path: Path, detections) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "confidence", "x1", "y1", "x2", "y2"])
        for det in detections:
            writer.writerow([det.class_id, det.confidence, det.x1, det.y1, det.x2, det.y2])


def main():
    parser = argparse.ArgumentParser(description="Export RF-DETR decoder query tensors for one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", default="large", choices=["nano", "small", "medium", "base", "large"])
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--num-classes", type=int, default=60)
    parser.add_argument("--output-dir", default="implementation_outputs/rfdetr_decoder_queries")
    parser.add_argument("--list-modules", action="store_true")
    parser.add_argument("--module-index", default="all", help="'all', 'last', or a comma-separated index list")
    args = parser.parse_args()

    os.environ["RFDETR_OPTIMIZE"] = "0"
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detector = RFDETRDetector(args.model, variant=args.variant, num_classes=args.num_classes)
    modules = find_decoder_modules(detector.model)
    if not modules:
        print("No decoder layer modules were found. Interesting modules:")
        for name, class_name in list_interesting_modules(detector.model):
            print(f"- {name} [{class_name}]")
        raise RuntimeError("No decoder layer modules were found.")

    if args.list_modules:
        for idx, (name, module) in enumerate(modules):
            print(f"{idx:02d}: {name} [{module.__class__.__name__}]")

    if args.module_index == "all":
        selected_indices = list(range(len(modules)))
    elif args.module_index == "last":
        selected_indices = [len(modules) - 1]
    else:
        selected_indices = [int(item.strip()) for item in args.module_index.split(",") if item.strip()]

    captures: Dict[int, List[torch.Tensor]] = {}
    handles = []

    def make_hook(index: int):
        def hook(module, inputs, output):
            tensors = [t.detach().cpu() for t in tensors_from_output(output) if looks_like_query_tensor(t, args.num_queries)]
            if tensors:
                captures[index] = tensors

        return hook

    for index in selected_indices:
        name, module = modules[index]
        handles.append(module.register_forward_hook(make_hook(index)))

    try:
        detections = detector.predict_rgb(image_rgb, args.conf)
    finally:
        for handle in handles:
            handle.remove()

    write_detection_csv(output_dir / f"{image_path.stem}_detections.csv", detections)

    if not captures:
        raise RuntimeError("Decoder modules were found, but no query-shaped tensors were captured.")

    manifest_path = output_dir / f"{image_path.stem}_decoder_query_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        manifest = csv.writer(manifest_file)
        manifest.writerow(["module_index", "module_name", "tensor_index", "tensor_shape", "npy_path", "summary_csv_path"])

        for module_index in selected_indices:
            if module_index not in captures:
                print(f"missing decoder capture: {module_index:02d} {modules[module_index][0]}")
                continue
            module_name = modules[module_index][0]
            module_safe = safe_name(f"{module_index:02d}_{module_name}")
            for tensor_index, tensor in enumerate(captures[module_index]):
                query_matrix = to_query_matrix(tensor, args.num_queries)
                npy_path = output_dir / f"{image_path.stem}_{module_safe}_tensor{tensor_index}_queries.npy"
                csv_path = output_dir / f"{image_path.stem}_{module_safe}_tensor{tensor_index}_summary.csv"
                np.save(npy_path, query_matrix)
                write_query_summary(csv_path, query_matrix)
                manifest.writerow(
                    [
                        module_index,
                        module_name,
                        tensor_index,
                        "x".join(str(dim) for dim in tuple(tensor.shape)),
                        npy_path,
                        csv_path,
                    ]
                )
                print(f"saved: {npy_path} shape={query_matrix.shape}")
                print(f"saved: {csv_path}")

    print(f"saved: {manifest_path}")
    print(f"saved: {output_dir / (image_path.stem + '_detections.csv')}")


if __name__ == "__main__":
    main()
