import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def read_rows(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    numeric_cols = {
        "rf_conf",
        "nms_iou",
        "window",
        "min_appear",
        "mean_latency_ms",
        "inference_rtf",
        "raw_mean_total",
        "temporal_mean_total",
        "temporal_max_total",
        "temporal_mean_abs_delta",
        "raw_mean_abs_delta",
    }
    for row in rows:
        for col in numeric_cols:
            if col in row and row[col] != "":
                row[col] = float(row[col])
    return rows


def group_by_conf(rows: Iterable[Dict[str, object]]) -> List[Dict[str, float]]:
    grouped: Dict[float, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(float(row["rf_conf"]), []).append(row)

    summary = []
    for conf, items in sorted(grouped.items()):
        best_stability = min(float(row["temporal_mean_abs_delta"]) for row in items)
        best_latency = min(float(row["mean_latency_ms"]) for row in items)
        mean_total = sum(float(row["temporal_mean_total"]) for row in items) / len(items)
        max_total = min(float(row["temporal_max_total"]) for row in items)
        summary.append(
            {
                "rf_conf": conf,
                "best_stability": best_stability,
                "best_latency": best_latency,
                "mean_total": mean_total,
                "min_max_total": max_total,
            }
        )
    return summary


def heatmap(rows: List[Dict[str, object]], value_col: str, title: str, out_path: Path, cmap: str = "viridis") -> None:
    confs = sorted({float(row["rf_conf"]) for row in rows})
    nms_values = sorted({float(row["nms_iou"]) for row in rows})
    value_map = {(float(row["rf_conf"]), float(row["nms_iou"])): float(row[value_col]) for row in rows}

    matrix = [[value_map.get((conf, nms), float("nan")) for nms in nms_values] for conf in confs]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("NMS IoU")
    ax.set_ylabel("RF-DETR confidence")
    ax.set_xticks(range(len(nms_values)), [f"{v:.2f}" for v in nms_values])
    ax.set_yticks(range(len(confs)), [f"{v:.2f}" for v in confs])

    for y, conf in enumerate(confs):
        for x, nms in enumerate(nms_values):
            value = value_map.get((conf, nms))
            if value is not None:
                ax.text(x, y, f"{value:.3f}", ha="center", va="center", fontsize=8, color="white")

    fig.colorbar(image, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def line_by_conf(summary: List[Dict[str, float]], out_path: Path) -> None:
    confs = [row["rf_conf"] for row in summary]
    stability = [row["best_stability"] for row in summary]
    mean_total = [row["mean_total"] for row in summary]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(confs, stability, marker="o", linewidth=2.2, color="#2563eb", label="Temporal mean abs delta")
    ax1.set_xlabel("RF-DETR confidence")
    ax1.set_ylabel("Temporal mean abs delta", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(confs, mean_total, marker="s", linewidth=2.2, color="#dc2626", label="Temporal mean total")
    ax2.set_ylabel("Temporal mean total count", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")

    fig.suptitle("Confidence Sweep: Count Stability vs Mean Count", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def latency_scatter(rows: List[Dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    scatter = ax.scatter(
        [float(row["mean_latency_ms"]) for row in rows],
        [float(row["temporal_mean_abs_delta"]) for row in rows],
        c=[float(row["rf_conf"]) for row in rows],
        s=75,
        cmap="plasma",
        edgecolors="#111827",
        linewidths=0.35,
    )
    ax.set_title("Latency vs Count Stability", fontsize=14, weight="bold")
    ax.set_xlabel("Mean latency (ms/image)")
    ax.set_ylabel("Temporal mean abs delta")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="RF-DETR confidence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def top_table(rows: List[Dict[str, object]], out_path: Path, top_k: int) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["temporal_mean_abs_delta"]),
            float(row["temporal_max_total"]),
            float(row["mean_latency_ms"]),
        ),
    )[:top_k]

    columns = ["run_name", "rf_conf", "nms_iou", "mean_latency_ms", "inference_rtf", "temporal_mean_total", "temporal_max_total", "temporal_mean_abs_delta"]
    display = []
    for row in ordered:
        display.append(
            [
                str(row.get("run_name", "")),
                f'{float(row["rf_conf"]):.2f}',
                f'{float(row["nms_iou"]):.2f}',
                f'{float(row["mean_latency_ms"]):.2f}',
                f'{float(row["inference_rtf"]):.2f}',
                f'{float(row["temporal_mean_total"]):.2f}',
                f'{float(row["temporal_max_total"]):.0f}',
                f'{float(row["temporal_mean_abs_delta"]):.4f}',
            ]
        )

    fig, ax = plt.subplots(figsize=(13, 0.52 * len(display) + 1.6))
    ax.axis("off")
    table = ax.table(cellText=display, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    for (row_idx, _), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#334155")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f8fafc")
    ax.set_title("Top RF-DETR Conf/NMS Settings", fontsize=14, weight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot binary temporal sweep summary CSV.")
    parser.add_argument("--input", required=True, help="current_binary_temporal_sweep_summary.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found: {input_path}")

    summary = group_by_conf(rows)
    heatmap(rows, "temporal_mean_abs_delta", "Temporal Count Stability by RF Confidence and NMS", output_dir / "heatmap_temporal_mean_abs_delta.png", cmap="magma_r")
    heatmap(rows, "temporal_mean_total", "Temporal Mean Total Count by RF Confidence and NMS", output_dir / "heatmap_temporal_mean_total.png", cmap="viridis")
    heatmap(rows, "mean_latency_ms", "Mean Latency by RF Confidence and NMS", output_dir / "heatmap_mean_latency_ms.png", cmap="cividis")
    line_by_conf(summary, output_dir / "confidence_stability_line.png")
    latency_scatter(rows, output_dir / "latency_vs_stability_scatter.png")
    top_table(rows, output_dir / "top_settings_table.png", args.top_k)

    print(output_dir)


if __name__ == "__main__":
    main()
