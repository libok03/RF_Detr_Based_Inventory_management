import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from label_map import LABEL_MAP, PRICE_MAP, class_name, class_price
except ImportError:
    from implementation.label_map import LABEL_MAP, PRICE_MAP, class_name, class_price


CLASS_RE = re.compile(r"^class_(\d+)$")


def class_columns(fieldnames: List[str]) -> List[str]:
    cols = [col for col in fieldnames if CLASS_RE.match(col)]
    return sorted(cols, key=lambda col: int(col.split("_")[1]))


def frame_number(event_id: str) -> int:
    match = re.search(r"frame_(\d+)", str(event_id))
    return int(match.group(1)) if match else 0


def event_sort_key(event_id: str) -> Tuple[str, int, str]:
    text = str(event_id)
    prefix = re.sub(r"_frame_\d+$", "", text)
    return prefix, frame_number(text), text


def infer_action(prev_counts: Dict[int, int], counts: Dict[int, int]) -> str:
    if not prev_counts:
        return "initial"
    delta_total = sum(counts.values()) - sum(prev_counts.values())
    if delta_total < 0:
        return "purchase"
    if delta_total > 0:
        return "return"
    return "no_change"


def total_inventory_value(counts: Dict[int, int]) -> float:
    return round(sum(count * class_price(class_id) for class_id, count in counts.items()), 2)


def load_fused_counts(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    cols = class_columns(fieldnames)
    if not cols:
        raise ValueError(f"No class_N columns found in {path}")
    rows.sort(key=lambda row: event_sort_key(str(row.get("event_id", ""))))
    return rows, cols


def build_rows(fused_rows: List[dict], class_cols: List[str], include_zero_items: bool) -> Tuple[List[dict], List[dict]]:
    long_rows: List[dict] = []
    wide_rows: List[dict] = []
    prev_counts: Dict[int, int] = {}

    for event_number, row in enumerate(fused_rows, start=1):
        counts = {
            int(col.split("_")[1]): int(float(row.get(col, 0) or 0))
            for col in class_cols
        }
        action = infer_action(prev_counts, counts)
        total_value = total_inventory_value(counts)
        total_count = sum(counts.values())

        wide_row = {
            "event_number": event_number,
            "event_id": row.get("event_id", ""),
            "action": action,
            "total_inventory_count": total_count,
            "total_inventory_value": f"{total_value:.2f}",
        }
        for class_id in sorted(LABEL_MAP):
            wide_row[class_name(class_id)] = counts.get(class_id, 0)
        wide_rows.append(wide_row)

        for class_id in sorted(LABEL_MAP):
            quantity = counts.get(class_id, 0)
            if quantity == 0 and not include_zero_items:
                continue
            long_rows.append(
                {
                    "event_number": event_number,
                    "event_id": row.get("event_id", ""),
                    "action": action,
                    "item_name": class_name(class_id),
                    "class_id": class_id,
                    "quantity_after_event": quantity,
                    "item_price": f"{class_price(class_id):.2f}",
                    "total_inventory_value": f"{total_value:.2f}",
                }
            )

        prev_counts = counts

    return long_rows, wide_rows


def korean_action(action: str) -> str:
    return {
        "initial": "초기값",
        "purchase": "구매",
        "return": "반환",
        "no_change": "변동없음",
    }.get(action, action)


def build_korean_rows(long_rows: List[dict]) -> List[dict]:
    rows = []
    for row in long_rows:
        rows.append(
            {
                "이벤트 번호": row["event_number"],
                "event_id": row["event_id"],
                "구매/반환 여부": korean_action(str(row["action"])),
                "품목명": row["item_name"],
                "class_id": row["class_id"],
                "이벤트 발생 후 상품별 재고 수량": row["quantity_after_event"],
                "상품 가격": row["item_price"],
                "이벤트 발생 후 총 재고 금액": row["total_inventory_value"],
            }
        )
    return rows


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create competition-format inventory submission CSV files.")
    parser.add_argument("--input", required=True, help="camera_fused_counts.csv or camera_fused_counts_temporal.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--long-name", default="competition_submission.csv")
    parser.add_argument("--korean-name", default="competition_submission_kr.csv")
    parser.add_argument("--wide-name", default="competition_submission_wide.csv")
    parser.add_argument(
        "--include-zero-items",
        action="store_true",
        help="Include all 60 item rows for every event. By default only non-zero inventory rows are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    df, cols = load_fused_counts(input_path)
    long_rows, wide_rows = build_rows(df, cols, args.include_zero_items)

    long_fields = [
        "event_number",
        "event_id",
        "action",
        "item_name",
        "class_id",
        "quantity_after_event",
        "item_price",
        "total_inventory_value",
    ]
    wide_fields = [
        "event_number",
        "event_id",
        "action",
        "total_inventory_count",
        "total_inventory_value",
        *[class_name(class_id) for class_id in sorted(LABEL_MAP)],
    ]

    long_path = output_dir / args.long_name
    korean_path = output_dir / args.korean_name
    wide_path = output_dir / args.wide_name
    write_csv(long_path, long_rows, long_fields)
    write_csv(
        korean_path,
        build_korean_rows(long_rows),
        [
            "이벤트 번호",
            "event_id",
            "구매/반환 여부",
            "품목명",
            "class_id",
            "이벤트 발생 후 상품별 재고 수량",
            "상품 가격",
            "이벤트 발생 후 총 재고 금액",
        ],
    )
    write_csv(wide_path, wide_rows, wide_fields)
    print(long_path)
    print(korean_path)
    print(wide_path)


if __name__ == "__main__":
    main()
