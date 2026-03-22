#!/usr/bin/env python3
"""Summarize evaluation logs from JSONL/CSV into a sortable comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def to_float(value: Any, default: float = float("inf")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["source_file"] = str(path)
            rows.append(row)
    return rows


def read_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["source_file"] = str(path)
            rows.append(row)
    return rows


def discover_log_files(root: Path, patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    unique_files = sorted(set(p for p in files if p.is_file()))
    return unique_files


def load_rows(files: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows.extend(read_jsonl(path))
        elif suffix == ".csv":
            rows.extend(read_csv(path))
    return rows


def rank_rows(rows: List[Dict[str, Any]], sort_by: str, descending: bool) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: to_float(r.get(sort_by)), reverse=descending)


def format_table(rows: List[Dict[str, Any]], top_k: int) -> str:
    view = rows[:top_k] if top_k > 0 else rows

    headers = [
        "rank",
        "perplexity",
        "avg_loss",
        "eval_used_samples",
        "model_name",
        "lora_path",
        "eval_dataset_name",
        "eval_split",
        "timestamp_utc",
        "source_file",
    ]

    rendered: List[Dict[str, str]] = []
    for idx, row in enumerate(view, start=1):
        rendered.append(
            {
                "rank": str(idx),
                "perplexity": f"{to_float(row.get('perplexity')):.4f}"
                if row.get("perplexity") is not None
                else "n/a",
                "avg_loss": f"{to_float(row.get('avg_loss')):.4f}"
                if row.get("avg_loss") is not None
                else "n/a",
                "eval_used_samples": str(row.get("eval_used_samples", "n/a")),
                "model_name": str(row.get("model_name", "n/a")),
                "lora_path": str(row.get("lora_path", "n/a")),
                "eval_dataset_name": str(row.get("eval_dataset_name", "n/a")),
                "eval_split": str(row.get("eval_split", "n/a")),
                "timestamp_utc": str(row.get("timestamp_utc", "n/a")),
                "source_file": str(row.get("source_file", "n/a")),
            }
        )

    widths = {h: len(h) for h in headers}
    for row in rendered:
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))

    lines = []
    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep_line = "-+-".join("-" * widths[h] for h in headers)
    lines.append(header_line)
    lines.append(sep_line)

    for row in rendered:
        lines.append(" | ".join(row[h].ljust(widths[h]) for h in headers))

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize eval logs and rank checkpoints")
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory to scan recursively for log files.",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["eval_metrics.jsonl", "*.eval.jsonl", "*eval*.csv", "eval_metrics.csv"],
        help="Glob-like filename patterns to include in recursive scan.",
    )
    parser.add_argument(
        "--sort-by",
        default="perplexity",
        help="Numeric field to rank by (for example perplexity or avg_loss).",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order (default is ascending).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Max rows to print (0 prints all).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()

    files = discover_log_files(root, args.patterns)
    if not files:
        print("No matching log files found.")
        return 1

    rows = load_rows(files)
    if not rows:
        print("No rows found in discovered log files.")
        return 1

    ranked = rank_rows(rows, sort_by=args.sort_by, descending=args.descending)
    print(f"Found {len(rows)} log rows across {len(files)} file(s).")
    print(f"Sorted by: {args.sort_by} ({'desc' if args.descending else 'asc'})")
    print()
    print(format_table(ranked, top_k=args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
