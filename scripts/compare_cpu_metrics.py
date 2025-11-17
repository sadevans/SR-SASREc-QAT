#!/usr/bin/env python3
"""Aggregate CPU evaluation logs and summarize PSNR/SSIM vs. latency trade-offs for Super-Resolution."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CPU metrics across SR models.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["results/*cpu_eval.json"],
        help="Glob(s) pointing to CPU evaluation JSON files.",
    )
    parser.add_argument(
        "--baseline",
        default="espcn_fp32",
        help="Model name used as the PSNR/latency reference.",
    )
    parser.add_argument(
        "--csv-out",
        default="results/cpu_metrics_summary.csv",
        help="Where to store the CSV summary.",
    )
    parser.add_argument(
        "--md-out",
        default="results/cpu_metrics_summary.md",
        help="Where to store the Markdown table.",
    )
    return parser.parse_args()


def collect_files(patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        for match in glob.glob(pattern):
            path = Path(match)
            if path.is_file():
                files.append(path)
    if not files:
        raise SystemExit(f"No input files found for patterns: {patterns}")
    return sorted(files)


def load_record(path: Path) -> Dict[str, float | str]:
    payload = json.loads(path.read_text())
    run_name = payload.get("run_name", path.stem)
    
    # Infer quant method from run_name if not present
    quant_method = payload.get("quant_method", "fp32")
    if quant_method == "fp32":
        if "lsq" in run_name:
            quant_method = "LSQ"
        elif "pact" in run_name:
            quant_method = "PACT"
        elif "apot" in run_name:
            quant_method = "APoT"
        elif "qdrop" in run_name:
            quant_method = "QDrop"
        elif "adaround" in run_name:
            quant_method = "AdaRound"
        elif "fp32" in run_name:
            quant_method = "FP32"
        else:
            quant_method = "FP32"

    return {
        "model": run_name,
        "quant_method": quant_method,
        "psnr": float(payload["psnr_y"]),
        "throughput_sps": float(payload["throughput_samples_per_sec"]),
        "avg_latency_ms": float(payload["avg_latency_ms"]),
        "median_latency_ms": float(payload.get("median_latency_ms", payload["avg_latency_ms"])),
    }


def annotate_against_baseline(records: List[Dict[str, float]], baseline_name: str):
    baseline = next((rec for rec in records if rec["model"] == baseline_name), None)
    if baseline is None:
        print(f"[warn] baseline '{baseline_name}' not found; skipping delta columns.", file=sys.stderr)
        for rec in records:
            rec["psnr_delta"] = None
            # rec["ssim_delta"] = None
            rec["throughput_speedup"] = None
            rec["latency_ratio"] = None
        return

    for rec in records:
        rec["psnr_delta"] = rec["psnr"] - baseline["psnr"]
        # rec["ssim_delta"] = rec["ssim"] - baseline["ssim"]
        rec["throughput_speedup"] = rec["throughput_sps"] / baseline["throughput_sps"]
        rec["latency_ratio"] = baseline["avg_latency_ms"] / rec["avg_latency_ms"]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(records: Sequence[Dict[str, float]], path: Path, columns: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for rec in records:
            row = []
            for col in columns:
                value = rec.get(col)
                if value is None:
                    row.append("")
                else:
                    row.append(f"{value}")
            handle.write(",".join(row) + "\n")


def fmt_value(column: str, value) -> str:
    if value is None:
        return "-"
    numeric_formats = {
        "psnr": "{:.4f}",               # PSNR-Y
        "throughput_sps": "{:.2f}",     # samples per second
        "avg_latency_ms": "{:.3f}",     # average latency in ms
        "median_latency_ms": "{:.3f}",  # optional
        "psnr_delta": "{:+.4f}",        # delta vs FP32 baseline
        "throughput_speedup": "{:.2f}", # speedup vs baseline
        "latency_ratio": "{:.2f}",      # baseline_latency / current_latency
    }
    if column in numeric_formats:
        try:
            return numeric_formats[column].format(float(value))
        except (TypeError, ValueError):
            return "-"
    return str(value)


def render_table(records: Sequence[Dict[str, float]], columns: Sequence[str]) -> str:
    widths = {col: len(col) for col in columns}
    for rec in records:
        for col in columns:
            widths[col] = max(widths[col], len(fmt_value(col, rec.get(col))))

    def render_row(values: Iterable[str]) -> str:
        parts = []
        for col, value in zip(columns, values):
            parts.append(value.ljust(widths[col]))
        return " | ".join(parts)

    lines = [render_row(columns)]
    lines.append(" | ".join("-" * widths[col] for col in columns))
    for rec in records:
        lines.append(render_row(fmt_value(col, rec.get(col)) for col in columns))
    return "\n".join(lines)


def write_markdown(records: Sequence[Dict[str, float]], path: Path, columns: Sequence[str]) -> None:
    ensure_parent(path)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for rec in records:
        values = [fmt_value(col, rec.get(col)) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    files = collect_files(args.inputs)
    records = [load_record(path) for path in files]
    annotate_against_baseline(records, args.baseline)
    columns = [
        "model",
        "quant_method",
        "psnr",
        "throughput_sps",
        "avg_latency_ms",
        "psnr_delta",
        "throughput_speedup",
        "latency_ratio",
    ]
    csv_path = Path(args.csv_out)
    write_csv(records, csv_path, columns)
    md_path = Path(args.md_out)
    write_markdown(records, md_path, columns)
    print(render_table(records, columns))
    print(f"\nSaved CSV to {csv_path} and Markdown to {md_path}")


if __name__ == "__main__":
    main()
