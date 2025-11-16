#!/usr/bin/env python3
"""
Aggregate CPU benchmarking results for Super Resolution (ESPCN).
Reads all `results/*_cpu_eval.json`, computes deltas vs. FP32 baseline,
and outputs CSV + Markdown table.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd


def load_cpu_results(pattern: str):
    files = glob.glob(pattern)
    records = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            records.append(data)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="results/*_cpu_eval.json")
    parser.add_argument("--baseline", default="espcn_fp32")
    parser.add_argument("--output-csv", default="results/cpu_metrics_summary_sr.csv")
    parser.add_argument("--output-md", default="results/cpu_metrics_summary_sr.md")
    args = parser.parse_args()

    df = load_cpu_results(args.inputs)
    if df.empty:
        print("No CPU eval files found!")
        return

    # Ensure required columns
    required = {"run_name", "psnr_y", "avg_latency_ms", "throughput_samples_per_sec", "device"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns in input JSONs: {missing}")

    # Filter CPU results only
    df = df[df["device"] == "cpu"].copy()
    df = df.set_index("run_name")

    if args.baseline not in df.index:
        raise ValueError(f"Baseline '{args.baseline}' not found in results. Available: {list(df.index)}")

    baseline = df.loc[args.baseline]
    df["psnr_y_delta"] = df["psnr_y"] - baseline["psnr_y"]
    df["latency_speedup"] = baseline["avg_latency_ms"] / df["avg_latency_ms"]
    df["throughput_speedup"] = df["throughput_samples_per_sec"] / baseline["throughput_samples_per_sec"]

    # Reorder and round
    cols = [
        "psnr_y", "psnr_y_delta",
        "avg_latency_ms", "latency_speedup",
        "throughput_samples_per_sec", "throughput_speedup"
    ]
    df_out = df[cols].round(3)

    # Save CSV
    df_out.to_csv(args.output_csv)
    print(f"Saved CSV to {args.output_csv}")

    # Generate Markdown table
    md_table = df_out.to_markdown(tablefmt="pipe", floatfmt=".3f")
    with open(args.output_md, "w") as f:
        f.write("# CPU Benchmarking: ESPCN Super Resolution\n\n")
        f.write(f"Baseline: `{args.baseline}` (PSNR: {baseline['psnr_y']:.3f} dB)\n\n")
        f.write(md_table)
    print(f"Saved Markdown to {args.output_md}")

    # Print to console
    print("\n" + "="*80)
    print("CPU BENCHMARK SUMMARY (Super Resolution)")
    print("="*80)
    print(f"Baseline: {args.baseline} → PSNR = {baseline['psnr_y']:.3f} dB")
    print(df_out.to_string())


if __name__ == "__main__":
    main()