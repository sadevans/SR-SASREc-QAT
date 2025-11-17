#!/usr/bin/env python3
"""
Aggregate all results/*.json (excluding *_cpu_eval.json) into a single CSV/table.
Supports Super Resolution (PSNR only).
"""

import glob
import json
import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pattern", default="results/*.json")
    parser.add_argument("--exclude", default="*_cpu_eval.json")
    parser.add_argument("--output", default="results/sr_metrics_summary.csv")
    args = parser.parse_args()

    all_files = glob.glob(args.input_pattern)
    exclude_pattern = args.exclude.replace("*", "").replace("_", "")
    files = [f for f in all_files if "_cpu_eval" not in f]

    records = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            record = {
                "run_name": data.get("run_name", os.path.basename(f).replace(".json", "")),
                "psnr": data.get("psnr", None),
                "config_hash": hash(str(data.get("config", {})))  # optional
            }
            records.append(record)

    df = pd.DataFrame(records).set_index("run_name")
    df.to_csv(args.output)
    print(f"✅ Exported {len(df)} results to {args.output}")
    print(df.round(3))

if __name__ == "__main__":
    main()