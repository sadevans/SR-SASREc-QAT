"""Evaluation script for trained ESPCN super-resolution models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from .data import SRBenchmarkDataset
from .models import QuantizedESPCN
from utils import configure_logging, ensure_dir, load_config


def rgb_to_y_tensor(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] tensor (B,3,H,W) to Y [0,1] (B,H,W)."""
    return 0.257 * rgb[:, 0] + 0.504 * rgb[:, 1] + 0.098 * rgb[:, 2] + 0.0627


def calc_psnr_y(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> float:
    if sr.shape[1] == 1:
        sr_y = sr.squeeze(1)
        hr_y = hr.squeeze(1)
    else:
        sr_y = rgb_to_y_tensor(sr)
        hr_y = rgb_to_y_tensor(hr)
    mse = (sr_y - hr_y).pow(2).mean()
    psnr = 10 * torch.log10((max_val ** 2) / (mse + 1e-12))
    return psnr.item()


@torch.no_grad()
def benchmark_cpu(
    model: torch.nn.Module,
    dataset: SRBenchmarkDataset,
    num_threads: int = 4,
) -> Dict[str, Any]:
    torch.set_num_threads(num_threads)
    model.to("cpu").eval()
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    # Warmup
    for lr, _, _ in loader:
        _ = model(lr)
        break

    latencies = []
    psnr_sum = 0.0
    total_samples = 0

    for lr, hr, _ in loader:
        start = time.perf_counter()
        sr = model(lr)
        latencies.append(time.perf_counter() - start)
        psnr_sum += calc_psnr_y(sr, hr)
        total_samples += 1

    latencies = torch.tensor(latencies)
    avg_latency = latencies.mean().item()
    median_latency = latencies.median().item()
    throughput = total_samples / latencies.sum().item()

    return {
        "psnr_y": psnr_sum / total_samples,
        "avg_latency_ms": avg_latency * 1000,
        # "median_latency_ms": median_latency * 1000,
        "throughput_samples_per_sec": throughput,
        # "num_samples": total_samples,
        "num_threads": num_threads,
        # "device": "cpu",
    }

@torch.no_grad()
def validate(model: torch.nn.Module, dataset: SRBenchmarkDataset, device: torch.device):
    """Compute average PSNR-Y on the dataset (no timing)."""
    model.eval()
    total_psnr = 0.0
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        total_psnr += calc_psnr_y(sr, hr)
    return total_psnr / len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained ESPCN model on CPU.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint (.ckpt).")
    parser.add_argument("--split", type=str, default="Set5", help="Name of SR benchmark (e.g., Set5, Set14).")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)

    device = torch.device("cpu")
    model = QuantizedESPCN(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        channels=config["model"]["channels"],
        upscale_factor=config["model"]["upscale_factor"],
        quant_config=config.get("quantization", None),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    dataset = SRBenchmarkDataset(
        hr_dir=Path("data") / args.split,
        scale=config["model"]["upscale_factor"],
        rgb_range=config["data"]["rgb_range"],
    )

    metrics = benchmark_cpu(model, dataset)

    # Attach metadata
    metrics["run_name"] = config["experiment"]["name"]
    metrics["task"] = "super_resolution"
    metrics["dataset"] = args.split

    # Save to results/
    results_dir = Path(config.get("paths", {}).get("results_dir", "results"))
    results_dir = ensure_dir(results_dir)
    out_path = results_dir / f"{config['experiment']['name']}_cpu_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
