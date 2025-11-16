import time
import json
import argparse
import torch
from utils import load_config
from .models import QuantizedESPCN
from .data import SRBenchmarkDataset
from torch.utils.data import DataLoader


def rgb_to_y_tensor(rgb):
    return 0.257 * rgb[:, 0] + 0.504 * rgb[:, 1] + 0.098 * rgb[:, 2] + 0.0627


def calc_psnr(sr, hr, max_val=1.0):
    sr_y = rgb_to_y_tensor(sr)
    hr_y = rgb_to_y_tensor(hr)
    mse = (sr_y - hr_y).pow(2).mean()
    return 10 * torch.log10(max_val ** 2 / mse)


@torch.no_grad()
def validate(model, dataset, device):
    model.eval()
    psnr_total = 0
    loader = DataLoader(dataset, batch_size=1, num_workers=0)
    for lr, hr, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        psnr_total += calc_psnr(sr, hr).item()
    return psnr_total / len(dataset)


@torch.no_grad()
def benchmark_cpu(model, dataset, num_threads=4):
    torch.set_num_threads(num_threads)
    model.to("cpu").eval()
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    # Warmup
    for lr, _, _ in loader:
        _ = model(lr)
        break

    start = time.perf_counter()
    total_samples = 0
    psnr_sum = 0
    for lr, hr, _ in loader:
        sr = model(lr)
        psnr_sum += calc_psnr(sr, hr).item()
        total_samples += 1
    total_time = time.perf_counter() - start

    avg_latency_ms = total_time / total_samples * 1000
    throughput = total_samples / total_time

    return {
        "psnr_y": psnr_sum / total_samples,
        "avg_latency_ms": avg_latency_ms,
        "throughput_samples_per_sec": throughput,
        "num_samples": total_samples,
        "num_threads": num_threads,
        "device": "cpu"
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="Set5")
    args = parser.parse_args()

    config = load_config(args.config)
    model = QuantizedESPCN(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        channels=config.model.channels,
        upscale_factor=config.model.upscale_factor,
        quant_config=config.quantization
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    dataset = SRBenchmarkDataset(
        hr_dir=f"data/{args.split}",
        scale=config.model.upscale_factor,
        rgb_range=config.data.rgb_range
    )

    # CPU benchmark
    metrics = benchmark_cpu(model, dataset)
    metrics["run_name"] = config.experiment.name
    metrics["task"] = "super_resolution"

    out_path = f"results/{config.experiment.name}_cpu_eval.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved CPU metrics to {out_path}")


if __name__ == "__main__":
    main()