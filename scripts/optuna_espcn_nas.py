#!/usr/bin/env python3
"""
Optuna hyperparameter search for ESPCN: channels, upscale, loss, LR.
Saves best config to results/optuna/espcn_nas_best_config_<timestamp>.yaml
"""

import os
import time
import yaml
import argparse
import optuna
from sr_experiments.train import train_one_epoch, validate
from sr_experiments.data import get_train_loader, get_val_datasets
from sr_experiments.models import QuantizedESPCN
from utils import set_seed

def objective(trial, base_config):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Search space
    channels = trial.suggest_categorical("channels", [32, 64, 96])
    upscale = trial.suggest_categorical("upscale_factor", [2, 3, 4])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = 10  # short training

    model = QuantizedESPCN(
        in_channels=3,
        out_channels=3,
        channels=channels,
        upscale_factor=upscale,
        quant_config={"method": "none"}
    ).to(device)

    config = type('Config', (), {
        'training': type('T', (), {
            'lr': lr,
            'batch_size': 16,
            'grad_clip': 1.0,
            'epochs': epochs
        })(),
        'model': type('M', (), {
            'upscale_factor': upscale
        })()
    })

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()
    train_loader = get_train_loader(base_config)
    val_dataset = get_val_datasets(base_config)[0]  # Set5

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device, 1.0)
    psnr = validate(model, val_dataset, device)

    return -psnr  # Optuna minimizes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--base-config", default="configs/base_sr.yaml")
    args = parser.parse_args()

    import torch
    base_config = load_config(args.base_config)  # реализуйте через utils

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, base_config), n_trials=args.trials)

    best = study.best_params
    best_config = {
        "experiment": {"name": f"espcn_nas_best_{int(time.time())}"},
        "model": {
            "channels": best["channels"],
            "upscale_factor": best["upscale_factor"]
        },
        "training": {
            "lr": best["lr"],
            "epochs": 30  # full training later
        },
        "quantization": {"method": "none"}
    }

    out_dir = "results/optuna"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/espcn_nas_best_config_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(best_config, f)
    print(f"✅ Best config saved to {out_path}")

if __name__ == "__main__":
    main()