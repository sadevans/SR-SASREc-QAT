import os
import json
import argparse
import torch
from torch import optim
from utils import load_config, set_random_seeds
from experiments.sr.models import QuantizedESPCN
from experiments.sr.data import get_train_loader, get_val_datasets
from experiments.sr.evaluate import validate

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print('config: ', args.config)
    config = load_config(args.config)
    set_random_seeds(config["experiment"].get("seed", 42), config["experiment"].get("deterministic", True))
    # logger = get_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantizedESPCN(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        channels=config.model.channels,
        upscale_factor=config.model.upscale_factor,
        quant_config=config.quantization
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
    criterion = torch.nn.L1Loss()  # или MSELoss

    train_loader = get_train_loader(config)
    val_datasets = get_val_datasets(config)

    best_psnr = 0
    run_dir = f"runs/{config.experiment.name}"
    os.makedirs(run_dir, exist_ok=True)

    for epoch in range(config.training.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config.training.grad_clip)
        psnr = validate(model, val_datasets[0], device)  # validate on Set5
        print(f"Epoch {epoch+1}/{config.training.epochs} | Loss: {loss:.4f} | PSNR: {psnr:.2f}")

        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), f"{run_dir}/best.ckpt")

    # Save final metrics
    final_psnr = validate(model, val_datasets[0], device)
    results = {"psnr": final_psnr, "config": dict(config)}
    with open(f"results/{config.experiment.name}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()