import os
import json
import argparse
from pathlib import Path
import torch
from torch import optim
from torch.utils.data import DataLoader
from utils import load_config, set_random_seeds
from experiments.sr.models import QuantizedESPCN
from experiments.sr.data import get_train_loader, get_val_datasets
from experiments.sr.evaluate import validate
from quant import QUANTIZER_MAP

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    total_loss = 0
    for batch in loader:
        if isinstance(batch, dict):
            lr = batch["lr"].to(device)
            hr = batch.get("hr", None)
            if hr is not None:
                hr = hr.to(device)
        else:
            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        sr = model(lr)
        if hr is not None:
            loss = criterion(sr, hr)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader) if total_loss > 0 else 0.0


def create_adaround_loader(loader):
    """Wrap DataLoader to return dict format for AdaRound calibration."""
    for lr, hr in loader:
        yield {"lr": lr, "hr": hr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fp32-checkpoint", type=str, default=None, 
                       help="Path to FP32 checkpoint (required for AdaRound)")
    args = parser.parse_args()
    config = load_config(args.config)
    set_random_seeds(config["experiment"].get("seed", 42), config["experiment"].get("deterministic", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant_method = config['quantization'].get('method', 'none')
    
    # Handle AdaRound (PTQ) - requires FP32 checkpoint
    if quant_method == "adaround":
        if args.fp32_checkpoint is None:
            # Try to find FP32 checkpoint automatically
            fp32_name = config.get("fp32_experiment_name", "espcn_fp32")
            fp32_ckpt = Path(f"runs/{fp32_name}/best.ckpt")
            if not fp32_ckpt.exists():
                raise FileNotFoundError(
                    f"AdaRound requires FP32 checkpoint. Provide --fp32-checkpoint or train FP32 first.\n"
                    f"Expected path: {fp32_ckpt}"
                )
            args.fp32_checkpoint = str(fp32_ckpt)
        
        print(f"🔧 AdaRound PTQ: Loading FP32 model from {args.fp32_checkpoint}")
        # Load FP32 model first (this will be the reference)
        fp32_model = QuantizedESPCN(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            channels=config['model']['channels'],
            upscale_factor=config['model']['upscale_factor'],
            quant_config={"method": "none"}
        ).to(device)
        fp32_model.load_state_dict(torch.load(args.fp32_checkpoint, map_location=device))
        fp32_model.eval()
        
        # Create quantized model - for AdaRound, model structure uses identity quantizers
        # We'll use strategy.attach() to replace the internal conv layers
        model = QuantizedESPCN(
            in_channels=config['model']['in_channels'],
            out_channels=config['model']['out_channels'],
            channels=config['model']['channels'],
            upscale_factor=config['model']['upscale_factor'],
            quant_config=config['quantization']
        ).to(device)
        
        # Copy weights from FP32 model
        model.quant_conv1.conv.weight.data.copy_(fp32_model.quant_conv1.conv.weight.data)
        if fp32_model.quant_conv1.conv.bias is not None:
            model.quant_conv1.conv.bias.data.copy_(fp32_model.quant_conv1.conv.bias.data)
        model.quant_conv2.conv.weight.data.copy_(fp32_model.quant_conv2.conv.weight.data)
        if fp32_model.quant_conv2.conv.bias is not None:
            model.quant_conv2.conv.bias.data.copy_(fp32_model.quant_conv2.conv.bias.data)
        model.quant_conv3.conv.weight.data.copy_(fp32_model.quant_conv3.conv.weight.data)
        if fp32_model.quant_conv3.conv.bias is not None:
            model.quant_conv3.conv.bias.data.copy_(fp32_model.quant_conv3.conv.bias.data)
        
        # Apply AdaRound strategy - manually replace conv layers with AdaRoundConv2d
        strategy_cls = QUANTIZER_MAP.get("adaround")
        if strategy_cls is None:
            raise ValueError("AdaRound strategy not found in QUANTIZER_MAP")
        
        strategy = strategy_cls(config['quantization'])
        strategy.reference_model = fp32_model
        
        # Manually replace conv layers (gather_quantizable_layers doesn't find Conv2d)
        from quant.adaround import AdaRoundConv2d
        from quant.base import UniformAffineQuantizer
        from utils import replace_module
        
        adaround_modules = []
        for name, conv_module in [
            ("quant_conv1.conv", model.quant_conv1.conv),
            ("quant_conv2.conv", model.quant_conv2.conv),
            ("quant_conv3.conv", model.quant_conv3.conv),
        ]:
            quantizer = UniformAffineQuantizer(
                bits=strategy.bits,
                symmetric=strategy.symmetric,
                per_channel=strategy.per_channel,
                channel_axis=0,
            )
            wrapped = AdaRoundConv2d(conv_module, quantizer)
            wrapped.set_quant_params()
            replace_module(model, name, wrapped)
            strategy.handles.append((name, wrapped))
            adaround_modules.append(wrapped)
        
        strategy.model = model
        
        # Calibration step
        print("🔧 Running AdaRound calibration...")
        train_loader = get_train_loader(config)
        adaround_loader = create_adaround_loader(train_loader)
        strategy.calibrate(adaround_loader)
        print("✅ AdaRound calibration complete")
        
        # Validate and save
        val_datasets = get_val_datasets(config)
        final_psnr = validate(model, val_datasets[0], device)
        print(f"Final PSNR: {final_psnr:.2f}")
        
        run_dir = f"runs/{config['experiment']['name']}"
        os.makedirs(run_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{run_dir}/best.ckpt")
        
        results = {"psnr": final_psnr, "config": dict(config)}
        os.makedirs("results", exist_ok=True)
        with open(f"results/{config['experiment']['name']}.json", "w") as f:
            json.dump(results, f, indent=2)
        return
    
    # Regular QAT training (LSQ, QDrop, PACT, APoT, etc.)
    model = QuantizedESPCN(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        channels=config['model']['channels'],
        upscale_factor=config['model']['upscale_factor'],
        quant_config=config['quantization']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion = torch.nn.L1Loss()

    train_loader = get_train_loader(config)
    val_datasets = get_val_datasets(config)

    best_psnr = 0
    run_dir = f"runs/{config['experiment']['name']}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"🚀 Training {config['experiment']['name']} with {quant_method} quantization...")
    for epoch in range(config['training']['epochs']):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, config['training']['grad_clip'])
        psnr = validate(model, val_datasets[0], device)  # validate on Set5
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Loss: {loss:.4f} | PSNR: {psnr:.2f}")

        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(model.state_dict(), f"{run_dir}/best.ckpt")

    # Save final metrics
    final_psnr = validate(model, val_datasets[0], device)
    results = {"psnr": final_psnr, "config": dict(config)}
    os.makedirs("results", exist_ok=True)
    with open(f"results/{config['experiment']['name']}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Training complete. Best PSNR: {best_psnr:.2f}, Final PSNR: {final_psnr:.2f}")


if __name__ == "__main__":
    main()
