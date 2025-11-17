from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utils import gather_quantizable_layers, move_batch_to_device, replace_module
from .base import QuantStrategy, UniformAffineQuantizer


class AdaRoundConv2d(nn.Module):
    """Conv2d layer wrapper implementing AdaRound soft rounding for PTQ."""

    def __init__(self, original: nn.Conv2d, quantizer: UniformAffineQuantizer) -> None:
        super().__init__()
        self.quantizer = quantizer
        self.weight_fp = nn.Parameter(original.weight.detach().clone(), requires_grad=False)
        bias = original.bias.detach().clone() if original.bias is not None else None
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Learnable alpha for soft rounding
        self.alpha = nn.Parameter(torch.zeros_like(self.weight_fp))

        # Buffers for quantization parameters
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.qmin, self.qmax = self.quantizer.qmin, self.quantizer.qmax

        # Inherit conv properties
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups

    def set_quant_params(self) -> None:
        """Initialize scale and zero_point from original weight."""
        self.quantizer.initialize_from_tensor(self.weight_fp)
        self.quantizer.initialized.fill_(True)
        scale, zero_point = self.quantizer._calc_qparams()
        self.scale = scale.detach()
        self.zero_point = zero_point.detach()

    def hard_round(self, value: Tensor) -> Tensor:
        return torch.clamp(torch.round(value), self.qmin, self.qmax)

    def soft_round(self, value: Tensor) -> Tensor:
        return torch.clamp(
            torch.floor(value) + torch.sigmoid(self.alpha),
            self.qmin,
            self.qmax,
        )

    def effective_weight(self) -> Tensor:
        device = self.weight_fp.device
        scale = self.scale.to(device)
        zero_point = self.zero_point.to(device)

        if scale.numel() == 1:
            # Per-tensor
            value = self.weight_fp / scale + zero_point
        else:
            # Per-channel: [C_out, 1, 1, 1]
            s = scale.view(-1, 1, 1, 1)
            zp = zero_point.view(-1, 1, 1, 1)
            value = self.weight_fp / s + zp

        if self.training:
            rounded = self.soft_round(value)
        else:
            rounded = self.hard_round(value)

        if scale.numel() == 1:
            dequant = (rounded - zero_point) * scale
        else:
            dequant = (rounded - zero_point.view(-1, 1, 1, 1)) * scale.view(-1, 1, 1, 1)
        return dequant

    def round_regularizer(self) -> Tensor:
        """L2 regularizer pushing alpha toward 0 or inf (i.e., hard rounding)."""
        sig = torch.sigmoid(self.alpha)
        return torch.mean((sig - 0.5) ** 2)

    def forward(self, input: Tensor) -> Tensor:
        weight_q = self.effective_weight()
        return F.conv2d(
            input,
            weight_q,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class AdaRoundQuantStrategy(QuantStrategy):
    """Post-training quantization using AdaRound optimisation for Conv2d layers."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.bits = config.get("bits", 8)
        self.symmetric = config.get("symmetric", True)
        self.per_channel = config.get("per_channel", True)
        self.rounding_iters = config.get("rounding_iters", 1000)
        self.rounding_reg = config.get("rounding_reg", 1e-4)
        self.lr = config.get("rounding_lr", 1e-2)
        self.reference_model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None
        self.handles: List[Tuple[str, nn.Module]] = []

    def _wrap_module(self, name: str, module: nn.Module) -> Optional[nn.Module]:
        # Not used — override via attach
        return None

    def attach(self, model: nn.Module) -> nn.Module:
        """Replace all quantizable Conv2d layers with AdaRoundConv2d."""
        self.reference_model = copy.deepcopy(model).eval()
        modules = gather_quantizable_layers(model, quantize_embedding=False)

        for name, module in modules:
            if not isinstance(module, nn.Conv2d):
                continue

            quantizer = UniformAffineQuantizer(
                bits=self.bits,
                symmetric=self.symmetric,
                per_channel=self.per_channel,
                channel_axis=0,  # quantize along output channels
            )
            wrapped = AdaRoundConv2d(module, quantizer)
            wrapped.set_quant_params()
            replace_module(model, name, wrapped)
            self.handles.append((name, wrapped))

        self.model = model
        return model

    def calibrate(self, loader) -> None:
        if self.model is None or self.reference_model is None:
            return

        device = next(self.model.parameters()).device
        self.device = device
        self.reference_model.to(device).eval()
        self.model.train()  # AdaRound layers must be in train mode

        adaround_modules = [m for _, m in self.handles if isinstance(m, AdaRoundConv2d)]
        if not adaround_modules:
            return

        optimizer = torch.optim.Adam([m.alpha for m in adaround_modules], lr=self.lr)
        criterion = nn.MSELoss()
        iterator = iter(loader)

        for iteration in range(self.rounding_iters):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            batch = move_batch_to_device(batch, device)
            # Assume your SR dataloader provides 'lr' (low-res input)
            if "lr" not in batch:
                raise KeyError("AdaRound calibration expects batch key 'lr' for super-resolution models.")
            lr = batch["lr"]

            optimizer.zero_grad()

            with torch.no_grad():
                target = self.reference_model(lr)

            output = self.model(lr)
            loss = criterion(output, target)
            reg = sum(m.round_regularizer() for m in adaround_modules)
            total_loss = loss + self.rounding_reg * reg

            total_loss.backward()
            optimizer.step()

        # Switch to eval mode: use hard rounding
        self.model.eval()

    def step(self) -> None:
        # No step needed in AdaRound (PTQ, not QAT)
        pass