"""AdaRound post-training quantization strategy."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..utils import gather_quantizable_layers, move_batch_to_device, replace_module
from .base import QuantStrategy, UniformAffineQuantizer


class AdaRoundLinear(nn.Module):
    """Linear layer wrapper implementing AdaRound soft rounding."""

    def __init__(self, original: nn.Linear, quantizer: UniformAffineQuantizer) -> None:
        super().__init__()
        self.quantizer = quantizer
        self.weight_fp = nn.Parameter(original.weight.detach().clone(), requires_grad=False)
        bias = original.bias.detach().clone() if original.bias is not None else None
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.alpha = nn.Parameter(torch.zeros_like(self.weight_fp))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zero_point", torch.tensor(0.0))
        self.qmin, self.qmax = self.quantizer.qmin, self.quantizer.qmax

    def set_quant_params(self) -> None:
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
        scale = self.scale.to(self.weight_fp.device)
        zero_point = self.zero_point.to(self.weight_fp.device)
        value = self.weight_fp / scale + zero_point
        if self.training:
            rounded = self.soft_round(value)
        else:
            rounded = self.hard_round(value)
        dequant = (rounded - zero_point) * scale
        return dequant

    def round_regularizer(self) -> Tensor:
        sig = torch.sigmoid(self.alpha)
        return torch.mean((sig - 0.5) ** 2)

    def forward(self, input: Tensor) -> Tensor:
        weight_q = self.effective_weight()
        return F.linear(input, weight_q, self.bias)


class AdaRoundQuantStrategy(QuantStrategy):
    """Post-training quantization using AdaRound optimisation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.bits = config.get("bits", 8)
        self.symmetric = config.get("symmetric", True)
        self.per_channel = config.get("per_channel", True)
        self.rounding_iters = config.get("rounding_iters", 1000)
        self.rounding_reg = config.get("rounding_reg", 1e-4)
        self.base_checkpoint = config.get("base_checkpoint")
        self.reference_model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None

    def _wrap_module(self, name: str, module: nn.Module) -> Optional[nn.Module]:  # type: ignore[override]
        # Not used: AdaRound overrides ``attach`` with a custom flow.
        return None

    def attach(self, model: nn.Module) -> nn.Module:
        self.reference_model = copy.deepcopy(model).eval()
        modules = gather_quantizable_layers(model, quantize_embedding=self.quantize_embedding)
        for name, module in modules:
            if not isinstance(module, nn.Linear):
                continue
            quantizer = UniformAffineQuantizer(
                bits=self.bits,
                symmetric=self.symmetric,
                per_channel=self.per_channel,
                channel_axis=0,
            )
            wrapped = AdaRoundLinear(module, quantizer)
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
        self.reference_model.to(device)
        self.reference_model.eval()
        adaround_modules = [module for _, module in self.handles if isinstance(module, AdaRoundLinear)]
        if not adaround_modules:
            return

        for module in adaround_modules:
            module.train()

        optimizer = torch.optim.Adam([module.alpha for module in adaround_modules], lr=1e-2)
        criterion = nn.MSELoss()
        iterator = iter(loader)

        for iteration in range(self.rounding_iters):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            batch = move_batch_to_device(batch, device)
            inputs = batch["input_ids"]
            lengths = batch.get("lengths")

            optimizer.zero_grad()
            with torch.no_grad():
                target = self.reference_model(inputs, lengths)
            output = self.model(inputs, lengths)
            loss = criterion(output, target)
            reg = sum(module.round_regularizer() for module in adaround_modules)
            loss = loss + self.rounding_reg * reg
            loss.backward()
            optimizer.step()

        for module in adaround_modules:
            module.eval()

    def step(self) -> None:
        # No per-step updates required for AdaRound once calibrated.
        return
