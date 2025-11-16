"""Additive Powers-of-Two quantization strategy."""

from __future__ import annotations

import itertools
from typing import Any, Dict, List

import torch
from torch import Tensor, nn

from .base import QATQuantStrategy, UniformAffineQuantizer, FakeQuantizer


class APoTQuantizer(FakeQuantizer):
    """Quantizer approximating Additive Powers-of-Two (APoT) behaviour."""

    def __init__(
        self,
        bits: int,
        m: int,
        k: int,
        symmetric: bool = True,
    ) -> None:
        super().__init__(bits=bits, symmetric=symmetric, per_channel=False)
        self.m = m
        self.k = k
        self.register_buffer("scale", torch.tensor(1.0))
        self.levels = self._build_levels()

    def _build_levels(self) -> Tensor:
        base_levels: List[float] = [0.0]
        for combo in itertools.product(range(2**self.k), repeat=self.m):
            value = 0.0
            for i, idx in enumerate(combo):
                value += 2.0 ** (-(idx + i * self.k))
            base_levels.append(value)
        levels = torch.tensor(sorted(set(base_levels)))
        return levels

    def initialize_from_tensor(self, tensor: Tensor) -> None:
        max_val = tensor.detach().abs().max()
        if max_val <= 0:
            max_val = torch.tensor(1.0, device=tensor.device)
        max_level = self.levels.max()
        scale = max_val / max_level.clamp(min=1e-6)
        self.scale.resize_(1).fill_(float(scale))

    def _forward_impl(self, tensor: Tensor) -> Tensor:
        device = tensor.device
        levels = self.levels.to(device)
        scale = self.scale.to(device)
        normalized = tensor / scale
        abs_norm = normalized.abs().view(-1)
        indices = torch.bucketize(abs_norm, levels)
        indices = torch.clamp(indices, 0, len(levels) - 1)
        quantized = levels[indices].view_as(normalized)
        quantized = quantized * normalized.sign()
        quantized_ste = normalized + (quantized - normalized).detach()
        dequant = quantized_ste * scale
        return dequant


class APoTQuantStrategy(QATQuantStrategy):
    """Strategy leveraging APoT quantizers for weights."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.apot_m = config.get("apot_m", 2)
        self.apot_k = config.get("apot_k", 4)

    def create_weight_quantizer(self, module_name: str, module: nn.Module) -> FakeQuantizer:
        return APoTQuantizer(
            bits=self.bits,
            m=self.apot_m,
            k=self.apot_k,
            symmetric=self.symmetric,
        )

    def create_activation_quantizer(self, module_name: str, module: nn.Module) -> UniformAffineQuantizer:
        return UniformAffineQuantizer(
            bits=self.activation_bits,
            symmetric=False,
            per_channel=False,
            channel_axis=-1,
        )
