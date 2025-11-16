"""Stochastic fake-quantization drop strategy (QDrop)."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from .base import FakeQuantizer, QATQuantStrategy, UniformAffineQuantizer


class QDropQuantizer(FakeQuantizer):
    """Wrapper that randomly bypasses fake quantization during training."""

    def __init__(self, inner: FakeQuantizer, drop_prob: float) -> None:
        super().__init__(
            bits=inner.bits,
            symmetric=inner.symmetric,
            per_channel=inner.per_channel,
            channel_axis=inner.channel_axis,
        )
        self.inner = inner
        self.drop_prob = drop_prob

    def initialize_from_tensor(self, tensor: torch.Tensor) -> None:
        self.inner.initialize_from_tensor(tensor)
        self.inner.initialized.fill_(True)

    def _forward_impl(self, tensor: torch.Tensor) -> torch.Tensor:
        self.inner.train(self.training)
        if self.training and torch.rand(1).item() < self.drop_prob:
            return tensor
        return self.inner(tensor)


class QDropQuantStrategy(QATQuantStrategy):
    """Apply QDrop stochastic bypass to activation quantizers."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.drop_prob = float(config.get("qdrop_p", 0.2))

    def create_weight_quantizer(self, module_name: str, module: nn.Module) -> FakeQuantizer:
        return UniformAffineQuantizer(
            bits=self.bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            channel_axis=0,
        )

    def create_activation_quantizer(self, module_name: str, module: nn.Module) -> FakeQuantizer:
        base = UniformAffineQuantizer(
            bits=self.activation_bits,
            symmetric=False,
            per_channel=False,
            channel_axis=-1,
        )
        return QDropQuantizer(base, drop_prob=self.drop_prob)
