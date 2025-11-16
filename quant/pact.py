"""Parametric Clipping Activation quantization strategy."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from .base import LearnableStepSizeQuantizer, QATQuantStrategy, UniformAffineQuantizer


class PACTActivationQuantizer(UniformAffineQuantizer):
    """PACT activation quantizer with learnable clipping."""

    def __init__(self, bits: int, alpha_init: float, signed: bool) -> None:
        super().__init__(bits=bits, symmetric=signed, per_channel=False, channel_axis=-1)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.signed = signed

    def initialize_from_tensor(self, tensor: torch.Tensor) -> None:
        alpha = self.alpha.abs().detach().to(tensor.device)
        if self.signed:
            clipped = torch.clamp(tensor, min=-alpha, max=alpha)
        else:
            clipped = torch.relu(tensor)
            clipped = torch.minimum(clipped, alpha)
        super().initialize_from_tensor(clipped)

    def _forward_impl(self, tensor: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.abs().to(tensor.device)
        if self.signed:
            clipped = torch.clamp(tensor, min=-alpha, max=alpha)
        else:
            clipped = torch.relu(tensor)
            clipped = torch.minimum(clipped, alpha)
        return super()._forward_impl(clipped)


class PACTQuantStrategy(QATQuantStrategy):
    """QAT strategy combining LSQ-like weights with PACT activations."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.alpha_init = config.get("alpha_init", 6.0)
        self.activation_signed = config.get("activation_signed", True)

    def create_weight_quantizer(self, module_name: str, module: nn.Module) -> LearnableStepSizeQuantizer:
        return LearnableStepSizeQuantizer(
            bits=self.bits,
            per_channel=self.per_channel,
            symmetric=self.symmetric,
            channel_axis=0,
            alpha_init=self.alpha_init,
        )

    def create_activation_quantizer(self, module_name: str, module: nn.Module) -> PACTActivationQuantizer:
        return PACTActivationQuantizer(
            bits=self.activation_bits,
            alpha_init=self.alpha_init,
            signed=self.activation_signed,
        )
