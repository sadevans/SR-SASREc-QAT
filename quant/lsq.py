"""Learned Step Size Quantization strategy."""

from __future__ import annotations

from typing import Any, Dict

from torch import nn

from .base import LearnableStepSizeQuantizer, QATQuantStrategy


class LSQQuantStrategy(QATQuantStrategy):
    """Learned Step Size Quantization (LSQ) with learnable step sizes."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.alpha_init = config.get("alpha_init", 6.0)
        self.activation_symmetric = config.get("activation_symmetric", config.get("symmetric", False))
        self.activation_per_channel = config.get("activation_per_channel", False)

    def create_weight_quantizer(self, module_name: str, module: nn.Module) -> LearnableStepSizeQuantizer:
        return LearnableStepSizeQuantizer(
            bits=self.bits,
            per_channel=self.per_channel,
            symmetric=self.symmetric,
            channel_axis=0,
            alpha_init=self.alpha_init,
        )

    def create_activation_quantizer(self, module_name: str, module: nn.Module) -> LearnableStepSizeQuantizer:
        return LearnableStepSizeQuantizer(
            bits=self.activation_bits,
            per_channel=self.activation_per_channel,
            symmetric=self.activation_symmetric,
            channel_axis=-1,
            alpha_init=self.alpha_init,
        )
