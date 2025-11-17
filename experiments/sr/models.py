import math
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from quant import QUANTIZER_MAP


def _normalize_quant_config(raw_config: Optional[dict]) -> Dict[str, Any]:
    """
    Merge user-provided quantization config with defaults and method-specific overrides.

    The YAML files follow the pattern:

    quantization:
      method: lsq
      weight_bits: 4
      act_bits: 4
      lsq:
        alpha_init: 1.0

    Strategies expect canonical names like ``bits``/``activation_bits`` and flat dictionaries.
    """

    defaults: Dict[str, Any] = {
        "method": "none",
        "weight_bits": 8,
        "act_bits": 8,
    }
    config = deepcopy(raw_config) if raw_config is not None else {}
    merged: Dict[str, Any] = {**defaults, **config}

    method = str(merged.get("method", "none")).lower()
    merged["method"] = method

    method_overrides = config.get(method) if isinstance(config, dict) else None
    if isinstance(method_overrides, dict):
        merged.update(method_overrides)

    # Canonical key names expected by strategy classes.
    if "weight_bits" in merged and "bits" not in merged:
        merged["bits"] = merged["weight_bits"]
    if "act_bits" in merged and "activation_bits" not in merged:
        merged["activation_bits"] = merged["act_bits"]

    # Common synonyms from the YAML configs.
    synonym_map = {
        "clip_init": "alpha_init",
        "drop_prob": "qdrop_p",
        "num_iterations": "rounding_iters",
        "m": "apot_m",
        "k": "apot_k",
    }
    for source, target in synonym_map.items():
        if source in merged and target not in merged:
            merged[target] = merged[source]

    return merged


class QuantizedConv2d(nn.Module):
    """
    Обёртка над nn.Conv2d, которая применяет квантование весов (и опционально активаций)
    во время forward pass.
    """
    def __init__(self, conv: nn.Conv2d, weight_quant, act_quant=None):
        super().__init__()
        self.conv = conv
        self.weight_quant = weight_quant
        self.act_quant = act_quant

    def forward(self, x: Tensor) -> Tensor:
        weight_q = self.weight_quant(self.conv.weight)
        bias = self.conv.bias

        x = F.conv2d(
            x,
            weight_q,
            bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )

        if self.act_quant is not None:
            x = self.act_quant(x)
        return x


class QuantizedESPCN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 64,
        upscale_factor: int = 3,
        quant_config: dict = None,
    ) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.quant_config = _normalize_quant_config(quant_config)
        self.method = self.quant_config['method']

        # Инициализация квантователей
        if self.method != "none":
            strategy_cls = QUANTIZER_MAP.get(self.method)
            if strategy_cls is None:
                raise ValueError(f"Unknown quantization method '{self.method}'. "
                                f"Available: {sorted(QUANTIZER_MAP.keys())}")

            # AdaRound — PTQ, не требует weight/act quantizers в модели
            if self.method == "adaround":
                self.weight_quant = self.act_quant = lambda x: x
            else:
                strategy = strategy_cls(self.quant_config)
                dummy_conv = nn.Conv2d(1, 1, 1)
                self.weight_quant = strategy.create_weight_quantizer("conv", dummy_conv)
                self.act_quant = strategy.create_activation_quantizer("conv", dummy_conv)
        else:
            self.weight_quant = self.act_quant = lambda x: x

        hidden_channels = channels // 2
        final_out_channels = int(out_channels * (upscale_factor ** 2))

        # === Создаём обычные Conv2d ===
        conv1 = nn.Conv2d(in_channels, channels, kernel_size=5, padding=2)
        conv2 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        conv3 = nn.Conv2d(hidden_channels, final_out_channels, kernel_size=3, padding=1)

        # === Оборачиваем в QuantizedConv2d ===
        # Для conv1 и conv2 — квантуем и веса, и активации
        self.quant_conv1 = QuantizedConv2d(conv1, self.weight_quant, self.act_quant)
        self.quant_conv2 = QuantizedConv2d(conv2, self.weight_quant, self.act_quant)
        # Для conv3 — только веса (активации не квантуем!)
        self.quant_conv3 = QuantizedConv2d(conv3, self.weight_quant, act_quant=None)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # Инициализация весов как в оригинальной ESPCN
        self._init_weights()

    def _init_weights(self):
        for module in [self.quant_conv1.conv, self.quant_conv2.conv, self.quant_conv3.conv]:
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data, 0.0, 0.001)
                else:
                    fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                    std = math.sqrt(2 / fan_in)
                    nn.init.normal_(module.weight.data, 0.0, std)
                nn.init.zeros_(module.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        # Feature maps
        x = torch.tanh(self.quant_conv1(x))  # quant внутри QuantizedConv2d
        x = torch.tanh(self.quant_conv2(x))

        # Sub-pixel layer: активации не квантуются
        x = self.quant_conv3(x)  # веса квантуются, активации — нет
        x = self.pixel_shuffle(x)
        x = torch.clamp_(x, 0.0, 1.0)
        return x
