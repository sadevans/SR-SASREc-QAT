"""Base abstractions and utilities for quantization strategies."""
from __future__ import annotations

import abc
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utils import gather_quantizable_layers, replace_module

class FakeQuantizer(nn.Module, metaclass=abc.ABCMeta):
    """Abstract fake-quantizer to share between strategies."""

    def __init__(
        self,
        bits: int,
        symmetric: bool = True,
        per_channel: bool = False,
        channel_axis: int = 0,
    ) -> None:
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.qmin, self.qmax = self._quant_bounds()
        self.register_buffer("initialized", torch.tensor(False), persistent=False)

    def _quant_bounds(self) -> Tuple[int, int]:
        if self.symmetric:
            qmax = 2 ** (self.bits - 1) - 1
            return -qmax - 1, qmax
        return 0, 2**self.bits - 1

    def initialize_from_tensor(self, tensor: Tensor) -> None:
        """Optional hook for one-off initialization on the first batch."""

    @abc.abstractmethod
    def _forward_impl(self, tensor: Tensor) -> Tensor:
        """Perform fake quantization."""

    def forward(self, tensor: Tensor) -> Tensor:
        if not bool(self.initialized):
            self.initialize_from_tensor(tensor.detach())
            self.initialized.fill_(True)
        return self._forward_impl(tensor)


class UniformAffineQuantizer(FakeQuantizer):
    """Uniform affine fake quantization with running min/max observers."""

    def __init__(
        self,
        bits: int,
        symmetric: bool,
        per_channel: bool,
        channel_axis: int = 0,
        momentum: float = 0.95,
    ) -> None:
        super().__init__(bits, symmetric=symmetric, per_channel=per_channel, channel_axis=channel_axis)
        shape = (1,)
        self.momentum = momentum
        self.register_buffer("running_min", torch.zeros(shape))
        self.register_buffer("running_max", torch.zeros(shape))

    def initialize_from_tensor(self, tensor: Tensor) -> None:
        dims = list(range(tensor.ndim))
        if self.per_channel:
            axis = self.channel_axis if self.channel_axis >= 0 else tensor.ndim + self.channel_axis
            dims.pop(axis)
        reduce_dims = tuple(dims)
        min_val = tensor.amin(dim=reduce_dims, keepdim=self.per_channel) if reduce_dims else tensor
        max_val = tensor.amax(dim=reduce_dims, keepdim=self.per_channel) if reduce_dims else tensor
        self.running_min.resize_as_(min_val).copy_(min_val.detach())
        self.running_max.resize_as_(max_val).copy_(max_val.detach())

    def update_ranges(self, tensor: Tensor) -> None:
        dims = list(range(tensor.ndim))
        if self.per_channel:
            axis = self.channel_axis if self.channel_axis >= 0 else tensor.ndim + self.channel_axis
            dims.pop(axis)
        reduce_dims = tuple(dims)
        current_min = tensor.amin(dim=reduce_dims, keepdim=self.per_channel) if reduce_dims else tensor
        current_max = tensor.amax(dim=reduce_dims, keepdim=self.per_channel) if reduce_dims else tensor
        self.running_min.mul_(self.momentum).add_(current_min * (1 - self.momentum))
        self.running_max.mul_(self.momentum).add_(current_max * (1 - self.momentum))

    def _calc_qparams(self) -> Tuple[Tensor, Tensor]:
        min_val = self.running_min
        max_val = self.running_max
        if self.symmetric:
            max_val = torch.max(max_val.abs(), min_val.abs())
            min_val = -max_val
        scale = (max_val - min_val) / float(self.qmax - self.qmin)
        scale = torch.clamp(scale, min=1e-8)
        zero_point = self.qmin - torch.round(min_val / scale)
        zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
        device = min_val.device
        return scale.to(device), zero_point.to(device)

    def _forward_impl(self, tensor: Tensor) -> Tensor:
        self.update_ranges(tensor.detach())
        scale, zero_point = self._calc_qparams()
        value = tensor / scale + zero_point
        q = torch.clamp(torch.round(value), self.qmin, self.qmax)
        value_q = value + (q - value).detach()
        dequant = (value_q - zero_point) * scale
        return dequant


class LearnableStepSizeQuantizer(FakeQuantizer):
    """Implementation of LSQ-style learnable step size quantization."""

    def __init__(
        self,
        bits: int,
        per_channel: bool,
        symmetric: bool = True,
        channel_axis: int = 0,
        alpha_init: float = 6.0,
    ) -> None:
        super().__init__(bits, symmetric=symmetric, per_channel=per_channel, channel_axis=channel_axis)
        self.alpha_init = alpha_init
        self.scale_param: Optional[nn.Parameter] = None
        self.register_buffer("grad_multiplier", torch.tensor(1.0), persistent=False)
        self._hook_registered = False

    def initialize_from_tensor(self, tensor: Tensor) -> None:
        axis = self.channel_axis if self.channel_axis >= 0 else tensor.ndim + self.channel_axis
        if self.per_channel:
            dims = tuple(i for i in range(tensor.ndim) if i != axis)
            scale = tensor.detach().abs().mean(dim=dims, keepdim=False)
        else:
            scale = tensor.detach().abs().mean()
        if self.alpha_init is not None:
            scale = scale * float(self.alpha_init)
        scale = scale / math.sqrt(self.qmax)
        scale = scale.to(tensor.device)
        if self.scale_param is None:
            if self.per_channel:
                initial = scale.reshape(-1).clone()
            else:
                initial = scale.reshape(1).clone()
            self.scale_param = nn.Parameter(initial)
            self.register_parameter("scale", self.scale_param)
        else:
            target = scale.reshape_as(self.scale_param)
            self.scale_param.data.copy_(target)

        grad_scale = self._compute_grad_multiplier(tensor)
        self.grad_multiplier = grad_scale
        if not self._hook_registered and self.scale_param is not None:
            def _hook(grad: Tensor) -> Tensor:
                multiplier = self.grad_multiplier.to(grad.device)
                return grad * multiplier

            self.scale_param.register_hook(_hook)
            self._hook_registered = True

    def _forward_impl(self, tensor: Tensor) -> Tensor:
        if self.scale_param is None:
            raise RuntimeError("LearnableStepSizeQuantizer not initialized.")
        scale = self.scale_param.abs()
        if self.per_channel:
            view_shape = [1] * tensor.ndim
            axis = self.channel_axis if self.channel_axis >= 0 else tensor.ndim + self.channel_axis
            view_shape[axis] = -1
            scale = scale.view(*view_shape)
        value = tensor / scale
        q = torch.clamp(torch.round(value), self.qmin, self.qmax)
        value_q = value + (q - value).detach()
        dequant = value_q * scale
        return dequant

    def _compute_grad_multiplier(self, tensor: Tensor) -> Tensor:
        tensor = tensor.detach()
        if self.per_channel:
            axis = self.channel_axis if self.channel_axis >= 0 else tensor.ndim + self.channel_axis
            permute_order = [axis] + [i for i in range(tensor.ndim) if i != axis]
            flattened = tensor.permute(permute_order).reshape(tensor.shape[axis], -1)
            numel = flattened.size(1)
            scale = 1.0 / math.sqrt(max(numel, 1) * self.qmax)
            grad_scale = tensor.new_full((tensor.shape[axis],), scale)
        else:
            numel = tensor.numel()
            scale = 1.0 / math.sqrt(max(numel, 1) * self.qmax)
            grad_scale = tensor.new_full((1,), scale)
        grad_scale = grad_scale.reshape(-1)
        if self.scale_param is not None:
            grad_scale = grad_scale.reshape_as(self.scale_param)
        return grad_scale


class QuantLinear(nn.Linear):
    """Linear layer wrapper that applies fake-quantizers to weights and activations."""

    def __init__(
        self,
        original: nn.Linear,
        weight_quantizer: FakeQuantizer,
        activation_quantizer: Optional[FakeQuantizer] = None,
    ) -> None:
        super().__init__(original.in_features, original.out_features, bias=original.bias is not None)
        device = original.weight.device
        self.to(device)
        self.weight.data.copy_(original.weight.data)
        if original.bias is not None and self.bias is not None:
            self.bias.data.copy_(original.bias.data)
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

    def forward(self, input: Tensor) -> Tensor:
        weight_q = self.weight_quantizer(self.weight)
        output = F.linear(input, weight_q, self.bias)
        if self.activation_quantizer is not None:
            output = self.activation_quantizer(output)
        return output


class QuantEmbedding(nn.Embedding):
    """Embedding wrapper that optionally quantizes embedding weights."""

    def __init__(self, original: nn.Embedding, weight_quantizer: FakeQuantizer) -> None:
        super().__init__(
            num_embeddings=original.num_embeddings,
            embedding_dim=original.embedding_dim,
            padding_idx=original.padding_idx,
        )
        self.to(original.weight.device)
        self.weight.data.copy_(original.weight.data)
        self.weight_quantizer = weight_quantizer

    def forward(self, input: Tensor) -> Tensor:
        weight_q = self.weight_quantizer(self.weight)
        return F.embedding(
            input,
            weight_q,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


class QuantStrategy(nn.Module, metaclass=abc.ABCMeta):
    """Base class shared by all quantization strategies."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.quantize_embedding = config.get("quantize_embedding", False)
        self.handles: List[Tuple[str, nn.Module]] = []
        self.model: Optional[nn.Module] = None

    def attach(self, model: nn.Module) -> nn.Module:
        self.model = model
        modules = gather_quantizable_layers(model, quantize_embedding=self.quantize_embedding)
        for name, module in modules:
            wrapped = self._wrap_module(name, module)
            if wrapped is None:
                continue
            replace_module(model, name, wrapped)
            self.handles.append((name, wrapped))
        return model

    @abc.abstractmethod
    def _wrap_module(self, name: str, module: nn.Module) -> Optional[nn.Module]:
        """Wrap a module with fake-quant operations."""

    def calibrate(self, loader) -> None:  # pragma: no cover - to be overridden when needed
        """Optional calibration step for certain strategies."""

    def step(self) -> None:
        """Optional hook to update internal state each training step."""

    def extra_state_dict(self) -> Dict[str, Any]:
        return {"config": self.config}

    def load_extra_state(self, state: Dict[str, Any]) -> None:
        self.config.update(state.get("config", {}))


class QATQuantStrategy(QuantStrategy):
    """Shared logic for QAT-style strategies."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.bits = config.get("bits", 8)
        self.activation_bits = config.get("activation_bits", self.bits)
        self.per_channel = config.get("per_channel", False)
        self.symmetric = config.get("symmetric", True)

    def create_weight_quantizer(self, module_name: str, module: nn.Module) -> FakeQuantizer:
        return UniformAffineQuantizer(
            bits=self.bits,
            symmetric=self.symmetric,
            per_channel=self.per_channel,
            channel_axis=0,
        )

    def create_activation_quantizer(self, module_name: str, module: nn.Module) -> FakeQuantizer:
        return UniformAffineQuantizer(
            bits=self.activation_bits,
            symmetric=False,
            per_channel=False,
            channel_axis=-1,
        )

    def _wrap_module(self, name: str, module: nn.Module) -> Optional[nn.Module]:
        if isinstance(module, nn.Linear):
            wq = self.create_weight_quantizer(name, module)
            aq = self.create_activation_quantizer(name, module)
            with torch.no_grad():
                wq.initialize_from_tensor(module.weight.data)
                wq.initialized.fill_(True)
            if aq is not None:
                aq.to(module.weight.device)
            device = module.weight.device
            wq.to(device)
            if aq is not None:
                aq.to(device)
            return QuantLinear(module, wq, aq)
        if isinstance(module, nn.Embedding):
            wq = self.create_weight_quantizer(name, module)
            with torch.no_grad():
                wq.initialize_from_tensor(module.weight.data)
                wq.initialized.fill_(True)
            wq.to(module.weight.device)
            return QuantEmbedding(module, wq)
        return None
