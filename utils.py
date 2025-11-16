"""Utility helpers for configuration management, logging, and training."""

from __future__ import annotations

import json
import logging
import random
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

LOGGER = logging.getLogger("quant_experiments")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a basic logging setup shared by training and evaluation."""
    if logging.getLogger().handlers:
        # Respect existing handlers (e.g. when running inside notebooks).
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def deep_update(target: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``target`` and return the result."""
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            target[key] = deep_update(dict(target[key]), value)
        else:
            target[key] = value
    return target


def _resolve_config_path(base_path: Path, candidate: Optional[str]) -> Optional[Path]:
    if not candidate:
        return None
    candidate_path = Path(candidate)
    if not candidate_path.is_absolute():
        candidate_path = (base_path.parent / candidate_path).resolve()
    return candidate_path


def load_yaml_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return data


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file, supporting ``base_config`` and ``model_config`` keys."""

    def _load(path: Path, stack: Optional[List[Path]] = None) -> Dict[str, Any]:
        stack = stack or []
        if path in stack:
            raise RuntimeError(f"Circular config dependency detected: {stack + [path]}")
        data = load_yaml_file(path)
        merged: Dict[str, Any] = {}

        base_cfg = _resolve_config_path(path, data.get("base_config"))
        print(base_cfg)
        if base_cfg:
            merged = deep_update(merged, _load(base_cfg, stack + [path]))

        model_cfg = _resolve_config_path(path, data.get("model_config"))
        if model_cfg:
            merged = deep_update(merged, _load(model_cfg, stack + [path]))

        current = {k: v for k, v in data.items() if k not in {"base_config", "model_config"}}
        merged = deep_update(merged, current)

        merged.setdefault("_metadata", {})
        merged["_metadata"]["loaded_from"] = str(path)
        return merged

    config = _load(Path(config_path).resolve())
    return config


def ensure_dir(path: str | Path, create: bool = True) -> Path:
    """Ensure a directory exists and return it as ``Path``."""
    path_obj = Path(path)
    if create:
        path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    """Serialize ``data`` to JSON with UTF-8 encoding."""
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def set_random_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return the number of total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move a batch of tensors to ``device``."""
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif isinstance(value, Mapping):
            moved[key] = move_batch_to_device(value, device)
        else:
            moved[key] = value
    return moved


def gather_quantizable_layers(
    model: torch.nn.Module,
    quantize_embedding: bool = False,
) -> List[Tuple[str, torch.nn.Module]]:
    """Return a list of (qualified_name, module) pairs for modules to quantize."""
    from torch import nn

    target_types = [nn.Linear]
    if quantize_embedding:
        target_types.append(nn.Embedding)

    matches: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, tuple(target_types)):
            matches.append((name, module))
    return matches


def replace_module(root: torch.nn.Module, name: str, new_module: torch.nn.Module) -> None:
    """Replace a child module referenced by ``name`` with ``new_module``."""
    components = name.split(".")
    parent = root
    for comp in components[:-1]:
        parent = getattr(parent, comp)
    setattr(parent, components[-1], new_module)


def init_clearml_task(logging_cfg: Mapping[str, Any], full_config: Mapping[str, Any]):
    """Initialize a ClearML task if requested."""
    backend = logging_cfg.get("backend")
    if backend != "clearml":
        return None

    try:
        from clearml import Task
    except ImportError:
        warnings.warn(
            "ClearML logging requested but the 'clearml' package is not installed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    project = logging_cfg.get("clearml", {}).get("project_name", "lstm-int8-quant")
    task_name = logging_cfg.get("clearml", {}).get("task_name")
    tags = logging_cfg.get("clearml", {}).get("tags", [])

    task = Task.init(project_name=project, task_name=task_name, tags=tags)
    # ClearML versions differ in Task.connect signature. Prefer modern form.
    try:
        task.connect(full_config, name="config")
    except TypeError:
        # Older/newer variants may accept only (object) or different kwargs.
        try:
            task.connect(full_config)
        except Exception:
            pass
    return task


@dataclass
class RunningAverage:
    """Track a scalar metric over time."""

    value: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n
        self.value = self.sum / max(self.count, 1)


def detach_tensors(batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Detach tensors to avoid holding computation graphs when logging."""
    detached: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            detached[key] = value.detach().cpu()
        elif isinstance(value, Mapping):
            detached[key] = detach_tensors(value)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            detached[key] = [
                item.detach().cpu() if torch.is_tensor(item) else item for item in value
            ]
        else:
            detached[key] = value
    return detached