# SPDX-License-Identifier: Apache-2.0
"""Tensor dump helpers for xllm-compatible debug dumps."""

import contextvars
import os
from collections.abc import Mapping, Sequence

import torch
from torch import nn
import logging

try:
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

_DUMP_STEP_CONTEXT: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "xllm_dump_step", default=None)
_DUMP_LAYER_CONTEXT: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "xllm_dump_layer", default=None)
_DUMP_INVALID_LAYER_LOGGED = False


def _env_enabled(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() in ("1", "true", "on", "yes")


def dump_enabled() -> bool:
    return _env_enabled("XLLM_DUMP_TENSOR")


def dump_target_layer() -> int:
    global _DUMP_INVALID_LAYER_LOGGED
    value = os.environ.get("XLLM_DUMP_LAYER", "0")
    try:
        layer = int(value)
        if layer < 0:
            raise ValueError
        return layer
    except ValueError:
        if not _DUMP_INVALID_LAYER_LOGGED:
            logger.warning(
                "Invalid XLLM_DUMP_LAYER=%r; fallback to layer 0.", value)
            _DUMP_INVALID_LAYER_LOGGED = True
        return 0


def dump_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass

    rank = os.environ.get("RANK")
    if rank is not None:
        try:
            return int(rank)
        except ValueError:
            logger.debug("Invalid RANK=%r; fallback to tensor parallel rank.", rank)

    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


def sanitize_dump_component(component: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("_", "-", ".") else "_"
        for ch in component)


def save_tensor_as_pickle(tensor: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor.detach().cpu().contiguous(), path)


def tensor_info(tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return "None"
    if not torch.is_tensor(tensor):
        return f"type={type(tensor).__name__}"
    return (f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"device={tensor.device} numel={tensor.numel()}")


def set_dump_step(step: int):
    return _DUMP_STEP_CONTEXT.set(step)


def reset_dump_step(token) -> None:
    _DUMP_STEP_CONTEXT.reset(token)


def set_dump_layer(layer_idx: int):
    return _DUMP_LAYER_CONTEXT.set(layer_idx)


def reset_dump_layer(token) -> None:
    _DUMP_LAYER_CONTEXT.reset(token)


def current_dump_layer() -> int | None:
    return _DUMP_LAYER_CONTEXT.get()


def parse_layer_idx(layer_name: str | None) -> int | None:
    if not layer_name:
        return None
    parts = layer_name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def log_tensor_info(module: str,
                    name: str,
                    tensor: torch.Tensor | None,
                    *,
                    layer_idx: int | None = None) -> None:
    if not dump_enabled():
        return

    step = _DUMP_STEP_CONTEXT.get()
    if step is None:
        step = 0
    if step != 0:
        logger.debug("Skip tensor info for %s/%s: step=%s is not 0.",
                     module, name, step)
        return

    if layer_idx is None:
        layer_idx = _DUMP_LAYER_CONTEXT.get()
    logger.debug("Tensor info layer=%s %s/%s %s.", layer_idx, module, name,
                 tensor_info(tensor))


def log_mapping_info(module: str,
                     prefix: str,
                     value,
                     *,
                     layer_idx: int | None = None) -> None:
    if torch.is_tensor(value) or value is None:
        log_tensor_info(module, prefix, value, layer_idx=layer_idx)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            log_mapping_info(module, f"{prefix}_{key}", item,
                             layer_idx=layer_idx)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for idx, item in enumerate(value):
            log_mapping_info(module, f"{prefix}_{idx}", item,
                             layer_idx=layer_idx)
    else:
        logger.debug("Tensor info layer=%s %s/%s type=%s.",
                     layer_idx, module, prefix, type(value).__name__)


def dump_tensor(module: str,
                name: str,
                tensor: torch.Tensor | None,
                *,
                layer_idx: int | None = None) -> None:
    if not dump_enabled():
        return

    step = _DUMP_STEP_CONTEXT.get()
    if step is None:
        step = 0
    if step != 0:
        logger.debug("Skip tensor dump for %s/%s: step=%s is not 0.",
                     module, name, step)
        return

    if layer_idx is None:
        layer_idx = _DUMP_LAYER_CONTEXT.get()
    if layer_idx is None:
        logger.debug("Skip tensor dump for %s/%s: layer context is unset.",
                     module, name)
        return

    target_layer = dump_target_layer()
    if layer_idx != target_layer:
        logger.debug(
            "Skip tensor dump for %s/%s: layer=%s target_layer=%s.",
            module, name, layer_idx, target_layer)
        return

    dump_dir = os.environ.get("DUMP_DIR")
    if not dump_dir:
        logger.debug("Skip tensor dump for %s/%s: DUMP_DIR is not set.",
                     module, name)
        return

    if tensor is None:
        logger.debug("Skip tensor dump for %s/%s: tensor is None.", module, name)
        return
    if not torch.is_tensor(tensor):
        logger.debug("Skip tensor dump for %s/%s: object type is %s.",
                     module, name, type(tensor).__name__)
        return

    rank = dump_rank()
    safe_module = sanitize_dump_component(module)
    safe_name = sanitize_dump_component(name)
    path = os.path.join(dump_dir, f"step{step}", f"rank{rank}",
                        f"layer{layer_idx}", safe_module, f"{safe_name}.pt")
    try:
        save_tensor_as_pickle(tensor, path)
        logger.debug(
            "Dumped tensor %s/%s to %s %s.",
            module, name, path, tensor_info(tensor))
    except Exception:
        logger.exception("Failed to dump tensor %s/%s to %s.", module, name, path)


def dump_mapping(module: str,
                 prefix: str,
                 value,
                 *,
                 layer_idx: int | None = None) -> None:
    if torch.is_tensor(value) or value is None:
        dump_tensor(module, prefix, value, layer_idx=layer_idx)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            dump_mapping(module, f"{prefix}_{key}", item, layer_idx=layer_idx)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for idx, item in enumerate(value):
            dump_mapping(module, f"{prefix}_{idx}", item, layer_idx=layer_idx)
    else:
        logger.debug("Skip tensor dump for %s/%s: object type is %s.",
                     module, prefix, type(value).__name__)


def first_tensor(value) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def dump_module_io(module: nn.Module, dump_module: str) -> None:
    def pre_hook(_module, inputs):
        dump_tensor(dump_module, "input", first_tensor(inputs))

    def post_hook(_module, _inputs, output):
        dump_tensor(dump_module, "output", first_tensor(output))

    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)
