# SPDX-License-Identifier: Apache-2.0
"""Tensor dump helpers for xllm-compatible debug dumps."""

import contextvars
import os
import threading
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
_DUMP_INVALID_STEP_LOGGED = False
_MODULE_SEQUENCE_LOCK = threading.Lock()
_MODULE_SEQUENCE_STATE: dict[tuple[int, int, int], dict[str, object]] = {}


def _env_disabled(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.lower() in ("0", "false", "off", "no")


def dump_enabled() -> bool:
    return not _env_disabled("XLLM_DUMP_TENSOR")


def dump_target_layers_raw() -> str:
    return os.environ.get("XLLM_DUMP_LAYER", "0") or "0"


def dump_target_layers() -> tuple[int, ...]:
    global _DUMP_INVALID_LAYER_LOGGED
    value = dump_target_layers_raw()
    layers: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            layer = int(token)
            if layer < 0:
                raise ValueError
            layers.append(layer)
        except ValueError:
            if not _DUMP_INVALID_LAYER_LOGGED:
                logger.warning(
                    "Invalid XLLM_DUMP_LAYER=%r; fallback to layer 0.", value)
                _DUMP_INVALID_LAYER_LOGGED = True
            return (0,)
    return tuple(layers) if layers else (0,)


def dump_target_layer() -> int:
    return dump_target_layers()[0]


def dump_target_steps_raw() -> str:
    return os.environ.get("XLLM_DUMP_STEP", "0") or "0"


def dump_target_steps() -> tuple[int, ...]:
    global _DUMP_INVALID_STEP_LOGGED
    value = dump_target_steps_raw()
    steps: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            step = int(token)
            if step < 0:
                raise ValueError
            steps.append(step)
        except ValueError:
            if not _DUMP_INVALID_STEP_LOGGED:
                logger.warning(
                    "Invalid XLLM_DUMP_STEP=%r; fallback to step 0.", value)
                _DUMP_INVALID_STEP_LOGGED = True
            return (0,)
    return tuple(steps) if steps else (0,)


def is_dump_target_step(step: int) -> bool:
    return step >= 0 and step in dump_target_steps()


def is_dummy_run() -> bool:
    try:
        from vllm.forward_context import get_forward_context
        forward_context = get_forward_context()
        return bool(getattr(forward_context, "is_dummy_run", False))
    except Exception:
        return False


def is_process_request() -> bool:
    try:
        from vllm.forward_context import get_forward_context
        forward_context = get_forward_context()
        return bool(getattr(forward_context, "is_process_request", False))
    except Exception:
        return False


def is_dump_target_layer(layer_idx: int) -> bool:
    return layer_idx in dump_target_layers()


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


def module_with_sequence(module: str, *, layer_idx: int | None = None) -> str:
    if not dump_enabled():
        return module

    step = _DUMP_STEP_CONTEXT.get()
    if step is None:
        step = 0

    if layer_idx is None:
        layer_idx = _DUMP_LAYER_CONTEXT.get()
    if layer_idx is None:
        return module

    rank = dump_rank()
    state_key = (step, rank, layer_idx)
    with _MODULE_SEQUENCE_LOCK:
        state = _MODULE_SEQUENCE_STATE.setdefault(
            state_key, {
                "next_seq": 1,
                "last_raw_module": None,
                "last_prefixed_module": None,
            })
        if state["last_raw_module"] == module and state["last_prefixed_module"]:
            return str(state["last_prefixed_module"])

        seq = int(state["next_seq"])
        state["next_seq"] = seq + 1
        prefixed_module = f"{seq:02d}_{module}"
        state["last_raw_module"] = module
        state["last_prefixed_module"] = prefixed_module
        return prefixed_module


def reset_module_sequence(step: int | None = None) -> None:
    rank = dump_rank()
    with _MODULE_SEQUENCE_LOCK:
        if step is None:
            keys = [key for key in _MODULE_SEQUENCE_STATE if key[1] == rank]
        else:
            keys = [key for key in _MODULE_SEQUENCE_STATE
                    if key[0] == step and key[1] == rank]
        for key in keys:
            _MODULE_SEQUENCE_STATE.pop(key, None)


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


def _tensor_print_value(tensor: torch.Tensor) -> str:
    try:
        return str(tensor.detach().cpu())
    except Exception as exc:
        return f"<failed to print tensor: {exc}>"


def _emit_dump_log(level: int, message: str, *args) -> None:
    text = message % args if args else message
    logger.log(level, text)
    print(text, flush=True)


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
    if is_dummy_run() or not is_process_request():
        return

    step = _DUMP_STEP_CONTEXT.get()
    if step is None:
        step = 0
    if not is_dump_target_step(step):
        logger.debug("Skip tensor info for %s/%s: step=%s target_steps=[%s].",
                     module, name, step, dump_target_steps_raw())
        return

    if layer_idx is None:
        layer_idx = _DUMP_LAYER_CONTEXT.get()
    _emit_dump_log(
        logging.INFO, "[TENSOR_DUMP] tensor layer=%s %s/%s %s value=%s",
        layer_idx, module, name, tensor_info(tensor),
        _tensor_print_value(tensor) if torch.is_tensor(tensor) else tensor)


def log_mapping_info(module: str,
                     prefix: str,
                     value,
                     *,
                     layer_idx: int | None = None) -> None:
    if not dump_enabled():
        return

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
    elif is_process_request():
        _emit_dump_log(logging.INFO,
                       "[TENSOR_DUMP] tensor layer=%s %s/%s type=%s",
                       layer_idx, module, prefix, type(value).__name__)


def dump_tensor(module: str,
                name: str,
                tensor: torch.Tensor | None,
                *,
                layer_idx: int | None = None) -> None:
    if not dump_enabled():
        return
    if is_dummy_run() or not is_process_request():
        return

    step = _DUMP_STEP_CONTEXT.get()
    if step is None:
        step = 0
    if not is_dump_target_step(step):
        _emit_dump_log(
            logging.INFO,
            "[TENSOR_DUMP] skip %s/%s: step=%s target_steps=[%s]",
            module, name, step, dump_target_steps_raw())
        return

    if layer_idx is None:
        layer_idx = _DUMP_LAYER_CONTEXT.get()
    if layer_idx is None:
        _emit_dump_log(logging.INFO,
                       "[TENSOR_DUMP] skip %s/%s: layer context is unset",
                       module, name)
        return

    if not is_dump_target_layer(layer_idx):
        _emit_dump_log(
            logging.INFO,
            "[TENSOR_DUMP] skip %s/%s: layer=%s target_layers=[%s] %s",
            module, name, layer_idx, dump_target_layers_raw(),
            tensor_info(tensor))
        return

    dump_dir = os.environ.get("DUMP_DIR")
    if not dump_dir:
        _emit_dump_log(logging.WARNING,
                       "[TENSOR_DUMP] skip %s/%s: DUMP_DIR is not set",
                       module, name)
        return

    if tensor is None:
        _emit_dump_log(logging.INFO,
                       "[TENSOR_DUMP] skip %s/%s: tensor is None",
                       module, name)
        return
    if not torch.is_tensor(tensor):
        _emit_dump_log(logging.INFO,
                       "[TENSOR_DUMP] skip %s/%s: object type is %s",
                       module, name, type(tensor).__name__)
        return

    rank = dump_rank()
    prefixed_module = module_with_sequence(module, layer_idx=layer_idx)
    safe_module = sanitize_dump_component(prefixed_module)
    safe_name = sanitize_dump_component(name)
    path = os.path.join(dump_dir, f"step{step}", f"rank{rank}",
                        f"layer{layer_idx}", safe_module, f"{safe_name}.pt")
    try:
        saved = tensor.detach().cpu().contiguous()
        save_tensor_as_pickle(saved, path)
        _emit_dump_log(
            logging.INFO,
            "[TENSOR_DUMP] saved %s/%s to %s layer=%s rank=%s step=%s %s value=%s",
            prefixed_module, name, path, layer_idx, rank, step,
            tensor_info(tensor), _tensor_print_value(saved))
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
        if dump_enabled():
            _emit_dump_log(logging.INFO,
                           "[TENSOR_DUMP] skip %s/%s: object type is %s",
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
