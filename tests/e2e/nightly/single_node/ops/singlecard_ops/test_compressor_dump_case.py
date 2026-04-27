import os
from typing import Any

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op


enable_custom_op()


REQUIRED_TENSORS = (
    "hidden_states",
    "wkv_weight",
    "wgate_weight",
    "kv_state",
    "score_state",
    "ape",
    "norm_weight",
    "compress_sin",
    "compress_cos",
    "state_block_table",
    "cu_seqlens",
    "start_pos",
)


def load_compressor_case() -> dict[str, Any]:
    """Load one dumped compressor case.

    Expected keys:
      required tensors in REQUIRED_TENSORS,
      optional attrs: rope_head_dim, cmp_ratio, coff, norm_eps, rotary_mode,
      optional golden tensor: output_compressed_kv.

    Replace this function with per-tensor torch.load calls if your dump is
    stored as separate files.
    """
    case_path = os.getenv("COMPRESSOR_CASE_PATH")
    if not case_path:
        pytest.skip("Set COMPRESSOR_CASE_PATH to a dumped compressor case .pt")

    case = torch.load(case_path, map_location="cpu")
    if not isinstance(case, dict):
        raise TypeError(f"Expected dict case from {case_path}, got {type(case)}")
    return case


def to_npu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.npu()
    if isinstance(value, dict):
        return {key: to_npu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_npu(item) for item in value)
    if isinstance(value, list):
        return [to_npu(item) for item in value]
    return value


def scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return value


def require_case_tensors(case: dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_TENSORS if name not in case]
    if missing:
        raise KeyError(f"Missing compressor case tensors: {missing}")


@torch.inference_mode()
def test_compressor_dump_single_case():
    case = load_compressor_case()
    require_case_tensors(case)

    torch_npu.npu.set_device(int(os.getenv("DEVICE_ID", "0")))
    case = to_npu(case)

    compress_sin = case["compress_sin"]
    compress_cos = case["compress_cos"]

    compressed_kv, out1, out2, out3, out4 = torch.ops._C_ascend.compressor(
        case["hidden_states"],
        case["wkv_weight"],
        case["wgate_weight"],
        case["kv_state"],
        case["score_state"],
        case["ape"],
        case["norm_weight"],
        compress_sin.view(-1, compress_sin.shape[-1]),
        compress_cos.view(-1, compress_cos.shape[-1]),
        kv_block_table=case["state_block_table"],
        score_block_table=case.get("score_block_table",
                                   case["state_block_table"]),
        cu_seqlens=case["cu_seqlens"],
        seqused=case.get("seqused"),
        start_pos=case["start_pos"],
        rope_head_dim=scalar(case.get("rope_head_dim", 64)),
        cmp_ratio=scalar(case.get("cmp_ratio", 4)),
        coff=scalar(case.get("coff", 1)),
        norm_eps=scalar(case.get("norm_eps", 1e-6)),
        rotary_mode=scalar(case.get("rotary_mode", 2)),
        enable_grad=scalar(case.get("enable_grad", False)),
    )

    assert compressed_kv.numel() > 0
    assert out1.numel() == 0
    assert out2.numel() == 0
    assert out3.numel() == 0
    assert out4.numel() == 0

    expected = case.get("output_compressed_kv")
    if expected is not None:
        torch.testing.assert_close(
            compressed_kv.cpu(),
            expected.cpu(),
            rtol=case.get("rtol", 1e-2),
            atol=case.get("atol", 1e-2),
        )
