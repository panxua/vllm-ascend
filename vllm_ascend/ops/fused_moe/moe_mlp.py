# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

from typing import Optional

import os
import numpy as np
import torch
import torch_npu
from torch.nn.functional import pad
from vllm.forward_context import get_forward_context
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import (AscendDeviceType, dispose_tensor,
                               enable_custom_op, get_ascend_device_type,
                               get_weight_prefetch_method)

_MLP_DUMP_DIR = os.environ.get("MOE_DUMP_DIR", "/tmp/moe_dump")
_MLP_DUMP_LAYER = int(os.environ.get("MOE_DUMP_LAYER_IDX", "0"))
_mlp_dump_counter = {}


def _mlp_dump(tensor, name):
    global _mlp_dump_counter
    key = name
    _mlp_dump_counter[key] = _mlp_dump_counter.get(key, 0) + 1
    step = _mlp_dump_counter[key]
    d = os.path.join(_MLP_DUMP_DIR, f"layer{_MLP_DUMP_LAYER}", f"step{step}")
    os.makedirs(d, exist_ok=True)
    if isinstance(tensor, torch.Tensor):
        try:
            np.save(os.path.join(d, f"{name}.npy"), tensor.detach().cpu().float().numpy())
        except Exception:
            try:
                np.save(os.path.join(d, f"{name}.npy"), tensor.detach().cpu().to(torch.float32).numpy())
            except Exception:
                np.save(os.path.join(d, f"{name}_raw.npy"), tensor.detach().cpu().numpy())
        with open(os.path.join(d, f"{name}_meta.txt"), "w") as f:
            f.write(f"name={name}\nshape={list(tensor.shape)}\ndtype={tensor.dtype}\n")


def _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
    return fusion and dynamic_eplb and enable_custom_op()


def cumsum_group_list(group_list: torch.Tensor,
                      src_list_type: int,
                      dst_list_type: int,
                      active_num: int = 0,
                      expert_num: int = 0) -> torch.Tensor:
    if src_list_type not in [0, 1, 2]:
        raise ValueError(
            f"group_list_type should be in [0, 1, 2], but received {src_list_type}"
        )

    if src_list_type == dst_list_type:
        return group_list
    if src_list_type == 1 and dst_list_type == 0:
        return group_list.cumsum(dim=0)
    if src_list_type == 0 and dst_list_type == 1:
        group_diff = torch.diff(group_list)
        new_group = torch.cat([group_list[0].unsqueeze(0), group_diff], dim=0)
        return new_group
    if src_list_type == 2 and dst_list_type == 0:
        experts = pad(group_list[:, 0], (1, 0))
        tokens = pad(group_list[:, 1].cumsum(dim=0), (1, 0))
        cumsum_group_list = torch.full(size=(expert_num, ),
                                       fill_value=active_num,
                                       dtype=group_list.dtype,
                                       device=group_list.device)

        for i, (start, end) in enumerate(zip(experts[:-1], experts[1:])):
            if end > start:
                cumsum_group_list[start:end] = tokens[i]

        return cumsum_group_list
    raise NotImplementedError(
        f"Conversion from src_list_type={src_list_type} to dst_list_type={dst_list_type} is not implemented yet. "
        "This feature is under development.")


def quant_apply_mlp(hidden_states: torch.Tensor,
                    w1: list[torch.Tensor],
                    w1_scale: list[torch.Tensor],
                    w2: list[torch.Tensor],
                    w2_scale: list[torch.Tensor],
                    group_list: torch.Tensor,
                    group_list_type: int = 1,
                    dynamic_scale: torch.Tensor = None,
                    w1_scale_bias: torch.Tensor = None,
                    w2_scale_bias: torch.Tensor = None,
                    w1_offset: Optional[torch.Tensor] = None,
                    w2_offset: Optional[torch.Tensor] = None,
                    fusion: bool = False,
                    dynamic_eplb: bool = False) -> torch.Tensor:
    if w1_offset is not None:
        unquantized_hidden_states = hidden_states
        quantized_hidden_states = None
    elif dynamic_scale is None:
        unquantized_hidden_states = hidden_states
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(
            hidden_states)
        # Dispose the original unquantized hidden states
        # to save npu memory because they're no longer used.
        dispose_tensor(unquantized_hidden_states)
        quantized_hidden_states = None
    else:
        unquantized_hidden_states = None
        pertoken_scale = dynamic_scale
        quantized_hidden_states = hidden_states

    bias1, bias2 = None, None
    _output_dtype = w2_scale[0].dtype

    _mlp_dump(hidden_states, "quant_mlp_input")
    _mlp_dump(group_list, "quant_mlp_group_list")

    weight_prefetch_method = get_weight_prefetch_method()
    if weight_prefetch_method:
        weight_prefetch_method.maybe_prefetch_moe_weight_postprocess(
            hidden_states)
    is_mc2 = get_forward_context().moe_comm_type == MoECommType.MC2
    if w1_scale_bias is None and w1_offset is None and is_mc2:
        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = (
                torch.ops._C_ascend.
                grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                    x=hidden_states,
                    weight=w1,
                    weight_scale=w1_scale,
                    x_scale=pertoken_scale,
                    group_list=cumsum_group_list(group_list, group_list_type,
                                                 0),
                ))
        elif fusion and not dynamic_eplb:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w1[0],
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=w1_scale[0],
                x_scale=pertoken_scale)
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            if w1_scale[0].dtype != torch.float32:
                w1_scale[0] = w1_scale[0].to(torch.float32)
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                split_item=3,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=torch.int32)[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            hidden_states, swiglu_out_scale = torch_npu.npu_dequant_swiglu_quant(
                x=hidden_states,
                weight_scale=w1_scale[0],
                activation_scale=pertoken_scale,
                bias=None,
                quant_scale=None,
                quant_offset=None,
                group_index=cumsum_group_list(group_list, group_list_type, 1),
                activate_left=True,
                quant_mode=1,
            )
        _mlp_dump(hidden_states, "gmm1_swiglu_output")
        _mlp_dump(swiglu_out_scale, "gmm1_swiglu_out_scale")
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            scale=w2_scale,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=w2_scale[0].dtype)[0]
        _mlp_dump(hidden_states, "gmm2_down_proj_output")
    elif w1_offset is not None:
        # gmm1: gate_up_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[unquantized_hidden_states],
            weight=[w1],
            antiquant_scale=[w1_scale],
            antiquant_offset=[w1_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]
        dispose_tensor(unquantized_hidden_states)
        _mlp_dump(hidden_states, "antiquant_gmm1_gate_up_output")
        # act_fn: swiglu
        hidden_states = torch_npu.npu_swiglu(hidden_states)
        _mlp_dump(hidden_states, "antiquant_swiglu_output")
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2],
            antiquant_scale=[w2_scale],
            antiquant_offset=[w2_offset],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]
        _mlp_dump(hidden_states, "antiquant_gmm2_down_proj_output")
    else:
        if w1_scale_bias is not None:
            if group_list_type == 0:
                group_list = torch.cat(
                    [group_list[:1],
                     torch.diff(group_list, dim=0)])
                group_list_type = 1
            bias1 = [w1_scale_bias] if not fusion else w1_scale_bias
            bias2 = [w2_scale_bias]
            # TODO w4a8 scene: dynamic acquisition of dtype in the future
            _output_dtype = torch.bfloat16

        if _custom_gmm_swiglu_enabled(fusion, dynamic_eplb):
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = (
                torch.ops._C_ascend.
                grouped_matmul_swiglu_quant_weight_nz_tensor_list(
                    x=hidden_states,
                    weight=w1,
                    weight_scale=w1_scale,
                    x_scale=pertoken_scale,
                    group_list=cumsum_group_list(group_list, group_list_type,
                                                 0),
                    bias=bias1,
                ))
        elif fusion and not dynamic_eplb:
            # gmm1: gate_up_proj & act_fn: swiglu
            hidden_states, swiglu_out_scale, _ = torch_npu.npu_grouped_matmul_swiglu_quant(
                x=hidden_states,
                weight=w1[0],
                bias=bias1,
                group_list=cumsum_group_list(group_list, group_list_type, 0),
                weight_scale=w1_scale[0],
                x_scale=pertoken_scale)
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
        else:
            w1_scale[0] = w1_scale[0].to(w2_scale[0].dtype)
            # gmm1: gate_up_proj
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=w1,
                scale=w1_scale,
                bias=bias1,
                per_token_scale=[pertoken_scale],
                split_item=2,
                group_list_type=group_list_type,
                group_type=0,
                group_list=group_list,
                output_dtype=_output_dtype)[0]
            if quantized_hidden_states is not None:
                dispose_tensor(quantized_hidden_states)
            # act_fn: swiglu
            if HAS_TRITON:
                from vllm_ascend.ops.triton.activation.swiglu_quant import \
                    swiglu_quant
                hidden_states, swiglu_out_scale = swiglu_quant(
                    hidden_states,
                    group_list=group_list,
                    group_list_type=group_list_type)
            else:
                hidden_states = torch_npu.npu_swiglu(hidden_states)
                hidden_states, swiglu_out_scale = torch_npu.npu_dynamic_quant(
                    hidden_states)
        _mlp_dump(hidden_states, "else_branch_gmm1_swiglu_output")
        # gmm2: down_proj
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            scale=w2_scale,
            bias=bias2,
            per_token_scale=[swiglu_out_scale],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=_output_dtype)[0]
        _mlp_dump(hidden_states, "else_branch_gmm2_down_proj_output")
    return hidden_states


def unquant_apply_mlp(hidden_states: torch.Tensor,
                      w1: torch.Tensor,
                      w2: torch.Tensor,
                      group_list: torch.Tensor,
                      group_list_type: int = 1,
                      topk_scales: Optional[torch.Tensor] = None,
                      need_trans: bool = True) -> torch.Tensor:

    if need_trans:
        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2)

    gate_up_out = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w1],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    if get_ascend_device_type() == AscendDeviceType._310P:
        gate_up_out = torch_npu.npu_swiglu(gate_up_out.to(torch.float32)).to(
            torch.float16)
    else:
        gate_up_out = torch_npu.npu_swiglu(gate_up_out)

    if topk_scales is not None:
        gate_up_out *= topk_scales

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[gate_up_out],
        weight=[w2],
        split_item=2,
        group_list_type=group_list_type,
        group_type=0,
        group_list=group_list,
    )[0]
    return hidden_states


def unified_apply_mlp(hidden_states: torch.Tensor,
                      w1: torch.Tensor | list[torch.Tensor],
                      w2: torch.Tensor | list[torch.Tensor],
                      group_list: torch.Tensor,
                      w1_scale: Optional[list[torch.Tensor]] = None,
                      w2_scale: Optional[list[torch.Tensor]] = None,
                      dynamic_scale: torch.Tensor = None,
                      group_list_type: int = 1,
                      w1_scale_bias: torch.Tensor = None,
                      w2_scale_bias: torch.Tensor = None,
                      w1_offset: Optional[torch.Tensor] = None,
                      w2_offset: Optional[torch.Tensor] = None,
                      topk_scales: Optional[torch.Tensor] = None,
                      with_quant: bool = False,
                      fusion: bool = False,
                      need_trans: bool = True,
                      dynamic_eplb: bool = False) -> torch.Tensor:
    if with_quant:
        assert w1_scale is not None and w2_scale is not None
        return quant_apply_mlp(hidden_states=hidden_states,
                               w1=w1,
                               w1_scale=w1_scale,
                               w2=w2,
                               w2_scale=w2_scale,
                               group_list=group_list,
                               dynamic_scale=dynamic_scale,
                               group_list_type=group_list_type,
                               w1_scale_bias=w1_scale_bias,
                               w2_scale_bias=w2_scale_bias,
                               w1_offset=w1_offset,
                               w2_offset=w2_offset,
                               fusion=fusion,
                               dynamic_eplb=dynamic_eplb)
    else:
        return unquant_apply_mlp(hidden_states=hidden_states,
                                 w1=w1,
                                 w2=w2,
                                 group_list=group_list,
                                 group_list_type=group_list_type,
                                 topk_scales=topk_scales,
                                 need_trans=need_trans)
