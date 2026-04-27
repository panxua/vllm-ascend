# DeepSeekV4 Sub-Module Dump Spec

## Why

原有的 dump 逻辑仅在 `step/rank/layer` 顶层保存少量 tensor（hidden_states、gate_router_logits、final_output 等），无法满足精细化调试需求。需要按子模块和小算子粒度进行逐层 dump，使每个子模块/算子的每个输入输出都保存为独立的 `.pt` 文件，便于定位数值问题和算子精度异常。

## What Changes

- 重构 dump 基础设施，引入 `_moe_dump_sub(tensor, data_name, operator_path)` 支持 `step{X}/rank{Y}/layer{Z}/<operator_path>/<data_name>.pt` 的目录结构
- **BREAKING**: 路径结构改为 `算子名/数据名.pt`，例如 `attention_q_rope/input_cos.pt` 表示 attention_q_rope 算子的 input_cos 数据
- 在 `DSAttention`（`_forward_prefill`、`_forward_decode`）中添加 20+ 个算子的 dump
- 在 `DeepseekV4Gate`（`DeepseekV4MoE.forward`）中添加 6+ 个算子的 dump
- 在 `Compressor` kernel 调用处添加 compressor 算子，包含所有输入输出
- 在 `DeepseekV4Indexer`（`indexer_select_qli`）中添加 15+ 个算子的 dump
- 在 `FusedMoE`（`fused_moe.py`、`experts_selector.py`、`moe_mlp.py`、`w8a8_dynamic.py`）中添加 22+ 个算子的 dump
- 新增 mHC (hc_pre/hc_post) 算子的 dump

## Impact

- 现有 dump 环境变量 `MOE_DUMP_DIR`、`MOE_DUMP_STEPS`、`MOE_DUMP_LAYERS` 行为保持不变
- dump 路径从 `step{X}/rank{Y}/layer{Z}/<name>.pt` 变为 `step{X}/rank{Y}/layer{Z}/<operator_path>/<data_name>.pt`

## ADDED Requirements

### Requirement: DSAttention 子模块逐算子 dump

系统应当在 `DSAttention` 的 prefill 和 decode 路径中，按照 `算子名/数据名.pt` 结构保存：

| 算子路径 | 数据名 | 描述 |
|---|---|---|
| `attention_ds_attention` | `input_hidden_states` | attention 模块入口 hidden_states |
| `attention_q_a_proj` | `output_qr` | wq_a 投影输出 qr |
| `attention_q_a_proj` | `input_hidden_states` | wq_a 投影输入 |
| `attention_q_a_layernorm` | `output` | decode 动态量化路径的 RMSNorm 输出 |
| `attention_q_a_layernorm` | `input` | RMSNorm 输入 |
| `attention_q_b_proj` | `output_q` | wq_b 投影输出 q（reshape 前） |
| `attention_q_b_proj` | `input_qr` | wq_b 投影输入 qr |
| `attention_q_rmsnorm` | `output_q` | q Triton RMSNorm 输出 |
| `attention_q_rmsnorm` | `input_q` | RMSNorm 输入 |
| `attention_kv_proj` | `output_kv` | wkv 投影输出 kv |
| `attention_kv_proj` | `input_hidden_states` | wkv 投影输入 |
| `attention_kv_layernorm` | `output_kv` | kv RMSNorm 输出 |
| `attention_kv_layernorm` | `input_kv` | kv RMSNorm 输入 |
| `attention_q_rope` | `input_q` | q rope 输入 |
| `attention_q_rope` | `input_cos` | q rope cos 输入 |
| `attention_q_rope` | `input_sin` | q rope sin 输入 |
| `attention_q_rope` | `output_q` | q rope 输出 |
| `attention_kv_rope` | `input_kv` | kv rope 输入 |
| `attention_kv_rope` | `input_cos` | kv rope cos 输入 |
| `attention_kv_rope` | `input_sin` | kv rope sin 输入 |
| `attention_kv_rope` | `output_kv` | kv rope 输出 |
| `attention_scatter_ori_kv` | `input_kv` | ori_kv scatter 输入 |
| `attention_scatter_ori_kv` | `output` | ori_kv scatter 输出（写入后） |
| `attention_sparse_attn` | `input_q` | sparse attention q 输入 |
| `attention_sparse_attn` | `input_kv` | sparse attention kv 输入 |
| `attention_sparse_attn` | `output` | sparse attention 输出 |
| `attention_output_rope` | `input` | output rope 反旋转输入 |
| `attention_output_rope` | `output` | output rope 输出 |
| `attention_o_a_proj` | `input` | wo_a 投影输入 |
| `attention_o_a_proj` | `output` | wo_a 投影输出 |
| `attention_o_b_proj` | `input` | wo_b 投影输入 |
| `attention_o_b_proj` | `output` | wo_b 最终投影输出 |

#### Scenario: prefill 路径 dump
- **WHEN** `has_prefill=True` 且 `layer_idx` 在 `MOE_DUMP_LAYERS` 中且 `step` 在 `MOE_DUMP_STEPS` 中
- **THEN** 在 `step{X}/rank{Y}/layer{Z}/<operator_path>/` 下保存对应 `<data_name>.pt` 文件

#### Scenario: decode 路径 dump
- **WHEN** `has_decode=True` 且 `layer_idx` 在 `MOE_DUMP_LAYERS` 中且 `step` 在 `MOE_DUMP_STEPS` 中
- **THEN** 在 `step{X}/rank{Y}/layer{Z}/<operator_path>/` 下保存对应 `<data_name>.pt` 文件

### Requirement: Compressor 子模块逐算子 dump

系统应当在 `AscendDSAImpl._forward_prefill` 和 `_forward_decode` 的 compressor 调用处：

| 算子路径 | 数据名 | 描述 |
|---|---|---|
| `compressor` | `input_hidden_states` | hidden_states 输入 |
| `compressor` | `input_cos` | compress cos 输入 |
| `compressor` | `input_sin` | compress sin 输入 |
| `compressor` | `input_block_table` | block_table 输入 |
| `compressor` | `input_cu_seqlens` | cu_seqlens 输入 |
| `compressor` | `input_start_pos` | start_pos 输入 |
| `compressor` | `weight_wkv` | wkv 权重 |
| `compressor` | `weight_wgate` | wgate 权重 |
| `compressor` | `weight_ape` | ape 权重 |
| `compressor` | `weight_norm` | norm 权重 |
| `compressor` | `state_kv` | kv_state 输入 |
| `compressor` | `state_score` | score_state 输入 |
| `compressor` | `output` | compressor kernel 输出 output_compressed_kv |

#### Scenario: Compressor dump
- **WHEN** `compress_ratio > 1` 且满足 dump 条件
- **THEN** 在 `step{X}/rank{Y}/layer{Z}/compressor/` 下保存对应 `<data_name>.pt` 文件

### Requirement: Compressor Scatter 子模块 dump

| 算子路径 | 数据名 | 描述 |
|---|---|---|
| `scatter_cmp_kv` | `input` | scatter 前 compressed_kv |
| `scatter_cmp_kv` | `output` | scatter 后 compressed_kv（写入 kv_cache 后） |

### Requirement: DeepseekV4Indexer 子模块逐算子 dump

系统应当在 `indexer_select_qli` 方法中：

| 算子路径 | 数据名 | 描述 |
|---|---|---|
| `indexer_build_query_wq_b` | `input_qr` | wq_b 输入 |
| `indexer_build_query_wq_b` | `output` | wq_b 输出 |
| `indexer_build_weights_weights_proj` | `input` | weights 投影输入 |
| `indexer_build_weights_weights_proj` | `output` | weights 投影输出 |
| `indexer_compress_kv` | `input_hidden_states` | indexer compressor 输入 |
| `indexer_compress_kv` | `output` | indexer compressor 输出 kv |
| `indexer_rope_q` | `input_q` | indexer q rope 输入 |
| `indexer_rope_q` | `input_cos` | indexer q rope cos 输入 |
| `indexer_rope_q` | `input_sin` | indexer q rope sin 输入 |
| `indexer_rope_q` | `output_q` | indexer q rope 输出 |
| `indexer_hadamard_q` | `input` | q hadamard 变换输入 |
| `indexer_hadamard_q` | `output` | q hadamard 变换输出 |
| `indexer_hadamard_kv` | `input` | kv hadamard 变换输入 |
| `indexer_hadamard_kv` | `output` | kv hadamard 变换输出 |
| `indexer_dynamic_quant_q` | `input` | q 动态量化输入 |
| `indexer_dynamic_quant_q` | `output` | q 动态量化输出 |
| `indexer_dynamic_quant_kv` | `input` | kv 动态量化输入 |
| `indexer_dynamic_quant_kv` | `output` | kv 动态量化输出 |
| `indexer_cache_update` | `input_kv` | cache 更新输入 kv |
| `indexer_cache_update` | `input_kv_scale` | cache 更新输入 kv_scale |
| `indexer_cache_update` | `output` | cache 更新输出 |
| `indexer_lightning` | `input_q` | lightning indexer q 输入 |
| `indexer_lightning` | `input_kv` | lightning indexer kv 输入 |
| `indexer_lightning` | `input_weights` | lightning indexer weights 输入 |
| `indexer_lightning` | `output` | lightning indexer 输出 topk_idxs |

#### Scenario: Indexer dump
- **WHEN** `compress_ratio == 4` 且满足 dump 条件
- **THEN** 在 `step{X}/rank{Y}/layer{Z}/<operator_path>/` 下保存对应 `<data_name>.pt` 文件

### Requirement: DeepseekV4Gate 子模块逐算子 dump

系统应当在 `DeepseekV4MoE.forward` 方法中：

| 算子路径 | 数据名 | 描述 |
|---|---|---|
| `gate_matmul` | `input_hidden_states` | matmul 输入 hidden_states |
| `gate_matmul` | `output_router_logits` | matmul 输出 router_logits |
| `gate_matmul` | `weight` | gate_weight |
| `gate_matmul` | `bias_e_score_correction` | e_score_correction_bias |
| `gate_gating_top_k_hash` | `input_router_logits` | hash gating 输入 |
| `gate_gating_top_k_hash` | `output_topk_weights` | hash gating 输出 topk_weights |
| `gate_gating_top_k_hash` | `output_topk_ids` | hash gating 输出 topk_ids |
| `gate_select_experts_native` | `input_router_logits` | native select 输入 |
| `gate_select_experts_native` | `output_weights` | native select 输出（scoring_func 后） |
| `gate_renormalize_topk_weights` | `input_weights` | renormalize 输入 topk_weights |
| `gate_renormalize_topk_weights` | `output_weights` | renormalize 输出 |
| `gate_renormalize_topk_weights_sqrtsoftplus` | `input_weights` | sqrtsoftplus renormalize 输入 |
| `gate_renormalize_topk_weights_sqrtsoftplus` | `output_weights` | sqrtsoftplus renormalize 输出 |
| `gate_final` | `input_routed` | routed_experts 输入 |
| `gate_final` | `input_shared` | shared_experts 输入 |
| `gate_final` | `output_routed` | routed_experts 输出 |
| `gate_final` | `output_shared` | shared_experts 输出 |

### Requirement: FusedMoE 子模块逐算子 dump

系统应当在 MoE 相关的 `fused_moe.py`、`experts_selector.py`、`moe_mlp.py`、`w8a8_dynamic.py` 中：

| 算子路径 | 数据名 | 描述 |
|---|---|---|
| `ffn_forward` | `input_hidden_states` | forward 入口 hidden_states |
| `ffn_forward` | `input_router_logits` | forward 入口 router_logits |
| `ffn_forward_with_selected_experts` | `output` | 已选专家的 forward 输出 |
| `ffn_dp_gather` | `input` | dp_gather 前输入 |
| `ffn_dp_gather` | `output` | dp_gather 输出 |
| `ffn_dp_slice` | `input` | dp_slice 前输入 |
| `ffn_dp_slice` | `output` | dp_slice 输出 |
| `ffn_shared_experts` | `input_hidden_states` | shared_experts 输入 |
| `ffn_shared_experts` | `output_gate_up` | shared_experts gate_up 输出 |
| `ffn_shared_expert_gate` | `input` | shared_expert_gate 输入 |
| `ffn_shared_expert_gate` | `output` | shared_expert_gate 输出 before allreduce |
| `ffn_gate_proj` | `input` | gate_proj 输入 |
| `ffn_gate_proj` | `weight_w13` | w13_weight |
| `ffn_gate_proj` | `weight_w2` | w2_weight |
| `ffn_forward_expert` | `input_hidden_states` | forward_expert hidden_states |
| `ffn_forward_expert` | `input_topk_weights` | forward_expert topk_weights |
| `ffn_forward_expert` | `input_topk_ids` | forward_expert topk_ids |
| `ffn_select_experts` | `input_hidden_states` | select_experts hidden_states 输入 |
| `ffn_select_experts` | `input_router_logits` | select_experts router_logits 输入 |
| `ffn_select_experts` | `output_topk_weights` | select_experts 输出 topk_weights |
| `ffn_select_experts` | `output_topk_ids` | select_experts 输出 topk_ids |
| `ffn_moe_active_topk` | `input` | moe_active_topk 输入 |
| `ffn_moe_active_topk` | `output_topk_weights` | moe_active_topk 输出 topk_weights |
| `ffn_moe_active_topk` | `output_topk_ids` | moe_active_topk 输出 topk_ids |
| `ffn_moe_init_routing_v2` | `input` | moe_init_routing_v2 输入 |
| `ffn_moe_init_routing_v2` | `output` | moe_init_routing_v2 输出 |
| `ffn_dynamic_quant` | `input_hidden_states` | 动态量化 hidden_states |
| `ffn_dynamic_quant` | `input_router_logits` | 动态量化 router_logits |
| `ffn_group_gemm1` | `input` | GEMM1 输入 |
| `ffn_group_gemm1` | `input_group_list` | GEMM1 group_list |
| `ffn_group_gemm1` | `output` | GEMM1 输出 |
| `ffn_dequant_swiglu_quant` | `input` | SwiGLU 输入 |
| `ffn_dequant_swiglu_quant` | `output` | SwiGLU 激活输出 |
| `ffn_dequant_swiglu_quant` | `output_scale` | SwiGLU scale |
| `ffn_group_gemm2` | `input` | GEMM2 输入 |
| `ffn_group_gemm2` | `output` | GEMM2 输出 |
| `ffn_activation` | `input` | activation 输入 |
| `ffn_activation` | `output` | activation 输出 |
| `ffn_moe_combine_result` | `input` | moe_combine_result 输入 |
| `ffn_moe_combine_result` | `output` | moe_combine_result 最终输出 |
| `ffn_shared_output_add` | `input_shared` | shared_output_add shared 输入 |
| `ffn_shared_output_add` | `input_routed` | shared_output_add routed 输入 |
| `ffn_shared_output_add` | `output` | shared_output_add 输出 |
| `ffn_tp_reduce` | `input` | tp_reduce 前输入 |
| `ffn_tp_reduce` | `output` | tp_reduce 后 routed_out |
| `ffn_ep_reduce` | `input` | ep_reduce 前输入 |
| `ffn_ep_reduce` | `output` | ep_reduce 后输出 |

### Requirement: mHC (hc_pre/hc_post) 子模块逐算子 dump

系统应当在 DeepSeekV4 的 mHC（multi-head Context）相关操作中：

| 算子路径 | 数据名 | 描述 |
|---|---|---|
| `mhc_hc_pre` | `input_hidden_states` | hc_pre 操作输入 hidden_states |
| `mhc_hc_pre` | `input_scale` | hc_pre scale 输入 |
| `mhc_hc_pre` | `output` | hc_pre 操作输出 |
| `mhc_hc_pre` | `weight` | hc_pre 操作权重 |
| `mhc_hc_pre` | `weight_rmsnorm` | hc_pre RMSNorm 权重 |
| `mhc_hc_pre` | `output_scale` | hc_pre scale 输出 |
| `mhc_hc_post` | `input_hidden_states` | hc_post 操作输入 hidden_states |
| `mhc_hc_post` | `input_scale` | hc_post scale 输入 |
| `mhc_hc_post` | `output` | hc_post 操作输出 |
| `mhc_hc_post` | `weight` | hc_post 操作权重 |
| `mhc_hc_post` | `weight_rmsnorm` | hc_post RMSNorm 权重 |
| `mhc_hc_post` | `output_scale` | hc_post scale 输出 |

## REMOVED Requirements

### Requirement: 旧 dump 辅助函数
**Reason**: `_shared_moe_dump`、`_selector_dump`、`_w8a8_dump_tensor`、`_mlp_dump` 功能重复且路径不一致
**Migration**: 统一使用 `_moe_dump_sub(tensor, data_name, operator_path)` 替代

### Requirement: gate_router_logits 等旧命名
**Reason**: 新路径结构按 `算子名/数据名.pt` 组织，旧命名如 `gate_router_logits` 改为 `gate_matmul/output_router_logits.pt`
**Migration**: 保持 `MOE_DUMP_LAYERS` 和 `MOE_DUMP_STEPS` 环境变量不变，路径自动迁移
