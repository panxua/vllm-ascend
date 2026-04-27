# Tasks

## 基础设施重构

- [ ] Task 1: 在 `deepseek_v4.py` 中新增 `_moe_dump_sub` 辅助函数
  - 新增 `_moe_dump_should_dump(layer_idx)` 判断是否需要 dump
  - 新增 `_moe_dump_base_dir(layer_idx)` 获取基础路径
  - 新增 `_moe_dump_sub(tensor, data_name, operator_path, layer_idx=None)` 支持 `算子名/数据名.pt` 结构
  - 保留原有 `_moe_dump_tensor` 用于顶层 dump

## DSAttention 子模块 dump

- [ ] Task 2: 在 `dsa_v1.py` AscendDSAImpl `_forward_prefill` 中添加 dump
  - `attention_ds_attention/input_hidden_states.pt` - attention 模块入口
  - `attention_q_a_proj/input_hidden_states.pt` 和 `attention_q_a_proj/output_qr.pt`
  - `attention_q_b_proj/input_qr.pt` 和 `attention_q_b_proj/output_q.pt`
  - `attention_q_rmsnorm/input_q.pt` 和 `attention_q_rmsnorm/output_q.pt`
  - `attention_kv_proj/input_hidden_states.pt` 和 `attention_kv_proj/output_kv.pt`
  - `attention_kv_layernorm/input_kv.pt` 和 `attention_kv_layernorm/output_kv.pt`
  - `attention_q_rope/input_q.pt`、`attention_q_rope/input_cos.pt`、`attention_q_rope/input_sin.pt`、`attention_q_rope/output_q.pt`
  - `attention_kv_rope/input_kv.pt`、`attention_kv_rope/input_cos.pt`、`attention_kv_rope/input_sin.pt`、`attention_kv_rope/output_kv.pt`
  - `attention_scatter_ori_kv/input_kv.pt` 和 `attention_scatter_ori_kv/output.pt`
  - `attention_sparse_attn/input_q.pt`、`attention_sparse_attn/input_kv.pt`、`attention_sparse_attn/output.pt`
  - `attention_output_rope/input.pt` 和 `attention_output_rope/output.pt`
  - `attention_o_a_proj/input.pt` 和 `attention_o_a_proj/output.pt`
  - `attention_o_b_proj/input.pt` 和 `attention_o_b_proj/output.pt`

- [ ] Task 3: 在 `dsa_v1.py` AscendDSAImpl `_forward_decode` 中添加 dump
  - 同 Task 2 的 prefill 逻辑，添加相同算子路径
  - 额外添加 `attention_q_a_layernorm/input.pt` 和 `attention_q_a_layernorm/output.pt`（decode 动态量化路径）

## Compressor 子模块 dump

- [ ] Task 4: 在 `dsa_v1.py` AscendDSAImpl `_forward_prefill` 和 `_forward_decode` 中添加 Compressor dump
  - `compressor/input_hidden_states.pt`、`compressor/input_cos.pt`、`compressor/input_sin.pt`
  - `compressor/input_block_table.pt`、`compressor/input_cu_seqlens.pt`、`compressor/input_start_pos.pt`
  - `compressor/weight_wkv.pt`、`compressor/weight_wgate.pt`、`compressor/weight_ape.pt`、`compressor/weight_norm.pt`
  - `compressor/state_kv.pt`、`compressor/state_score.pt`、`compressor/output.pt`

## Compressor Scatter 子模块 dump

- [ ] Task 5: 添加 scatter 子模块 dump
  - `scatter_cmp_kv/input.pt` 和 `scatter_cmp_kv/output.pt`

## DeepseekV4Gate 子模块 dump

- [ ] Task 6: 在 `deepseek_v4.py` DeepseekV4MoE.forward 中添加 dump
  - `gate_matmul/input_hidden_states.pt`、`gate_matmul/output_router_logits.pt`、`gate_matmul/weight.pt`、`gate_matmul/bias_e_score_correction.pt`
  - `gate_final/input_routed.pt`、`gate_final/input_shared.pt`、`gate_final/output_routed.pt`、`gate_final/output_shared.pt`

## DeepseekV4Indexer 子模块 dump

- [ ] Task 7: 在 `dsa_v1.py` indexer_select_qli 中添加 dump
  - `indexer_build_query_wq_b/input_qr.pt` 和 `indexer_build_query_wq_b/output.pt`
  - `indexer_build_weights_weights_proj/input.pt` 和 `indexer_build_weights_weights_proj/output.pt`
  - `indexer_compress_kv/input_hidden_states.pt` 和 `indexer_compress_kv/output.pt`
  - `indexer_rope_q/input_q.pt`、`indexer_rope_q/input_cos.pt`、`indexer_rope_q/input_sin.pt`、`indexer_rope_q/output_q.pt`
  - `indexer_hadamard_q/input.pt` 和 `indexer_hadamard_q/output.pt`
  - `indexer_hadamard_kv/input.pt` 和 `indexer_hadamard_kv/output.pt`
  - `indexer_dynamic_quant_q/input.pt` 和 `indexer_dynamic_quant_q/output.pt`
  - `indexer_dynamic_quant_kv/input.pt` 和 `indexer_dynamic_quant_kv/output.pt`
  - `indexer_cache_update/input_kv.pt`、`indexer_cache_update/input_kv_scale.pt`、`indexer_cache_update/output.pt`
  - `indexer_lightning/input_q.pt`、`indexer_lightning/input_kv.pt`、`indexer_lightning/input_weights.pt`、`indexer_lightning/output.pt`

## FusedMoE 子模块 dump

- [ ] Task 8: 在 `fused_moe.py` AscendSharedFusedMoE.forward_impl 中添加 dump
  - `ffn_forward/input_hidden_states.pt`、`ffn_forward/input_router_logits.pt`
  - `ffn_dp_gather/input.pt` 和 `ffn_dp_gather/output.pt`
  - `ffn_tp_reduce/input.pt` 和 `ffn_tp_reduce/output.pt`
  - `ffn_shared_experts/input_hidden_states.pt` 和 `ffn_shared_experts/output_gate_up.pt`
  - `ffn_shared_expert_gate/input.pt` 和 `ffn_shared_expert_gate/output.pt`

- [ ] Task 9: 在 `fused_moe.py` AscendUnquantizedFusedMoEMethod.apply 中添加 dump
  - `ffn_select_experts/input_hidden_states.pt`、`ffn_select_experts/input_router_logits.pt`、`ffn_select_experts/output_topk_weights.pt`、`ffn_select_experts/output_topk_ids.pt`
  - `ffn_moe_active_topk/input.pt`、`ffn_moe_active_topk/output_topk_weights.pt`、`ffn_moe_active_topk/output_topk_ids.pt`
  - `ffn_forward_expert/input_hidden_states.pt`、`ffn_forward_expert/input_topk_weights.pt`、`ffn_forward_expert/input_topk_ids.pt`
  - `ffn_gate_proj/input.pt`、`ffn_gate_proj/weight_w13.pt`、`ffn_gate_proj/weight_w2.pt`
  - `ffn_moe_combine_result/input.pt` 和 `ffn_moe_combine_result/output.pt`

- [ ] Task 10: 在 `experts_selector.py` select_experts 中添加 dump
  - `gate_gating_top_k_hash/input_router_logits.pt`、`gate_gating_top_k_hash/output_topk_weights.pt`、`gate_gating_top_k_hash/output_topk_ids.pt`
  - `gate_select_experts_native/input_router_logits.pt` 和 `gate_select_experts_native/output_weights.pt`
  - `gate_renormalize_topk_weights/input_weights.pt` 和 `gate_renormalize_topk_weights/output_weights.pt`
  - `gate_renormalize_topk_weights_sqrtsoftplus/input_weights.pt` 和 `gate_renormalize_topk_weights_sqrtsoftplus/output_weights.pt`

- [ ] Task 11: 在 `moe_mlp.py` apply_mlp 中添加 dump
  - `ffn_group_gemm1/input.pt`、`ffn_group_gemm1/input_group_list.pt`、`ffn_group_gemm1/output.pt`
  - `ffn_dequant_swiglu_quant/input.pt`、`ffn_dequant_swiglu_quant/output.pt`、`ffn_dequant_swiglu_quant/output_scale.pt`
  - `ffn_group_gemm2/input.pt` 和 `ffn_group_gemm2/output.pt`
  - `ffn_activation/input.pt` 和 `ffn_activation/output.pt`

- [ ] Task 12: 在 `w8a8_dynamic.py` AscendW8A8DynamicFusedMoEMethod.apply 中添加 dump
  - `ffn_dynamic_quant/input_hidden_states.pt` 和 `ffn_dynamic_quant/input_router_logits.pt`
  - `ffn_moe_active_topk/input.pt`、`ffn_moe_active_topk/output_topk_weights.pt`、`ffn_moe_active_topk/output_topk_ids.pt`
  - `ffn_gate_proj/weight_w13.pt` 和 `ffn_gate_proj/weight_w2.pt`
  - `ffn_moe_combine_result/output.pt`

## mHC 子模块 dump

- [ ] Task 13: 在 DeepSeekV4 mHC 相关操作中添加 dump
  - `mhc_hc_pre/input_hidden_states.pt`、`mhc_hc_pre/input_scale.pt`、`mhc_hc_pre/output.pt`、`mhc_hc_pre/weight.pt`、`mhc_hc_pre/weight_rmsnorm.pt`、`mhc_hc_pre/output_scale.pt`
  - `mhc_hc_post/input_hidden_states.pt`、`mhc_hc_post/input_scale.pt`、`mhc_hc_post/output.pt`、`mhc_hc_post/weight.pt`、`mhc_hc_post/weight_rmsnorm.pt`、`mhc_hc_post/output_scale.pt`

## 清理

- [ ] Task 14: 移除各文件中的旧 dump 辅助函数
  - 移除 `_shared_moe_dump`（`fused_moe.py`）
  - 移除 `_selector_dump`（`experts_selector.py`）
  - 移除 `_mlp_dump`（`moe_mlp.py`）
  - 移除 `_w8a8_dump_tensor`（`w8a8_dynamic.py`）
  - 清理各文件中不再需要的 `import os`

## 验证

- [ ] Task 15: 验证 dump 路径结构正确
  - 检查所有 `_moe_dump_sub` 调用使用 `算子名/数据名.pt` 结构
  - 检查没有遗漏的旧 dump 函数引用
  - 检查所有导入一致性

# Task Dependencies

- Task 1 是所有其他任务的前置依赖
- Task 2、Task 3、Task 4、Task 5、Task 6、Task 7 可以在 Task 1 完成后并行执行
- Task 8、Task 9、Task 10、Task 11、Task 12 可以在 Task 1 完成后并行执行
- Task 13 依赖 Task 1 完成
- Task 14 依赖 Task 8-12 完成
- Task 15 依赖 Task 14 完成
