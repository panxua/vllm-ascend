# Checklist

## 基础设施

- [x] `_moe_dump_should_dump` 正确判断 step 和 layer 是否匹配
- [x] `_moe_dump_base_dir` 返回正确的基础路径格式 `step{X}/rank{Y}/layer{Z}`
- [x] `_moe_dump_sub` 正确创建子目录并保存 tensor（使用 `算子名/数据名.pt` 结构）

## DSAttention

- [x] `attention_ds_attention/input_hidden_states.pt` 保存 attention 入口 hidden_states
- [x] `attention_q_a_proj/input_hidden_states.pt` 保存 wq_a 输入
- [x] `attention_q_a_proj/output_qr.pt` 保存 wq_a 输出 qr
- [x] `attention_q_a_layernorm/input.pt` 保存 RMSNorm 输入（decode 动态量化路径）
- [x] `attention_q_a_layernorm/output.pt` 保存 RMSNorm 输出
- [x] `attention_q_b_proj/input_qr.pt` 保存 wq_b 输入 qr
- [x] `attention_q_b_proj/output_q.pt` 保存 wq_b 输出 q
- [x] `attention_q_rmsnorm/input_q.pt` 保存 RMSNorm 输入
- [x] `attention_q_rmsnorm/output_q.pt` 保存 RMSNorm 输出 q
- [x] `attention_kv_proj/input_hidden_states.pt` 保存 wkv 输入
- [x] `attention_kv_proj/output_kv.pt` 保存 wkv 输出 kv
- [x] `attention_kv_layernorm/input_kv.pt` 保存 kv RMSNorm 输入
- [x] `attention_kv_layernorm/output_kv.pt` 保存 kv RMSNorm 输出
- [x] `attention_q_rope/input_q.pt` 保存 q rope 输入
- [x] `attention_q_rope/input_cos.pt` 保存 q rope cos 输入
- [x] `attention_q_rope/input_sin.pt` 保存 q rope sin 输入
- [x] `attention_q_rope/output_q.pt` 保存 q rope 输出
- [x] `attention_kv_rope/input_kv.pt` 保存 kv rope 输入
- [x] `attention_kv_rope/input_cos.pt` 保存 kv rope cos 输入
- [x] `attention_kv_rope/input_sin.pt` 保存 kv rope sin 输入
- [x] `attention_kv_rope/output_kv.pt` 保存 kv rope 输出
- [x] `attention_scatter_ori_kv/input_kv.pt` 保存 scatter 前 kv
- [x] `attention_scatter_ori_kv/output.pt` 保存 scatter 后输出
- [x] `attention_sparse_attn/input_q.pt` 保存 sparse attention q 输入
- [x] `attention_sparse_attn/input_kv.pt` 保存 sparse attention kv 输入
- [x] `attention_sparse_attn/output.pt` 保存 sparse attention 输出
- [x] `attention_output_rope/input.pt` 保存 output rope 输入
- [x] `attention_output_rope/output.pt` 保存 output rope 输出
- [x] `attention_o_a_proj/input.pt` 保存 wo_a 输入
- [x] `attention_o_a_proj/output.pt` 保存 wo_a 输出
- [x] `attention_o_b_proj/input.pt` 保存 wo_b 输入
- [x] `attention_o_b_proj/output.pt` 保存 wo_b 最终输出

## Compressor

- [x] `compressor/input_hidden_states.pt` 保存 hidden_states
- [x] `compressor/input_cos.pt` 保存 compress cos 输入
- [x] `compressor/input_sin.pt` 保存 compress sin 输入
- [x] `compressor/input_block_table.pt` 保存 block_table
- [x] `compressor/input_cu_seqlens.pt` 保存 cu_seqlens
- [x] `compressor/input_start_pos.pt` 保存 start_pos
- [x] `compressor/weight_wkv.pt` 保存 wkv 权重
- [x] `compressor/weight_wgate.pt` 保存 wgate 权重
- [x] `compressor/weight_ape.pt` 保存 ape 权重
- [x] `compressor/weight_norm.pt` 保存 norm 权重
- [x] `compressor/state_kv.pt` 保存 kv_state
- [x] `compressor/state_score.pt` 保存 score_state
- [x] `compressor/output.pt` 保存 output_compressed_kv

## Compressor Scatter

- [x] `scatter_cmp_kv/input.pt` 保存 scatter 前 compressed_kv
- [x] `scatter_cmp_kv/output.pt` 保存 scatter 后 compressed_kv

## DeepseekV4Indexer

- [x] `indexer_build_query_wq_b/input_qr.pt` 保存 wq_b 输入
- [x] `indexer_build_query_wq_b/output.pt` 保存 wq_b 输出
- [x] `indexer_build_weights_weights_proj/input.pt` 保存 weights 投影输入
- [x] `indexer_build_weights_weights_proj/output.pt` 保存 weights 投影输出
- [x] `indexer_compress_kv/input_hidden_states.pt` 保存 indexer compressor 输入
- [x] `indexer_compress_kv/output.pt` 保存 indexer compressor 输出
- [x] `indexer_rope_q/input_q.pt` 保存 indexer q rope 输入
- [x] `indexer_rope_q/input_cos.pt` 保存 indexer q rope cos 输入
- [x] `indexer_rope_q/input_sin.pt` 保存 indexer q rope sin 输入
- [x] `indexer_rope_q/output_q.pt` 保存 indexer q rope 输出
- [x] `indexer_hadamard_q/input.pt` 保存 hadamard q 输入
- [x] `indexer_hadamard_q/output.pt` 保存 hadamard q 输出
- [x] `indexer_hadamard_kv/input.pt` 保存 hadamard kv 输入
- [x] `indexer_hadamard_kv/output.pt` 保存 hadamard kv 输出
- [x] `indexer_dynamic_quant_q/input.pt` 保存 q 量化输入
- [x] `indexer_dynamic_quant_q/output.pt` 保存 q 量化输出
- [x] `indexer_dynamic_quant_kv/input.pt` 保存 kv 量化输入
- [x] `indexer_dynamic_quant_kv/output.pt` 保存 kv 量化输出
- [x] `indexer_cache_update/input_kv.pt` 保存 cache 更新 kv 输入
- [x] `indexer_cache_update/input_kv_scale.pt` 保存 cache 更新 kv_scale 输入
- [x] `indexer_cache_update/output.pt` 保存 cache 更新输出
- [x] `indexer_lightning/input_q.pt` 保存 lightning q 输入
- [x] `indexer_lightning/input_kv.pt` 保存 lightning kv 输入
- [x] `indexer_lightning/input_weights.pt` 保存 lightning weights 输入
- [x] `indexer_lightning/output.pt` 保存 lightning 输出 topk_idxs

## DeepseekV4Gate

- [x] `gate_matmul/input_hidden_states.pt` 保存 matmul 输入 hidden_states
- [x] `gate_matmul/output_router_logits.pt` 保存 matmul 输出 router_logits
- [x] `gate_matmul/weight.pt` 保存 gate_weight
- [x] `gate_matmul/bias_e_score_correction.pt` 保存 e_score_correction_bias
- [x] `gate_gating_top_k_hash/input_router_logits.pt` 保存 hash gating 输入
- [x] `gate_gating_top_k_hash/output_topk_weights.pt` 保存 hash gating topk_weights
- [x] `gate_gating_top_k_hash/output_topk_ids.pt` 保存 hash gating topk_ids
- [x] `gate_select_experts_native/input_router_logits.pt` 保存 native select 输入
- [x] `gate_select_experts_native/output_weights.pt` 保存 native select 输出
- [x] `gate_renormalize_topk_weights/input_weights.pt` 保存 renormalize 输入
- [x] `gate_renormalize_topk_weights/output_weights.pt` 保存 renormalize 输出
- [x] `gate_renormalize_topk_weights_sqrtsoftplus/input_weights.pt` 保存 sqrtsoftplus 输入
- [x] `gate_renormalize_topk_weights_sqrtsoftplus/output_weights.pt` 保存 sqrtsoftplus 输出
- [x] `gate_final/input_routed.pt` 保存 routed_experts 输入
- [x] `gate_final/input_shared.pt` 保存 shared_experts 输入
- [x] `gate_final/output_routed.pt` 保存 routed_experts 输出
- [x] `gate_final/output_shared.pt` 保存 shared_experts 输出

## FusedMoE

- [x] `ffn_forward/input_hidden_states.pt` 保存 forward 入口 hidden_states
- [x] `ffn_forward/input_router_logits.pt` 保存 forward 入口 router_logits
- [x] `ffn_dp_gather/input.pt` 保存 dp_gather 输入
- [x] `ffn_dp_gather/output.pt` 保存 dp_gather 输出
- [x] `ffn_dp_slice/input.pt` 保存 dp_slice 输入
- [x] `ffn_dp_slice/output.pt` 保存 dp_slice 输出
- [x] `ffn_tp_reduce/input.pt` 保存 tp_reduce 输入
- [x] `ffn_tp_reduce/output.pt` 保存 tp_reduce 输出
- [x] `ffn_ep_reduce/input.pt` 保存 ep_reduce 输入
- [x] `ffn_ep_reduce/output.pt` 保存 ep_reduce 输出
- [x] `ffn_shared_experts/input_hidden_states.pt` 保存 shared_experts 输入
- [x] `ffn_shared_experts/output_gate_up.pt` 保存 gate_up 输出
- [x] `ffn_shared_expert_gate/input.pt` 保存 shared_expert_gate 输入
- [x] `ffn_shared_expert_gate/output.pt` 保存 shared_expert_gate 输出
- [x] `ffn_forward_with_selected_experts/output.pt` 保存已选专家 forward 输出
- [x] `ffn_select_experts/input_hidden_states.pt` 保存 select_experts hidden_states 输入
- [x] `ffn_select_experts/input_router_logits.pt` 保存 select_experts router_logits 输入
- [x] `ffn_select_experts/output_topk_weights.pt` 保存 topk_weights
- [x] `ffn_select_experts/output_topk_ids.pt` 保存 topk_ids
- [x] `ffn_moe_active_topk/input.pt` 保存 moe_active_topk 输入
- [x] `ffn_moe_active_topk/output_topk_weights.pt` 保存 topk_weights
- [x] `ffn_moe_active_topk/output_topk_ids.pt` 保存 topk_ids
- [x] `ffn_forward_expert/input_hidden_states.pt` 保存 forward_expert hidden_states
- [x] `ffn_forward_expert/input_topk_weights.pt` 保存 forward_expert topk_weights
- [x] `ffn_forward_expert/input_topk_ids.pt` 保存 forward_expert topk_ids
- [x] `ffn_gate_proj/input.pt` 保存 gate_proj 输入
- [x] `ffn_gate_proj/weight_w13.pt` 保存 w13_weight
- [x] `ffn_gate_proj/weight_w2.pt` 保存 w2_weight
- [x] `ffn_moe_combine_result/input.pt` 保存 moe_combine_result 输入
- [x] `ffn_moe_combine_result/output.pt` 保存 moe_combine_result 输出

## moe_mlp

- [x] `ffn_group_gemm1/input.pt` 保存 GEMM1 输入
- [x] `ffn_group_gemm1/input_group_list.pt` 保存 GEMM1 group_list
- [x] `ffn_group_gemm1/output.pt` 保存 GEMM1 输出
- [x] `ffn_dequant_swiglu_quant/input.pt` 保存 SwiGLU 输入
- [x] `ffn_dequant_swiglu_quant/output.pt` 保存 SwiGLU 激活输出
- [x] `ffn_dequant_swiglu_quant/output_scale.pt` 保存 SwiGLU scale
- [x] `ffn_group_gemm2/input.pt` 保存 GEMM2 输入
- [x] `ffn_group_gemm2/output.pt` 保存 GEMM2 输出
- [x] `ffn_activation/input.pt` 保存 activation 输入
- [x] `ffn_activation/output.pt` 保存 activation 输出

## w8a8_dynamic

- [x] `ffn_dynamic_quant/input_hidden_states.pt` 保存 hidden_states
- [x] `ffn_dynamic_quant/input_router_logits.pt` 保存 router_logits

## mHC

- [x] `mhc_hc_pre/input_hidden_states.pt` 保存 hc_pre 输入 hidden_states
- [x] `mhc_hc_pre/input_scale.pt` 保存 hc_pre scale 输入
- [x] `mhc_hc_pre/output.pt` 保存 hc_pre 输出
- [x] `mhc_hc_pre/weight.pt` 保存 hc_pre 权重
- [x] `mhc_hc_pre/weight_rmsnorm.pt` 保存 hc_pre RMSNorm 权重
- [x] `mhc_hc_pre/output_scale.pt` 保存 hc_pre scale 输出
- [x] `mhc_hc_post/input_hidden_states.pt` 保存 hc_post 输入 hidden_states
- [x] `mhc_hc_post/input_scale.pt` 保存 hc_post scale 输入
- [x] `mhc_hc_post/output.pt` 保存 hc_post 输出
- [x] `mhc_hc_post/weight.pt` 保存 hc_post 权重
- [x] `mhc_hc_post/weight_rmsnorm.pt` 保存 hc_post RMSNorm 权重
- [x] `mhc_hc_post/output_scale.pt` 保存 hc_post scale 输出

## 清理验证

- [x] `fused_moe.py` 中 `_shared_moe_dump` 已移除
- [x] `experts_selector.py` 中 `_selector_dump` 已移除
- [x] `moe_mlp.py` 中 `_mlp_dump` 已移除
- [x] `w8a8_dynamic.py` 中 `_w8a8_dump_tensor` 已移除
- [x] 所有文件统一使用 `_moe_dump_sub`
- [x] 不再需要的 `import os` 已清理
- [x] 所有路径使用 `算子名/数据名.pt` 结构
