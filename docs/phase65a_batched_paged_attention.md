# Phase 6.5a：真正 batched Paged GQA Decode Attention

本阶段将 Qwen3 paged decode 的 attention 从逐 row single-attention 调用改为一次
覆盖整个 B=1/2/4 batch 的 CUDA kernel launch。输入布局为：

```text
q                  [B, num_q_heads, head_dim] FP16 CUDA
block_tables       [B, max_blocks]             I32 CUDA
positions_before   [B]                         I32 CUDA
output             [B, num_q_heads, head_dim] FP16 CUDA
```

每个 CUDA block 处理一个 `(batch_row, q_head)`，线程覆盖 `head_dim` 输出维度。
每个 row 使用自己的 `positions_before[row] + 1` KV length，并通过：

```text
logical_block  = token / block_size
physical_block = block_tables[row, logical_block]
offset         = token % block_size
kv_head        = q_head / (num_q_heads / num_kv_heads)
```

读取 Phase 6.2 的 `[layers, blocks, K/V, block_size, kv_heads, head_dim]` storage。
score、stable softmax maximum/denominator 和 V 累加均使用 FP32，最终写回 FP16。
不同 batch row 可以有不同 KV length，已覆盖长度 1/4/5/8、跨 block、非连续
physical block table 与 K/V tail sentinel。

Qwen3 layer 先完成所有 row 的 K/V append，再对整个 batch 发起一次 paged
attention kernel；B=1 的 `decode_logits_paged()` 也复用同一 batch API。移除了每个
row 的 Q/K/V Tensor 创建、D2D staging copy、single attention 调用和 attention 输出
拷贝。block table 与 position metadata 仍来自 model-owned workspace。

`cuda_paged_gqa_attention_decode_batch_out()` 只向 caller-provided output 写入，
不创建 CUDA Tensor，也不下载已验证 metadata；测试中 B=1/2/4 warmup 后 operator
自身的 Tensor allocation 为 0。模型整层仍可能有其他临时 Tensor allocation，未被
本阶段隐藏或宣称消除。测试 launch counter 验证每个 Qwen3 batch decode 的 28 层各
发起一次 batch attention launch。

当前实现是 correctness-first kernel，尚未实现 FlashAttention、warp/tiled 优化、
fused QKV、paged prefill、CUDA Graph、prefix sharing/COW 或 long-context。
