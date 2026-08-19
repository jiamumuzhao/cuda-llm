# Phase 6.3a：单 sequence decode Paged GQA Attention

本阶段实现独立的 correctness baseline：单 sequence、单 query token 的
FP16 Paged GQA decode attention。它直接读取 Phase 6.2 GPU pool storage 和
上传到 CUDA 的 I32 device block table，不改动现有 contiguous
`Qwen3KvCache`、模型 decode 或 scheduler。

## API 与映射

`cuda_paged_gqa_attention_decode` 要求 `q` 为 CUDA/F16 contiguous
`[num_q_heads, head_dim]`，block table 为 CUDA/I32 contiguous `[num_logical_blocks]`，
并校验 layer、KV length、allocated BlockId、pool head dimensions 和 GQA ratio。
`PagedSequenceKvCache::make_device_block_table_i32()` 将当前 CPU block table
一次上传为独立 Tensor；空 table 被拒绝。

每个 Q head 使用：

```text
kv_head = q_head / (num_q_heads / num_kv_heads)
logical_block = token / block_size
offset = token % block_size
physical_block = device_block_table[logical_block]
```

kernel 遵循 Phase 6.2 layout
`[layers, blocks, 2, block_size, num_kv_heads, head_dim]`，对 K/V 访问使用
logical token 的物理 block 和 offset。score、max-subtraction softmax、概率和
V 累加均使用 FP32，输出为 CUDA/F16 `[num_q_heads, head_dim]`。

## 验证范围与限制

`test_paged_gqa_attention_cuda` 覆盖 `kv_length=1/4/5/8`、跨物理 block、
4 Q heads/2 KV heads 的映射、最后 block tail sentinel masking，并分别与
独立 CPU FP32 reference 和现有 contiguous GQA attention 对齐。还覆盖 table
dtype/device/shape、空 table、长度、layer、GQA ratio、free/非法 BlockId 和
CUDA health check 的拒绝路径。

这是正确性优先的 naive kernel，不是 FlashAttention 或性能实现。尚未实现
Q/K/V projection 直接写入 pool、Qwen3 model decode 切换、batch/prefill
paged attention、scheduler/block-manager 接入、prefix sharing/COW 或性能优化。
