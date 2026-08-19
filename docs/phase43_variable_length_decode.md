# Phase 4.3：不同 KV 长度的静态 batched decode

Phase 4.3 支持 Qwen3 FP16 `B=1/2/4` 的 mixed-length static decode。每个请求
保留独立 contiguous KV Cache 和 GPU pointer；decode 不复制或重排完整 KV Cache。

## 每行 metadata 关系

对每个 batch row `b`，Host 在 decode 前读取已提交的
`cache_lengths_before[b] = cache[b]->length()`，并一次性上传两个 CUDA/I32
数组：

```text
position_ids[b]                 = cache_lengths_before[b]
append write index              = cache_lengths_before[b]
attention visible length        = cache_lengths_before[b] + 1
cache_lengths_after_append[b]   = cache_lengths_before[b] + 1
commit_decode(previous_length)  = cache_lengths_before[b]
committed length after decode   = cache_lengths_before[b] + 1
```

RoPE、GQA attention 和 K/V append 都按 row 使用自己的 metadata。attention 只
读取该 row 的 `[0, cache_lengths_after_append[b])`，不会读取其他请求的尾部。
全部 28 层成功并完成 CUDA 检查后逐条 commit；异常时所有 cache 执行
`abort_decode()`，逻辑长度不前进。

## 所有权与限制

position/length metadata 由 `Qwen3CudaModel` 的 `DecodeWorkspace` 持有，最大
batch=4，较小 batch 使用前缀 view；CUDA Tensor RAII 自动释放。模型热路径每次
variable-length decode 只上传一次小型 `[B]` metadata，checked attention 不做
D2H。公开低层 API 可为防御性校验读取 metadata，但不由模型热路径调用。

当前仍是静态 batch、每请求 contiguous KV、`B<=4`。尚未实现 continuous
batching、Paged KV/block manager、long context、CUDA Graph、左侧 padding 或
变长 decode 之外的通用调度策略。
