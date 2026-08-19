# Phase 6.4b-2：Paged Decode GPU Metadata Workspace

`Qwen3CudaModel` 现在独占一个 `PagedDecodeMetadataWorkspace`，仅用于 paged
decode 的小型 GPU metadata，不承载 KV 数据：

- `positions`：CUDA/I32 `[4]`，每次调用只写入前 B 项；
- `block_tables`：CUDA/I32 `[4, ceil(32 / block_size)]`，每个 cache 写入自己
  的 row 前缀；当前 Qwen3 block size=16 时为 `[4,2]`；
- Tensor view 只改变 shape/offset、共享原 storage，不为每个 row 分配显存。

workspace 首次使用时按第一个兼容 pool 的 block size lazy 初始化，模型是单线程
使用约束；后续调用要求 block size 一致并复用同一组 buffer。positions 和每行
block table 在一次 decode transaction 中各上传一次，28 层沿用这些 view。仍保留
`make_device_block_table_i32()` 作为兼容 API；模型热路径使用
`copy_block_table_to_cuda(Tensor&)` 写入 caller-provided view。

当前 block size=16 时 resident bytes 为：

```text
positions[4]                 = 4 * 4  = 16 bytes
block_tables[4,2]             = 4 * 2 * 4 = 32 bytes
metadata workspace total      = 48 bytes
```

测试输出区分 metadata workspace 与整个 decode：warmup/lazy init 之后，workspace
resident bytes 不再变化，metadata allocation 为 0；模型只读快照
`last_paged_decode_metadata_upload_stats()` 在最近一次 metadata 准备区间内记录：

```text
cuda_malloc_calls
cuda_free_calls
cuda_allocated_bytes
cuda_freed_bytes
positions_uploads
block_table_uploads
```

该区间只覆盖 non-owning view 获取以及 positions/block-table H2D copy，不包含
embedding、28 层 decoder、attention、final norm、LM head、logits 或其它临时
Tensor。B=1 单请求路径的 upload 次数为 `1/1`；B=1/2/4 batch 路径分别为
`positions_uploads=1`、`block_table_uploads=B`，warmup 后四项 allocation/free
增量均为 0。整次 decode 仍会产生 layer 内临时 Tensor 和返回 logits 的
allocation，本阶段不消除这些分配。

`test_qwen3_paged_batched_decode` 覆盖 B=1/2/4、不同长度、跨 block、多步 logits
对齐、top-5、固定 seed sampling、pool exhaustion rollback 及拒绝路径，并执行
CUDA health check。

本阶段不实现真正 batched paged-attention kernel、paged scheduler B=2/4、paged
prefill、prefix sharing/COW、long-context 或其它层内性能优化。
