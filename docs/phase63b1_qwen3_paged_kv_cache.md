# Phase 6.3b-1：Qwen3PagedKvCache transaction baseline

`Qwen3PagedKvCache` 是基于 Phase 6.2 GPU pool 和 sequence block table 的
独立 correctness wrapper，不接入 `Qwen3CudaModel`、现有 contiguous
`Qwen3KvCache`、scheduler 或 paged attention 热路径。

## Transaction semantics

`begin_decode()` 为下一个 token 预留逻辑位置和必要 physical block，但
`length()` 仍是已 commit 的可见长度。每次 `append_layer_kv()` 要求 CUDA/F16
contiguous `[8,128]` K/V，并通过 checked device-to-device copy 写入对应 layer
的 paged K/V token。28 个 layer 全部成功后 `commit_decode()` 才增加 visible
length；漏写、重复写和非法顺序都会拒绝。

`abort_decode()` 恢复 begin 前的 length 与 block table，并归还本次新申请的
block。它不清零已经写入的物理显存，但缩短后的 logical table/length 使其
不可见。`release_all()` 会先 abort 再释放已 commit blocks，且可重复调用。

## Seed helper

`seed_from_contiguous_cache()` 仅用于测试/迁移验证：它从现有
`Qwen3KvCache` 的只读 CUDA key/value Tensor 逐 token、逐 layer 执行 D2D copy，
不发生 CPU round-trip，也不改变 source 的 length 或内容。它要求目标为空、
无活跃 transaction；失败时回滚并归还新 block，不是未来 decode 热路径 API。

本阶段只覆盖 Qwen3 的 28 layers、8 KV heads、128 head dim、FP16，最大序列
长度 32。尚未完成 `decode_logits_paged`、模型 K/V projection 接入、真实
paged attention decode、batch/scheduler 接入、prefix sharing/COW 或性能优化。
