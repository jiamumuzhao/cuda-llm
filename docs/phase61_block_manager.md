# Phase 6.1：CPU Block Manager 与 Sequence Block Table

本阶段只实现 Paged KV 的 CPU 元数据基础，不绑定 CUDA pointer、Tensor、
`Qwen3KvCache` 或 scheduler。

## 映射

默认 block size 为 16 时，逻辑 token 使用：

```text
logical_block = logical_token / block_size
offset        = logical_token % block_size
block_id      = block_table[logical_block]
```

`SequenceBlockTable::locate` 只接受当前 `[0, token_count)` 范围内的 token。

## 所有权与事务

`BlockManager::allocate()` 返回 refcount 为 1 的已分配 block；`retain` 增加引用，
`release` 在最后一个引用释放时把 block 放回 free-list。每个
`SequenceBlockTable` 持有自己 block table 中 block 的一个引用；`release_all()`
幂等，析构时自动归还。

`append_tokens` 只有在需要跨越逻辑 block 边界时申请新 block。它先准备并申请
全部新 block，任何 pool 耗尽或异常都会释放本次申请的 block，恢复原 token count、
block table、free/used/refcount 状态。

## 当前限制

模块不是并发安全的，不实现 prefix sharing、copy-on-write、共享 GPU block pool、
Paged KV 写入、paged attention、scheduler 接入、swap/offload 或 long-context。
