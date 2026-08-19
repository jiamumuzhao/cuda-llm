# Phase 6.2：共享 GPU Paged KV Block Pool

## Storage layout

`PagedKvCachePool` 在构造时一次性分配 CUDA/F16 Tensor：

```text
[num_layers, total_blocks, 2, block_size, num_kv_heads, head_dim]
```

维度 `2` 的 `0/1` 分别表示 K/V。对当前 Qwen3-0.6B 配置
`28/8/16/128`，`block_bytes()` 是完整的物理 K+V block：

```text
28 * 2 * 8 * 16 * 128 * sizeof(FP16) = 1835008 bytes
```

`kv_plane_bytes()` 是单 layer、单 physical block、单个 K 或 V plane：

```text
16 * 8 * 128 * sizeof(FP16) = 32768 bytes
```

`917504` bytes 表示单个 K 或 V、跨全部 28 layers、单 physical BlockId
的总大小（`28 * 32768`），不是 `kv_plane_bytes()`。`token_bytes()`
是连续的 `[num_kv_heads, head_dim]` 单 token。因此
`resident_bytes() == total_blocks * block_bytes()`；pointer API 仍分别返回
K 或 V 区域。

Pool 的 pointer API 先验证 layer、allocated block、K/V selector 和 block
offset，再返回 layer/block/KV 或单 token 的连续 `[num_kv_heads, head_dim]`
区域。free block 不会返回可写地址；同一 K/V block 内相邻 token 的指针差
恰好为 `token_bytes()`。

## Sequence lifetime

`PagedSequenceKvCache` 非拥有地引用 pool，并拥有自己的
`SequenceBlockTable`。pool 必须比所有 sequence cache 活得更久；本阶段不引入
shared ownership、锁或线程同步。append 只修改 CPU block table 并复用
`BlockManager`，不会分配新的 CUDA Tensor。析构/release_all 会归还其 block。

## 验证范围与限制

测试使用受检查的 host FP16 bit-pattern copy 验证不同 layer、K/V、logical token
和 sequence 之间无串扰，并验证 pool 构造后 append/release/copy 不产生额外项目
CUDA Tensor allocation。该 copy API 仅是 Phase 6.2 验证接口，不是 decode 热路径。

尚未实现 Q/K/V 写入接入、Paged Attention、scheduler 迁移、prefix sharing、
copy-on-write、GPU block table metadata、swap/offload 和 CUDA Graph。
