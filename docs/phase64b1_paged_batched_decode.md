# Phase 6.4b-1：B=2/4 Paged Decode

本阶段提供独立的 `Qwen3CudaModel::decode_logits_paged_batch` 路径，支持
B=2/4 的不等 cache length（当前测试覆盖 `[4,6]` 与 `[3,4,6,8]`）。每个
batch row 保留自己的 `Qwen3PagedKvCache` 和 physical block table；paged
attention 读取对应 row 的 block table，因而不同 sequence 不共享逻辑 token
地址或 cache ownership。

调用开始时会为所有 row 建立 decode transaction，并在 28 层循环前一次性创建
每 row 的 device block-table tensor 及 positions metadata；这些 device metadata
在整次 layer loop 中复用。各层把新 K/V 追加到对应 cache 后，按 row 调用现有
paged decode attention。只有全部 row、全部层及最终 logits 同步成功后才统一
`commit_decode()`；异常时对所有已开始事务调用 `abort_decode()`，不会留下
半步长度或 pool block 泄漏。

## 验收覆盖

`test_qwen3_paged_batched_decode` 验证：

- B=2/B=4 混合长度与跨 block 多步 logits 对齐；
- layer 7 和 layer 27 新 token K/V 与串行 paged decode 逐 bit 对齐；
- greedy/top-5 使用的 logits 以及固定 seed sampling 对齐；
- 故障注入在指定 row/layer 后，所有 cache 的 length/table 和 pool 状态完整
  回滚，清除故障后下一次调用可以成功；
- batch size、token/cache 数、重复 cache、非法 token 和活动事务等拒绝路径。

本阶段仍是 correctness-first 的 per-row paged attention 调用，尚未实现真正
融合的 batched paged attention kernel、paged prefill、paged scheduler B=2/4、
GPU block-table workspace、prefix sharing/COW、long-context 或性能优化。
