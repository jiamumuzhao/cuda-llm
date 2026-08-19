# Phase 6.4c：Paged Decode Scheduler B=2/4

`Qwen3PagedRequestScheduler` 的 paged decode 现在支持配置为 `max_decode_batch_size`
1、2 或 4。prefill/admission 仍然是逐请求的 contiguous prefill + D2D seed；当
active 队列尚未填满目标 decode batch 时，scheduler 先按 FIFO admission 等待请求，
随后通过一次 `decode_logits_paged_batch` 执行 B=2/4 decode。默认值仍为 1，保留
原有 B=1 行为。

这是一种 correctness-first 的 bounded batch-fill policy：每次 `step()` 最多提交
一次 CUDA 工作；当 `waiting` 非空且 `active.size() < max_decode_batch_size` 时，
本 step 只执行一次单请求 admission prefill，并在返回前停止。因而在没有 defer、
请求直接完成、取消或 timeout 的情况下，最多连续执行
`max_decode_batch_size - active_before` 次 admission，随后下一步必定提交 decode
batch。默认 B=1 不改变原有调度行为。这不是最终的 token-budget prefill/decode
fairness 策略，复杂竞争仍留给 Phase 7。

每个 batch row 使用自己的 paged KV cache、block table、cache length 和 sampler
状态；模型层按 row 调用现有 paged attention，因此这是 correctness-first 的 batch
调度接线，不是融合的 batched paged-attention kernel。模型当前不支持 B=3，四路
batch 中途剩三路时 scheduler 会保留 FIFO 顺序，先执行 B=2，再执行 B=1。

模型 batch 调用保持 all-or-nothing cache transaction。若任一层或 row 失败，模型
会 abort 所有已开始的 cache；scheduler 将本次选中的 request IDs 按原顺序恢复到
active 队首，token、generated IDs 和 sampling draw offset 均不前进，并记录
`decode_error`、`decode_batch_size`、`selected_request_ids` 与
`decode_batch_error`。成功时每一行独立更新采样状态，EOS/max-token/max-seq 完成
后立即释放该行的 paged blocks。

`PagedSchedulerMetrics` 保留 `selected_request_id` 兼容字段，并追加
`decode_batch_size`、`selected_request_ids` 和 `decode_batch_error`，便于审计 FIFO
选择、B=1/2/4 行为和错误回滚。scheduler 仍是单线程、非并发安全对象；取消与
timeout 在 scheduler step/batch 边界生效。

测试还覆盖了 B=2 动态加入、batch 边界取消与 fake-clock timeout：已选 batch 不会
被新 waiting request 改写，完成/取消/超时的 row 会释放自己的 blocks，剩余 row
继续以合法的 B=1 或 B=2 batch 执行。故障注入时，request state、generated IDs、
last token、draw offset、cache length、active FIFO 和 pool free/used 快照均保持
不变，清除故障后原 batch 可重试成功。

本阶段未实现：paged prefill、真正融合的 batched paged-attention kernel、paged
continuous scheduler 的全面替换、prefix sharing/COW、long-context、CUDA Graph
以及其它性能优化。
