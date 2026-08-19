# Phase 6.4a：单请求 Paged Scheduler

`Qwen3PagedRequestScheduler` 是独立、opt-in、非线程安全的 scheduler。每次
`step()` 最多执行一个请求的一次 CUDA prefill 或一次 `decode_logits_paged`
decode，当前不提供 B=2/4 paged batch 或吞吐优化。

admission 使用保守预算：

```text
required_blocks = ceil(request.max_seq_len / pool.block_size)
```

这是 admission accounting，不会预先分配所有 block；实际 physical block 仍由
`Qwen3PagedKvCache` 的 seed/lazy append 分配。由于当前尚无 paged prefill，admission
暂时执行 contiguous prefill，随后 D2D seed 到 paged cache，并立即释放临时
contiguous cache。这是过渡路径，不代表最终 paged serving 架构。

waiting 队列严格 FIFO。pool block 不足或 active 数达到上限时请求保持 waiting，
不会触发 CUDA；前序请求完成、取消或超时后释放 block，后续请求才可 admission。
waiting 超限立即返回 `resource_exhausted`。取消在 scheduler batch 边界生效，
queue/request timeout 使用可注入 monotonic clock，所有终止路径都会释放 paged
blocks。metrics 记录 action、选中请求、队列数量、pool free/used 和 resident bytes。

即使 admission 预算检查通过，若 seed 阶段 pool 在 prefill 后耗尽，也属于可恢复
的 `deferred`：临时 target 与 contiguous seed cache 会释放，请求恢复到 FIFO 队首，
active 队列不变，下一次资源可用时可重新 admission。动态加入的请求在 B=1 模式下
不会抢占当前 active decode；active request timeout 在下一次 step/batch 边界生效并
归还其 blocks。

未实现：paged prefill、B=2/4 paged decode、paged batch attention、替换现有
`Qwen3RequestScheduler`、GPU block-table workspace、prefix sharing/COW 和性能优化。
