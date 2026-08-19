# Phase 6.6b：B=1/2/4 Direct Paged Prefill

本阶段支持相同 sequence length 的 B=1/2/4 direct paged prefill。模型先校验所有
prompt、cache、pool 和 batch metadata，再为所有 row 调用 `begin_prefill(S)`；每层
复用已有 batched prefill trace，K/V 通过 `append_prefill_layer_kv_batched_slice`
直接 D2D 写入每个 row 的 paged physical block table，不经过 contiguous cache 或
`seed_from_contiguous_cache`。

所有 row 的 28 层写入、final norm/LM head 和 CUDA 错误确认完成后才统一 commit。
任一 begin、layer copy、fault injection 或 CUDA 异常都会对已 begin 的所有 row
执行 `abort_prefill()`，恢复 length、block table 和 pool ownership。B=1 的
`prefill_logits_paged` 复用同一 batch 核心，避免两套事务语义漂移。

`Qwen3PagedRequestScheduler` admission 已改为单请求 `prefill_logits_paged`，删除
contiguous prefill→seed 过渡；metrics action 为 `paged_prefill`。scheduler 仍然
每 step 只 admission 一个请求，不进行 prefill batching，既有 bounded batch-fill
decode 策略保持不变。

测试覆盖 B=1/2/4、context 1/4/16/17/31；B=2/4 每个 live row 都有明确的
fragmented/non-contiguous physical block table 证据，并验证 blocker 隔离、logits、
最后 token top-5、固定 seed sampling、全部 row/layer/token K/V bit-exact 及 prefill
后 batched decode 对齐。测试还覆盖 batch size/count、null/duplicate cache、不同
pool、长度/容量、token range、非空 cache、活动事务和不兼容 Qwen3 pool 的拒绝，
且每次拒绝后检查所有 cache 与 pool 状态不变，并验证空 cache 可复用。

当 begin_prefill 的中途 row 因 pool exhaustion 失败时，B=2/B=4 均验证已 begin
的前序 row 由 catch 路径全部 abort，所有 length/table/transaction 与 blocker
状态恢复；释放 blocker 后同一批可成功恢复且无泄漏。中间 layer fault rollback
同时覆盖 B=2 和 B=4，确认所有 row 回滚后可再次 prefill。scheduler admission
metrics 和资源回收也继续回归。

当前仍未完成：变长/packed paged prefill、scheduler prefill batching、paged prefill
attention、prefix sharing/COW、长上下文、FlashAttention、CUDA Graph 与性能优化。
