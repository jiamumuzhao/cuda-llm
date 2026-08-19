# Phase 4.4：静态 token-budget batching 与调度可观测性

Phase 4.4 在现有静态 scheduler API 之外提供 opt-in 的 token-budget 选择，
不改变默认 `step()`、等长 prefill/decode、padded prefill 或 Phase 4.3
variable-length decode 的行为。

## FIFO 前缀选择

所有候选都严格来自 Waiting/Decode 队列前缀，不会跳过前面的长请求选择后面
较短请求。支持的最终 batch 只有 `B=1/2/4`；当队列中有 3 个可选请求时，
最多提交前 2 个，第三个保持在队列中，避免产生模型不支持的 B=3。

Prefill 对候选前缀计算：

```text
S_max = max(prompt_length)
valid_tokens = sum(prompt_length)
padded_tokens = B * S_max
padding_tokens = padded_tokens - valid_tokens
padding_ratio = padding_tokens / padded_tokens
```

只有满足 `max_prefill_tokens`、`max_context_len` 和 `max_padding_ratio` 的
最大 FIFO 合法前缀才会提交。首请求本身无法满足预算时返回空 batch，保持
队列和 request 状态不变，并记录 `skipped_budget_count`。

Decode 使用 Phase 4.3 的每请求 KV length 和 variable-length CUDA path。每个
请求必须满足 `cache_length + 1 <= max_context_len`，并受
`max_decode_tokens` 与 `max_batch_size` 限制。Decode metrics 中
`valid_tokens=decode_tokens=batch_size`，`padded_tokens=padding_tokens=0`。

## Metrics 与 CSV

`SchedulerBatchMetrics` 为每轮不可变记录，包含 round ID、kind、FIFO selected
request IDs、提交前队列大小、提交后 active 数量、token 成本、padding ratio、
budget skip 次数和 `steady_clock` 的端到端 scheduler elapsed time。CSV 使用固定
表头，`selected_request_ids` 使用 `1|2|4` 的稳定格式，空 batch 为空字段。

每个 request 保存 submit 时间、首次生成 token 的逻辑 round 和 TTFT；首次 token
只记录一次。elapsed/TTFT 只用于观测，不作为测试中的固定性能断言。

## 范围

这仍是静态 batch：每次调用从当前队列选出一个不可变 FIFO 前缀并完成一次
prefill 或 decode。尚未实现 continuous batching、运行中动态加入/退出、取消、
超时、backpressure、Paged KV、long-context prefill 或 CUDA Graph。
