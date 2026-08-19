# Phase 5.3：请求取消、超时与资源背压

Phase 5.1/5.2 的连续 batch、prefill/decode 混合、token budget、fairness 和
metrics 接口保持不变。`step_continuous` 仍是兼容入口；本阶段只在其 admission
和 batch-boundary 之前增加控制逻辑。

## 取消与终止原因

`cancel_request(request_id)` 对 Waiting 请求立即移出等待队列并返回
`FinishedRequest{stop_reason="cancelled"}`；对 Decode 请求在当前同步调用模型下
于下一次 batch 边界生效，释放其 contiguous KV Cache，不中断已经提交的 CUDA
batch。未知、已完成或重复取消返回 `false`，不会重复释放资源。终止原因包括
`eos`、`max_new_tokens`、`cancelled`、`queue_timeout`、`request_timeout` 和
`resource_exhausted`。

Scheduler 明确不是并发调用安全对象；本阶段没有线程、锁、异步 stream 或 GPU
preemption。

## 注入时钟与超时

`Qwen3RequestScheduler` 支持可选 monotonic `ClockNow`，生产默认使用
`steady_clock`，测试可注入确定性时钟而无需 sleep。`queue_timeout_ms` 只约束
Waiting 到 admission 前的时间；`request_timeout_ms` 从 submit 起约束整个生命
周期。每次 `step_continuous` 在 admission/batch 选择前检查超时。取消优先于
超时，EOS/max-token 完成后不会被改写。

## KV 背压与观测

`ContinuousSchedulerConfig` 新增 `max_waiting_requests` 和
`max_total_kv_cache_bytes`，零表示不限制。等待队列已满时 submit 以
`resource_exhausted` 完成且不分配 KV。KV budget 按当前
`Qwen3KvCache::resident_bytes()` 的实际布局计算：28 层、K/V 两份、8 个 KV
head、128 head_dim、F16；预算不足的 FIFO 请求保持 Waiting，完成、取消或超时
后立即归还预算。

Batch metrics 追加 finished/cancelled/timed_out/resource_rejected 计数及当前/峰值
KV resident bytes；request CSV 追加 stop reason 和两个 timeout 配置字段。
原有 CSV 列保持前缀兼容。

The scheduler uses FIFO legal prefixes. If both queues can run, it begins with
prefill, forces prefill after the configured consecutive decode limit, and
forces decode after the configured consecutive prefill limit. When the
preferred queue cannot form a legal batch, it tries the other queue. If neither
can run, it returns `false` without emitting an empty metrics record, changing
streaks, or mutating queue state. FIFO legal-prefix selection is unchanged.

Each executed batch retains the existing `SchedulerBatchMetrics` and CSV fields,
with appended consecutive prefill/decode streak columns. Request-level
snapshots are available through `request_metrics()` and
`export_request_metrics_csv()`, covering submit, admission, first-token, TTFT,
generated count, completion time, and stop reason. All scheduler times use a
monotonic clock; they are not CUDA event timings. CUDA errors are handled by
the existing budgeted batch paths: selected queues are restored and cache
transactions are aborted by the model path.

This still does not add Paged KV、prefix cache、CUDA Graph、HTTP/streaming server、
真正异步取消、swap/offload 或服务层 backpressure。
