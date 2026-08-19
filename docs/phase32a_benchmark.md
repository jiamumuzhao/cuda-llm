# Phase 3.2a CUDA decode benchmark

`qwen3_cuda_benchmark` is an observation-only benchmark. It reuses the loaded
`Qwen3CudaModel`, `Qwen3KvCache`, and `CudaSampler`; it does not change model
forward, scheduler, KV-cache layout, or CUDA operator semantics. It is not a
CTest test.

## Measurement boundary

All latency samples use CUDA events on the current stream. KV-cache allocation
and prefill setup are outside the decode event. The `prefill` row measures only
the existing batched prefill call. The `decode` row measures one continuous
single-token decode per sample, after a prefill and warmup sequence; its
`context_start` and `context_end` columns describe the interval covered by the
aggregate, not a fixed-length claim. The `sample` row measures the existing GPU
sampler separately and invokes every batch row.

Warmup calls are discarded. `iterations` is the number of recorded samples.
Samples are sorted deterministically; p50 and p99 use linear interpolation at
`p * (count - 1)`. Decode `tokens_per_second` is
`batch_size * 1000 / mean_ms`. No CPU wall-clock value is used as a GPU latency
measurement.

## Allocation observability

The CSV allocation fields cover only CUDA allocations directly made by this
project's `Tensor` storage path. They do not claim to include CUDA runtime,
cuBLAS, driver, or library-internal workspace allocations. Each CSV row
aggregates one snapshot per recorded iteration: call counts and allocated/free
bytes are sums; `cuda_live_bytes` and `cuda_peak_live_bytes` are maxima;
`device_free_bytes_before` and `device_free_bytes_after` are minima. The
device total must remain constant across snapshots. Decode's greedy token
selection occurs after the operation snapshot, and its logits are destroyed
before the next reset. Sample's precomputed logits are not sample temporaries,
so their final free is intentionally excluded from the sample row.
`cudaMemGetInfo` fields are auxiliary device-level observations and can include
unrelated allocations. `reset` clears cumulative counters while preserving
currently live bytes as the measurement baseline.

## Phase 3.2b workspace interpretation

The Qwen3 FP16 decode path uses one model-owned `DecodeWorkspace`, resident
before benchmark warmup. It contains the maximum batch-4 hidden, projection,
attention, MLP, final-norm, token-id, and position-id buffers (163,860 bytes in
the current build); batch 1/2 use shared-storage prefix views. KV Cache storage
is not copied into the workspace. Layer temporaries are written through
`*_out` CUDA interfaces and are therefore excluded from steady-state Tensor
allocation counts. The returned logits remain caller-owned and may account for
one necessary Tensor allocation per decode step. The CSV allocation fields are
iteration aggregates, so divide decode call/byte totals by `count` for the
per-step value; the workspace regression requires that value to be at most two.

The current model prefill API accepts sequence lengths through 32. The default
benchmark contexts are therefore `{4,16}` and all user-supplied contexts are
validated before model loading, output-directory creation, or CSV creation.
Contexts above 32 are rejected with an explicit diagnostic until long-prefill
support is implemented.

## Nsight Systems

For a trace on a fixed machine, use for example:

```bash
nsys profile --trace=cuda,nvtx,cublas --stats=true \
  -o build/phase32a_nsys ./build/qwen3_cuda_benchmark \
  --model artifacts/phase15/qwen3-0.6b-f32 \
  --output build/phase32a.csv --contexts 4,16 --batches 1,2,4 \
  --warmup 5 --iterations 30 --max-seq-len 128
```

Results should not be compared directly across different GPUs, clock
policies, CUDA/toolchain versions, or build types.
