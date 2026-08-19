# Phase 4.1 right-padded static batch prefill

Phase 4.1 adds an opt-in right-padded prefill path for Qwen3-0.6B. The
host-only `PaddedPrefillBatch` builder accepts batch sizes 1, 2, and 4 and
prompts of length 1--32. It emits row-major `[B,S_max]` token IDs, valid
lengths, and one unique, capacity-checked contiguous KV cache per request.
Padding uses token ID zero and never mutates the caller's prompts.

The builder generates zero-valued right padding, and the model entry point
re-validates every metadata row before launching CUDA work. In particular,
every position in `[valid_length,S_max)` must still contain token `0`, so
callers cannot bypass the right-padding contract by hand-constructing the
public `PaddedPrefillBatch` structure.

`Qwen3CudaModel::prefill_logits_padded_batch_with_caches` runs one existing
FP16 batched prefill at `[B,S_max]` and returns full `[B,S_max,151936]` logits.
For each request, only the prefix `[0,valid_length)` of every layer's K/V is
copied and committed. Thus a short request does not inherit `S_max` in its
logical cache length. `select_last_valid_logits` copies the last valid row to
an owned `[B,151936]` tensor for sampling, so the selected result is safe to
retain after a later model call.

`Qwen3RequestScheduler::prefill_waiting_padded_static_batch` is explicitly
opt-in; the equal-length scheduler API is unchanged. It consumes only the
FIFO waiting prefix. If exactly three requests are available, it submits the
first two and leaves the third in `Waiting`.

The padded path uses the existing causal mask semantics: right padding is
future relative to every true token, so true-token logits remain equivalent to
serial prefill. Transactional cache writes have an `abort_prefill()` cleanup
path; commits happen only after CUDA synchronization. The path intentionally
does not implement left padding, variable-length decode, long-context
prefill, generic attention masks, or paged KV.

Regression coverage:

- `test_qwen3_padded_prefill`: host layout/rejection checks, B=2 lengths 4/6,
  B=4 lengths 3/4/6/8, serial logits/top-token/sampling alignment, exact cache
  lengths, short-cache capacity, isolation, and CUDA health.
- `test_qwen3_scheduler_padded_prefill`: mixed-length FIFO scheduling,
  per-request cache lengths, serial greedy token alignment, and the three-item
  FIFO rule.
