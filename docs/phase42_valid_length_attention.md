# Phase 4.2 explicit valid-length attention mask

The padded prefill path now creates one CUDA `I32` Tensor containing the
host-side `[B]` valid lengths before the first layer. That RAII-owned device
metadata is passed through all 28 layer calls and reused; it is not uploaded or
allocated once per layer.

The padded model path performs exactly one valid-length H2D upload and zero
valid-length D2H copies. The public low-level masked attention API retains
defensive value validation and may perform a D2H copy when called directly;
that API is not used by the model hot path. A thread-safe test counter exposes
these two metadata-transfer classes without changing kernel semantics.

For each row `b`, the masked GQA kernel applies:

- query valid iff `q_pos < valid_lengths[b]`;
- key valid iff `k_pos < valid_lengths[b]`;
- a valid query sees only valid keys with `k_pos <= q_pos`;
- an invalid query writes an all-zero output and does not read K/V.

The existing unmasked batched GQA API and equal-length prefill path are
unchanged. The padded path still uses shared positions `0..S_max-1`; this is a
right-padding-only constraint, not support for arbitrary positions or left
padding. Padded K/V tails are never copied into KV Cache, and each cache
commits its own valid length.

Coverage in `test_gqa_valid_lengths_cuda` checks F32/F16 row isolation, causal
valid prefixes, zero invalid queries, and invalid metadata/device/dtype paths.
`test_qwen3_padded_prefill` records that the model uses the
`valid_lengths_cuda` mask path and retains the existing serial logits, top-5,
greedy, sampling, and per-cache-length comparisons.

Left padding, arbitrary attention masks, long-context prefill, and variable-
length decode remain out of scope.
