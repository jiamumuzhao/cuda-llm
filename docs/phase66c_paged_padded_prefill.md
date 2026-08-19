# Phase 6.6c: Variable-Length Direct Paged Prefill

`Qwen3CudaModel::prefill_logits_paged_padded_batch` now supports right-padded
B=1/2/4 prompts with different valid lengths. It reuses the existing
valid-length attention path, but each row begins a paged prefill transaction
for its own valid length and copies only that prefix of each layer's K/V into
the shared paged pool. Padding tokens never enter a logical block table.

All metadata, token IDs, right padding, cache states, compatibility, and common
pool ownership are checked before any transaction or CUDA work. All rows begin
before compute; an allocation, CUDA, or test fault failure aborts every begun
row and restores block ownership. Commit occurs only after all 28 layers and
logits complete successfully.

The regression test compares B=2 lengths 4/6 and B=4 lengths 3/4/6/8 against
the contiguous padded path, including last-valid logits, bit-exact K/V for all
layers and valid tokens, variable-length paged decode continuation, invalid
right-padding rejection, begin-time pool exhaustion rollback, and fault
rollback/retry.

This completes right-padded variable-length paged prefill. Packed prefill,
scheduler prefill batching, long context, prefix sharing/COW, FlashAttention,
CUDA Graph, and performance tuning remain future work.
