#pragma once

#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm {

class Qwen3KvCache;
class Qwen3PagedKvCache;

// Host-side metadata for right-padded, equal-work-shape prefill. The input
// prompts are copied into padded_token_ids; callers retain ownership of the
// original prompts and cache objects.
struct PaddedPrefillBatch {
  size_t batch_size = 0;
  size_t max_seq_len = 0;
  std::vector<int32_t> padded_token_ids;
  std::vector<size_t> valid_lengths;
  std::vector<Qwen3KvCache*> caches;
};

struct PagedPaddedPrefillBatch {
  size_t batch_size = 0;
  size_t max_seq_len = 0;
  std::vector<int32_t> padded_token_ids;
  std::vector<size_t> valid_lengths;
  std::vector<Qwen3PagedKvCache*> caches;
};

PagedPaddedPrefillBatch build_paged_padded_prefill_batch(
    const std::vector<std::vector<int32_t>>& prompts,
    const std::vector<Qwen3PagedKvCache*>& caches);

PaddedPrefillBatch build_padded_prefill_batch(
    const std::vector<std::vector<int32_t>>& prompts,
    const std::vector<Qwen3KvCache*>& caches);

// Selects each row's last valid vocabulary row from [B,S,V] into an owned
// [B,V] tensor. The returned tensor is safe to retain across later calls.
Tensor select_last_valid_logits(const Tensor& padded_logits,
                                const std::vector<size_t>& valid_lengths);

}  // namespace llm
