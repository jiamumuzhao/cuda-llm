#pragma once

#include "qwen3_layer_cuda.h"
#include "qwen3_kv_cache.h"
#include "qwen3_paged_kv_cache.h"
#include "ops_cuda.h"
#include "qwen3_decode_workspace.h"
#include "qwen3_padded_prefill.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llm {

// Test-only deterministic fault injection for direct paged prefill. Disabled
// by default and intentionally process-local; production callers never set it.
void qwen3_set_paged_prefill_fault_for_testing(std::size_t layer);
void qwen3_clear_paged_prefill_fault_for_testing();

struct GreedyGenerationResult {
  std::vector<int32_t> generated_ids;
  std::string stop_reason;
  size_t final_cache_length = 0;
};

struct PagedDecodeMetadataUploadStats {
  bool valid = false;
  uint64_t cuda_malloc_calls = 0;
  uint64_t cuda_free_calls = 0;
  uint64_t cuda_allocated_bytes = 0;
  uint64_t cuda_freed_bytes = 0;
  size_t positions_uploads = 0;
  size_t block_table_uploads = 0;
};

class Qwen3CudaModel {
 public:
  explicit Qwen3CudaModel(const std::filesystem::path& package_root);

  Tensor prefill_hidden_layers(const std::vector<int32_t>& token_ids,
                               const std::vector<int32_t>& position_ids,
                               size_t layer_count) const;
  Tensor prefill_final_hidden(const std::vector<int32_t>& token_ids,
                              const std::vector<int32_t>& position_ids) const;
  Tensor prefill_logits(const std::vector<int32_t>& token_ids,
                        const std::vector<int32_t>& position_ids) const;
  Tensor prefill_logits_with_cache(const std::vector<int32_t>& prompt_ids,
                                   Qwen3KvCache& cache) const;
  Tensor prefill_logits_paged(const std::vector<int32_t>& prompt_ids,
                              Qwen3PagedKvCache& cache) const;
  Tensor prefill_logits_paged_batch(
      const std::vector<std::vector<int32_t>>& prompt_ids_batch,
      const std::vector<Qwen3PagedKvCache*>& caches) const;
  Tensor prefill_logits_paged_packed_batch(
      const std::vector<std::vector<int32_t>>& prompt_ids_batch,
      const std::vector<Qwen3PagedKvCache*>& caches) const;
  Tensor prefill_logits_paged_padded_batch(
      const PagedPaddedPrefillBatch& batch) const;
  Tensor prefill_logits_batch_with_caches(
      const std::vector<std::vector<int32_t>>& prompt_ids_batch,
      const std::vector<Qwen3KvCache*>& caches) const;
  Tensor prefill_logits_padded_batch_with_caches(
      const PaddedPrefillBatch& batch) const;
  Tensor decode_logits(int32_t next_input_id, Qwen3KvCache& cache) const;
  Tensor decode_logits_paged(int32_t next_input_id,
                             Qwen3PagedKvCache& cache) const;
  Tensor decode_logits_paged_batch(
      const std::vector<int32_t>& next_input_ids,
      const std::vector<Qwen3PagedKvCache*>& caches) const;
  Tensor decode_logits_batch_with_caches(
      const std::vector<int32_t>& next_input_ids,
      const std::vector<Qwen3KvCache*>& caches) const;
  Tensor decode_logits_variable_length_batch_with_caches(
      const std::vector<int32_t>& next_input_ids,
      const std::vector<Qwen3KvCache*>& caches) const;
  GreedyGenerationResult generate_greedy(
      const std::vector<int32_t>& prompt_ids, size_t max_new_tokens,
      std::optional<int32_t> eos_token_id, size_t max_seq_len) const;
  GreedyGenerationResult generate_greedy_paged(
      PagedKvCachePool& pool, const std::vector<int32_t>& prompt_ids,
      size_t max_new_tokens, std::optional<int32_t> eos_token_id,
      size_t max_seq_len) const;
  GreedyGenerationResult generate_sampled_paged(
      PagedKvCachePool& pool, const std::vector<int32_t>& prompt_ids,
      size_t max_new_tokens, std::optional<int32_t> eos_token_id,
      size_t max_seq_len, const SamplingConfig& sampling) const;
  GreedyGenerationResult generate_sampled(
      const std::vector<int32_t>& prompt_ids, size_t max_new_tokens,
      std::optional<int32_t> eos_token_id, size_t max_seq_len,
      const SamplingConfig& sampling) const;

  size_t num_layers() const { return layers_.size(); }
  size_t resident_weight_bytes() const { return resident_weight_bytes_; }
  size_t decode_workspace_bytes() const {
    return decode_workspace_ ? decode_workspace_->resident_bytes() : 0;
  }
  size_t paged_decode_metadata_workspace_bytes() const {
    return paged_decode_metadata_workspace_
               ? paged_decode_metadata_workspace_->resident_bytes()
               : 0;
  }
  PagedDecodeMetadataUploadStats last_paged_decode_metadata_upload_stats() const {
    return last_paged_decode_metadata_upload_stats_;
  }

 private:
  Tensor token_embedding_;
  Tensor final_norm_;
  std::vector<Qwen3CudaLayerWeights> layers_;
  float rms_norm_eps_ = 0.0f;
  float rope_theta_ = 0.0f;
  size_t resident_weight_bytes_ = 0;
  std::unique_ptr<DecodeWorkspace> decode_workspace_;
  mutable std::unique_ptr<PagedDecodeMetadataWorkspace>
      paged_decode_metadata_workspace_;
  mutable PagedDecodeMetadataUploadStats last_paged_decode_metadata_upload_stats_;
};

}  // namespace llm
