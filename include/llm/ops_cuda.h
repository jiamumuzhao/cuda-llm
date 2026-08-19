#pragma once
#include "tensor.h"
#include <cstddef>
#include <cstdint>
#include <vector>
namespace llm {
class PagedKvCachePool;
struct ValidLengthTransferStats {
  uint64_t h2d_uploads = 0;
  uint64_t d2h_validation_copies = 0;
};
struct VariableDecodeTransferStats {
  uint64_t h2d_uploads = 0;
  uint64_t d2h_validation_copies = 0;
};
ValidLengthTransferStats valid_length_transfer_stats();
void reset_valid_length_transfer_stats();
void record_valid_length_h2d_upload();
VariableDecodeTransferStats variable_decode_transfer_stats();
void reset_variable_decode_transfer_stats();
void record_variable_decode_h2d_upload();

struct SamplingConfig {
  float temperature = 0.0f;
  int32_t top_k = 0;
  float top_p = 1.0f;
  uint64_t seed = 0;
};

class CudaSampler {
 public:
  explicit CudaSampler(size_t max_vocab);
  ~CudaSampler();
  CudaSampler(const CudaSampler&) = delete;
  CudaSampler& operator=(const CudaSampler&) = delete;
  int32_t sample_last_row(const Tensor& logits_cuda,
                          const SamplingConfig& config,
                          uint64_t draw_offset);
  int32_t sample_row(const Tensor& logits_bv, size_t batch_index,
                     const SamplingConfig& config, uint64_t draw_offset);
  int32_t sample_last_token_of_batch(const Tensor& logits_bsv,
                                     size_t batch_index,
                                     const SamplingConfig& config,
                                     uint64_t draw_offset);
 private:
  int32_t sample_row_offset(const Tensor& logits, size_t row_offset,
                            const SamplingConfig& config,
                            uint64_t draw_offset);
  void* workspace_ = nullptr;
  size_t max_vocab_ = 0;
};

Tensor cuda_linear(const Tensor&,const Tensor&);
Tensor cuda_rms_norm(const Tensor&,const Tensor&,float);
Tensor cuda_add(const Tensor&,const Tensor&);
Tensor cuda_swiglu(const Tensor&,const Tensor&);
Tensor cuda_rope(const Tensor&,const std::vector<int32_t>&,float);
Tensor cuda_rope_batched(const Tensor&,const std::vector<int32_t>&,float);
Tensor cuda_softmax_last_dim(const Tensor&);
Tensor cuda_gqa_attention(const Tensor&,const Tensor&,const Tensor&);
Tensor cuda_gqa_attention_batched(const Tensor&,const Tensor&,const Tensor&);
Tensor cuda_gqa_attention_batched_valid_lengths(const Tensor&, const Tensor&,
                                                const Tensor&, const Tensor&);
Tensor cuda_gqa_attention_batched_valid_lengths_checked(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&);
Tensor cuda_gqa_decode_attention(const Tensor& q_one, const Tensor& k_cache,
                                 const Tensor& v_cache, size_t cache_length);
Tensor cuda_gqa_decode_attention_batched(
    const Tensor& q_bhd, const std::vector<const Tensor*>& key_caches,
    const std::vector<const Tensor*>& value_caches, size_t cache_length);
void cuda_linear_out(const Tensor& input, const Tensor& weight, Tensor& output);
void cuda_rms_norm_out(const Tensor& input, const Tensor& weight, float eps,
                       Tensor& output);
void cuda_add_out(const Tensor& lhs, const Tensor& rhs, Tensor& output);
void cuda_swiglu_out(const Tensor& gate, const Tensor& up, Tensor& output);
void cuda_rope_batched_out(const Tensor& input,
                           const Tensor& positions_device, float theta,
                           Tensor& output);
void cuda_rope_batched_positions_out(const Tensor& input,
                                     const Tensor& positions_device,
                                     float theta, Tensor& output);
void cuda_gqa_decode_attention_batched_out(
    const Tensor& q_bhd, const std::vector<const Tensor*>& key_caches,
    const std::vector<const Tensor*>& value_caches, size_t cache_length,
    Tensor& output);
Tensor cuda_gqa_decode_attention_batched_variable_lengths(
    const Tensor& q_bhd, const std::vector<const Tensor*>& key_caches,
    const std::vector<const Tensor*>& value_caches,
    const Tensor& cache_lengths_after_append_cuda);
Tensor cuda_gqa_decode_attention_batched_variable_lengths_checked(
    const Tensor& q_bhd, const std::vector<const Tensor*>& key_caches,
    const std::vector<const Tensor*>& value_caches,
    const Tensor& cache_lengths_after_append_cuda);
void cuda_gqa_decode_attention_batched_variable_lengths_out(
    const Tensor& q_bhd, const std::vector<const Tensor*>& key_caches,
    const std::vector<const Tensor*>& value_caches,
    const Tensor& cache_lengths_after_append_cuda, Tensor& output);
void cuda_embedding_lookup_out(const Tensor& embedding_weight,
                               const std::vector<int32_t>& token_ids,
                               Tensor& token_ids_device, Tensor& output);
int32_t cuda_argmax_last_row(const Tensor& logits_cuda);
Tensor cuda_embedding_lookup(const Tensor& embedding_weight_cuda,
                             const std::vector<int32_t>& token_ids);
Tensor cuda_lm_head(const Tensor& hidden_states_cuda,
                    const Tensor& tied_token_embedding_cuda);
Tensor cuda_paged_gqa_attention_decode(
    const Tensor& q, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& device_block_table_i32, std::size_t kv_length,
    std::size_t num_q_heads, std::size_t num_kv_heads, std::size_t head_dim);

// Correctness-first batched paged GQA decode. The public convenience wrapper
// defensively validates device metadata; the caller-provided output path used
// by the Qwen3 hot path assumes metadata was validated by the model workspace
// and performs no metadata D2H copy or Tensor allocation.
Tensor cuda_paged_gqa_attention_decode_batch(
    const Tensor& q_bhd, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& block_tables_bm_i32,
    const Tensor& positions_before_append_b_i32,
    std::size_t num_q_heads, std::size_t num_kv_heads, std::size_t head_dim);
void cuda_paged_gqa_attention_decode_batch_out(
    const Tensor& q_bhd, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& block_tables_bm_i32,
    const Tensor& positions_before_append_b_i32,
    std::size_t num_q_heads, std::size_t num_kv_heads, std::size_t head_dim,
    Tensor& output_bhd);
void reset_cuda_paged_gqa_attention_decode_batch_launch_count();
std::uint64_t cuda_paged_gqa_attention_decode_batch_launch_count();
}
