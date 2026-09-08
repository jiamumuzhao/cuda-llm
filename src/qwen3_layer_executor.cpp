#include "llm/qwen3_layer_executor.h"

#include "llm/qwen3_layer_cuda.h"
#include "llm/qwen3_kv_cache.h"
#include "llm/qwen3_paged_kv_cache.h"

#include <utility>

namespace llm {

Tensor Qwen3LayerExecutor::prefill(
    const Tensor& hidden, const std::vector<int32_t>& position_ids,
    const DecoderLayerWeights& weights, float rms_norm_eps,
    float rope_theta) const {
  return qwen3_decoder_layer_cuda_trace_fp16(
             hidden, position_ids, weights, rms_norm_eps, rope_theta)
      .layer_output;
}

DecoderLayerOutput Qwen3LayerExecutor::prefill_with_trace(
    const Tensor& hidden, const std::vector<int32_t>& position_ids,
    const DecoderLayerWeights& weights, float rms_norm_eps,
    float rope_theta) const {
  const Qwen3LayerTrace trace = qwen3_decoder_layer_cuda_trace_fp16(
      hidden, position_ids, weights, rms_norm_eps, rope_theta);
  return {std::move(trace.k_rope), std::move(trace.v_linear),
          std::move(trace.layer_output)};
}

DecoderLayerOutput Qwen3LayerExecutor::prefill_batch(
    const Tensor& hidden, const std::vector<int32_t>& position_ids,
    const DecoderLayerWeights& weights, float rms_norm_eps,
    float rope_theta) const {
  const Qwen3BatchLayerTrace trace = qwen3_decoder_layer_cuda_trace_fp16_batch(
      hidden, position_ids, weights, rms_norm_eps, rope_theta);
  return {std::move(trace.k_rope), std::move(trace.v_linear),
          std::move(trace.layer_output)};
}

DecoderLayerOutput Qwen3LayerExecutor::prefill_batch_valid_lengths(
    const Tensor& hidden, const std::vector<int32_t>& position_ids,
    const Tensor& valid_lengths_cuda, const DecoderLayerWeights& weights,
    float rms_norm_eps, float rope_theta) const {
  const Qwen3BatchLayerTrace trace =
      qwen3_decoder_layer_cuda_trace_fp16_batch_valid_lengths(
          hidden, position_ids, valid_lengths_cuda, weights, rms_norm_eps,
          rope_theta);
  return {std::move(trace.k_rope), std::move(trace.v_linear),
          std::move(trace.layer_output)};
}

DecoderLayerOutput Qwen3LayerExecutor::prefill_packed(
    const Tensor& hidden, const Tensor& positions_cuda,
    const Tensor& offsets_cuda, const DecoderLayerWeights& weights,
    float rms_norm_eps, float rope_theta) const {
  const Qwen3PackedLayerTrace trace = qwen3_decoder_layer_cuda_trace_fp16_packed(
      hidden, positions_cuda, offsets_cuda, weights, rms_norm_eps, rope_theta);
  return {std::move(trace.k_rope), std::move(trace.v_linear),
          std::move(trace.layer_output)};
}

void Qwen3LayerExecutor::decode_batch_into(
    const Tensor& hidden, int32_t shared_position,
    const DecoderLayerWeights& weights,
    const std::vector<DecoderKvCache*>& caches, std::size_t layer_index,
    DecodeWorkspace& workspace, Tensor& output, float rms_norm_eps,
    float rope_theta) const {
  std::vector<Qwen3KvCache*> qwen3_caches;
  qwen3_caches.reserve(caches.size());
  for (DecoderKvCache* cache : caches)
    qwen3_caches.push_back(static_cast<Qwen3KvCache*>(cache));
  qwen3_decoder_layer_cuda_decode_fp16_batch_into(
      hidden, shared_position, weights, qwen3_caches, layer_index, workspace,
      output, rms_norm_eps, rope_theta);
}

void Qwen3LayerExecutor::decode_batch_variable_into(
    const Tensor& hidden, const Tensor& position_ids_cuda,
    const Tensor& cache_lengths_after_append_cuda,
    const DecoderLayerWeights& weights,
    const std::vector<DecoderKvCache*>& caches, std::size_t layer_index,
    DecodeWorkspace& workspace, Tensor& output, float rms_norm_eps,
    float rope_theta) const {
  std::vector<Qwen3KvCache*> qwen3_caches;
  qwen3_caches.reserve(caches.size());
  for (DecoderKvCache* cache : caches)
    qwen3_caches.push_back(static_cast<Qwen3KvCache*>(cache));
  qwen3_decoder_layer_cuda_decode_fp16_batch_variable_into(
      hidden, position_ids_cuda, cache_lengths_after_append_cuda, weights,
      qwen3_caches, layer_index, workspace, output, rms_norm_eps, rope_theta);
}

Tensor Qwen3LayerExecutor::decode_paged(
    const Tensor& hidden, int32_t position,
    const DecoderLayerWeights& weights, DecoderPagedKvCache& cache,
    std::size_t layer_index, const Tensor& block_table_cuda,
    float rms_norm_eps, float rope_theta) const {
  auto* qwen3_cache = static_cast<Qwen3PagedKvCache*>(&cache);
  return qwen3_decoder_layer_cuda_decode_fp16_paged(
      hidden, position, weights, *qwen3_cache, layer_index, block_table_cuda,
      rms_norm_eps, rope_theta);
}

void Qwen3LayerExecutor::decode_paged_into(
    const Tensor& hidden, int32_t position,
    const DecoderLayerWeights& weights, DecoderPagedKvCache& cache,
    std::size_t layer_index, const Tensor& block_table_cuda,
    DecodeWorkspace& workspace, Tensor& output, float rms_norm_eps,
    float rope_theta, cudaStream_t stream) const {
  auto* qwen3_cache = static_cast<Qwen3PagedKvCache*>(&cache);
  qwen3_decoder_layer_cuda_decode_fp16_paged_into(
      hidden, position, weights, *qwen3_cache, layer_index, block_table_cuda,
      workspace, output, rms_norm_eps, rope_theta, stream);
}

void Qwen3LayerExecutor::decode_paged_batch_into(
    const Tensor& hidden, const Tensor& positions_cuda,
    const std::vector<int32_t>& positions,
    const DecoderLayerWeights& weights,
    const std::vector<DecoderPagedKvCache*>& caches,
    std::size_t layer_index,
    const Tensor& block_tables_cuda, DecodeWorkspace& workspace,
    Tensor& output, float rms_norm_eps, float rope_theta) const {
  std::vector<Qwen3PagedKvCache*> qwen3_caches;
  qwen3_caches.reserve(caches.size());
  for (DecoderPagedKvCache* cache : caches)
    qwen3_caches.push_back(static_cast<Qwen3PagedKvCache*>(cache));
  qwen3_decoder_layer_cuda_decode_fp16_paged_batch_into(
      hidden, positions_cuda, positions, weights, qwen3_caches, layer_index,
      block_tables_cuda, workspace, output, rms_norm_eps, rope_theta);
}

}  // namespace llm
