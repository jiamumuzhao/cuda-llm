#pragma once

#include "decoder_layer_weights.h"
#include "tensor.h"

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

namespace llm {

class DecoderKvCache;
class DecoderPagedKvCache;
class DecodeWorkspace;

struct DecoderLayerOutput {
  Tensor k_rope;
  Tensor v_linear;
  Tensor layer_output;
};

class DecoderLayerExecutor {
 public:
  virtual ~DecoderLayerExecutor() = default;
  virtual Tensor prefill(const Tensor& hidden,
                         const std::vector<int32_t>& position_ids,
                         const DecoderLayerWeights& weights, float rms_norm_eps,
                         float rope_theta) const = 0;
  virtual DecoderLayerOutput prefill_with_trace(
      const Tensor& hidden, const std::vector<int32_t>& position_ids,
      const DecoderLayerWeights& weights, float rms_norm_eps,
      float rope_theta) const = 0;
  virtual DecoderLayerOutput prefill_batch(
      const Tensor& hidden, const std::vector<int32_t>& position_ids,
      const DecoderLayerWeights& weights, float rms_norm_eps,
      float rope_theta) const = 0;
  virtual DecoderLayerOutput prefill_batch_valid_lengths(
      const Tensor& hidden, const std::vector<int32_t>& position_ids,
      const Tensor& valid_lengths_cuda, const DecoderLayerWeights& weights,
      float rms_norm_eps, float rope_theta) const = 0;
  virtual DecoderLayerOutput prefill_packed(
      const Tensor& hidden, const Tensor& positions_cuda,
      const Tensor& offsets_cuda, const DecoderLayerWeights& weights,
      float rms_norm_eps, float rope_theta) const = 0;
  virtual void decode_batch_into(
      const Tensor& hidden, int32_t shared_position,
      const DecoderLayerWeights& weights,
      const std::vector<DecoderKvCache*>& caches, std::size_t layer_index,
      DecodeWorkspace& workspace, Tensor& output, float rms_norm_eps,
      float rope_theta) const = 0;
  virtual void decode_batch_variable_into(
      const Tensor& hidden, const Tensor& position_ids_cuda,
      const Tensor& cache_lengths_after_append_cuda,
      const DecoderLayerWeights& weights,
      const std::vector<DecoderKvCache*>& caches, std::size_t layer_index,
      DecodeWorkspace& workspace, Tensor& output, float rms_norm_eps,
      float rope_theta) const = 0;
  virtual Tensor decode_paged(
      const Tensor& hidden, int32_t position,
      const DecoderLayerWeights& weights, DecoderPagedKvCache& cache,
      std::size_t layer_index, const Tensor& block_table_cuda,
      float rms_norm_eps, float rope_theta) const = 0;
  virtual void decode_paged_into(
      const Tensor& hidden, int32_t position,
      const DecoderLayerWeights& weights, DecoderPagedKvCache& cache,
      std::size_t layer_index, const Tensor& block_table_cuda,
      DecodeWorkspace& workspace, Tensor& output, float rms_norm_eps,
      float rope_theta, cudaStream_t stream = 0) const = 0;
  virtual void decode_paged_batch_into(
      const Tensor& hidden, const Tensor& positions_cuda,
      const std::vector<int32_t>& positions,
      const DecoderLayerWeights& weights,
      const std::vector<DecoderPagedKvCache*>& caches,
      std::size_t layer_index, const Tensor& block_tables_cuda,
      DecodeWorkspace& workspace, Tensor& output, float rms_norm_eps,
      float rope_theta) const = 0;
};

}  // namespace llm
