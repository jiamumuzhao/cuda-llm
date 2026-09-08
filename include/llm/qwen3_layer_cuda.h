#pragma once

#include "qwen3_layer.h"
#include "qwen3_decode_workspace.h"
#include "decoder_layer_weights.h"
#include <cuda_runtime_api.h>

namespace llm {

class Qwen3KvCache;
class Qwen3PagedKvCache;


using Qwen3CudaLayerWeights = DecoderLayerWeights;

struct Qwen3BatchLayerTrace {
  Tensor k_rope;
  Tensor v_linear;
  Tensor layer_output;
};

struct Qwen3PackedLayerTrace {
  Tensor k_rope;
  Tensor v_linear;
  Tensor layer_output;
};

Qwen3LayerTrace qwen3_decoder_layer_cuda_trace(
    const Tensor& hidden_states_cuda,
    const std::vector<int32_t>& position_ids,
    const Qwen3CudaLayerWeights& weights,
    float rms_norm_eps,
    float rope_theta);

Qwen3LayerTrace qwen3_decoder_layer_cuda_trace_fp16(
    const Tensor& hidden_states_cuda_f16,
    const std::vector<int32_t>& position_ids,
    const Qwen3CudaLayerWeights& weights_cuda_f16,
    float rms_norm_eps,
    float rope_theta);

Tensor qwen3_decoder_layer_cuda_decode_fp16(
    const Tensor& hidden_one_cuda_f16,
    int32_t position_id,
    const Qwen3CudaLayerWeights& weights,
    Tensor& k_cache_layer,
    Tensor& v_cache_layer,
    size_t previous_cache_length,
    float rms_norm_eps,
    float rope_theta);

Tensor qwen3_decoder_layer_cuda_decode_fp16_paged(
    const Tensor& hidden_one_cuda_f16, int32_t position_id,
    const Qwen3CudaLayerWeights& weights, Qwen3PagedKvCache& cache,
    size_t layer_index, const Tensor& device_block_table_i32,
    float rms_norm_eps, float rope_theta);

void qwen3_decoder_layer_cuda_decode_fp16_paged_into(
    const Tensor& hidden_one_cuda_f16, int32_t position_id,
    const Qwen3CudaLayerWeights& weights, Qwen3PagedKvCache& cache,
    size_t layer_index, const Tensor& device_block_table_i32,
    DecodeWorkspace& workspace, Tensor& output, float rms_norm_eps,
    float rope_theta, cudaStream_t stream = 0);

Tensor qwen3_decoder_layer_cuda_decode_fp16_paged_batch(
    const Tensor& hidden_bh, const Tensor& positions_cuda,
    const std::vector<int32_t>& positions,
    const Qwen3CudaLayerWeights& weights,
    const std::vector<Qwen3PagedKvCache*>& caches, size_t layer_index,
    const Tensor& block_tables_bm_i32,
    float rms_norm_eps, float rope_theta);

void qwen3_decoder_layer_cuda_decode_fp16_paged_batch_into(
    const Tensor& hidden_bh, const Tensor& positions_cuda,
    const std::vector<int32_t>& positions,
    const Qwen3CudaLayerWeights& weights,
    const std::vector<Qwen3PagedKvCache*>& caches, size_t layer_index,
    const Tensor& block_tables_bm_i32, DecodeWorkspace& workspace,
    Tensor& output_bh, float rms_norm_eps, float rope_theta);

// Test-only fault injection; disabled unless explicitly set and cleared.
void qwen3_set_paged_batch_fault_for_testing(size_t batch_row,
                                             size_t layer_index);
void qwen3_clear_paged_batch_fault_for_testing();

Qwen3BatchLayerTrace qwen3_decoder_layer_cuda_trace_fp16_batch(
    const Tensor& hidden_bsh,
    const std::vector<int32_t>& shared_position_ids,
    const Qwen3CudaLayerWeights& weights,
    float rms_norm_eps,
    float rope_theta);

Qwen3BatchLayerTrace qwen3_decoder_layer_cuda_trace_fp16_batch_valid_lengths(
    const Tensor& hidden_bsh,
    const std::vector<int32_t>& shared_position_ids,
    const Tensor& valid_lengths_cuda,
    const Qwen3CudaLayerWeights& weights,
    float rms_norm_eps,
    float rope_theta);

Qwen3PackedLayerTrace qwen3_decoder_layer_cuda_trace_fp16_packed(
    const Tensor& hidden_th, const Tensor& positions_cuda,
    const Tensor& offsets_cuda, const Qwen3CudaLayerWeights& weights,
    float rms_norm_eps, float rope_theta);

Tensor qwen3_decoder_layer_cuda_decode_fp16_batch(
    const Tensor& hidden_bh, int32_t shared_position,
    const Qwen3CudaLayerWeights& weights,
    const std::vector<Qwen3KvCache*>& caches, size_t layer_index,
    float rms_norm_eps, float rope_theta);

void qwen3_decoder_layer_cuda_decode_fp16_batch_into(
    const Tensor& hidden_bh, int32_t shared_position,
    const Qwen3CudaLayerWeights& weights,
    const std::vector<Qwen3KvCache*>& caches, size_t layer_index,
    DecodeWorkspace& workspace, Tensor& output,
    float rms_norm_eps, float rope_theta);
void qwen3_decoder_layer_cuda_decode_fp16_batch_variable_into(
    const Tensor& hidden_bh, const Tensor& position_ids_cuda,
    const Tensor& cache_lengths_after_append_cuda,
    const Qwen3CudaLayerWeights& weights,
    const std::vector<Qwen3KvCache*>& caches, size_t layer_index,
    DecodeWorkspace& workspace, Tensor& output,
    float rms_norm_eps, float rope_theta);


}
