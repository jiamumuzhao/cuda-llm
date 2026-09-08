#include "llm/qwen3_layer_cuda.h"
#include "llm/cuda_check.h"
#include "llm/ops_cuda.h"
#include "llm/qwen3_kv_cache.h"
#include "llm/qwen3_paged_kv_cache.h"
#include <stdexcept>
#include <string>

namespace llm {
namespace {

std::size_t g_paged_batch_fault_row = static_cast<std::size_t>(-1);
std::size_t g_paged_batch_fault_layer = static_cast<std::size_t>(-1);

[[noreturn]] void invalid(const char* function, const std::string& message) {
  throw std::invalid_argument(std::string(function) + ": " + message);
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::string result = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) result += ",";
    result += std::to_string(shape[i]);
  }
  return result + "]";
}

struct DecoderLayerShape {
  std::size_t hidden = 0;
  std::size_t q_heads = 0;
  std::size_t kv_heads = 0;
  std::size_t head_dim = 0;
  std::size_t intermediate = 0;
  std::size_t q_dim() const noexcept { return q_heads * head_dim; }
  std::size_t kv_dim() const noexcept { return kv_heads * head_dim; }
};

DecoderLayerShape infer_layer_shape(const Qwen3CudaLayerWeights& w,
                                     const char* function) {
  if (w.q_proj.shape().size() != 2 || w.k_proj.shape().size() != 2 ||
      w.q_norm.shape().size() != 1 || w.gate_proj.shape().size() != 2)
    invalid(function, "layer weights have invalid rank");
  const auto hidden = static_cast<std::size_t>(w.q_proj.shape()[1]);
  const auto head_dim = static_cast<std::size_t>(w.q_norm.shape()[0]);
  const auto q_dim = static_cast<std::size_t>(w.q_proj.shape()[0]);
  const auto kv_dim = static_cast<std::size_t>(w.k_proj.shape()[0]);
  if (hidden == 0 || head_dim == 0 || q_dim % head_dim != 0 ||
      kv_dim % head_dim != 0)
    invalid(function, "layer weights have incompatible attention dimensions");
  return {hidden, q_dim / head_dim, kv_dim / head_dim, head_dim,
          static_cast<std::size_t>(w.gate_proj.shape()[0])};
}

void require_tensor(const Tensor& tensor, const std::vector<int64_t>& shape,
                    const char* name, DType expected_dtype, const char* function) {
  if (tensor.device() != DeviceType::CUDA || tensor.dtype() != expected_dtype ||
      !tensor.is_contiguous() || tensor.shape() != shape) {
    invalid(function, std::string(name) + " actual device=" +
            (tensor.device() == DeviceType::CUDA ? "CUDA" : "CPU") +
            " dtype=" + dtype_name(tensor.dtype()) +
            " contiguous=" + (tensor.is_contiguous() ? "true" : "false") +
            " shape=" + shape_string(tensor.shape()) +
            ", expected CUDA " + dtype_name(expected_dtype) +
            " contiguous shape=" + shape_string(shape));
  }
}

void validate(const Tensor& hidden, const std::vector<int32_t>& positions,
              const Qwen3CudaLayerWeights& w, float eps, float theta,
              DType expected_dtype, const char* function) {
  const auto shape = infer_layer_shape(w, function);
  if (hidden.shape().size() != 2)
    invalid(function, "hidden_states must have rank 2");
  const int64_t seq = hidden.shape()[0];
  require_tensor(hidden, {seq, static_cast<int64_t>(shape.hidden)},
                 "hidden_states", expected_dtype, function);
  if (seq < 1 || seq > 32)
    invalid(function, "unsupported seq_len=" + std::to_string(seq) + ", expected 1..32");
  if (positions.size() != static_cast<size_t>(seq))
    invalid(function, "position_ids length=" + std::to_string(positions.size()) +
            " does not match seq_len=" + std::to_string(seq));
  if (!(eps > 0.0f)) invalid(function, "rms_norm_eps must be > 0, actual=" + std::to_string(eps));
  if (!(theta > 0.0f)) invalid(function, "rope_theta must be > 0, actual=" + std::to_string(theta));
  require_tensor(w.input_norm, {static_cast<int64_t>(shape.hidden)}, "input_norm", expected_dtype, function);
  require_tensor(w.q_proj, {static_cast<int64_t>(shape.q_dim()), static_cast<int64_t>(shape.hidden)}, "q_proj", expected_dtype, function);
  require_tensor(w.k_proj, {static_cast<int64_t>(shape.kv_dim()), static_cast<int64_t>(shape.hidden)}, "k_proj", expected_dtype, function);
  require_tensor(w.v_proj, {static_cast<int64_t>(shape.kv_dim()), static_cast<int64_t>(shape.hidden)}, "v_proj", expected_dtype, function);
  require_tensor(w.q_norm, {static_cast<int64_t>(shape.head_dim)}, "q_norm", expected_dtype, function);
  require_tensor(w.k_norm, {static_cast<int64_t>(shape.head_dim)}, "k_norm", expected_dtype, function);
  require_tensor(w.o_proj, {static_cast<int64_t>(shape.hidden), static_cast<int64_t>(shape.q_dim())}, "o_proj", expected_dtype, function);
  require_tensor(w.post_attention_norm, {static_cast<int64_t>(shape.hidden)}, "post_attention_norm", expected_dtype, function);
  require_tensor(w.gate_proj, {static_cast<int64_t>(shape.intermediate), static_cast<int64_t>(shape.hidden)}, "gate_proj", expected_dtype, function);
  require_tensor(w.up_proj, {static_cast<int64_t>(shape.intermediate), static_cast<int64_t>(shape.hidden)}, "up_proj", expected_dtype, function);
  require_tensor(w.down_proj, {static_cast<int64_t>(shape.hidden), static_cast<int64_t>(shape.intermediate)}, "down_proj", expected_dtype, function);
}

}

Qwen3LayerTrace run_trace(
    const Tensor& hidden_states, const std::vector<int32_t>& positions,
    const Qwen3CudaLayerWeights& w, float eps, float theta, DType dtype,
    const char* function) {
  validate(hidden_states, positions, w, eps, theta, dtype, function);
  const auto shape = infer_layer_shape(w, function);
  const int64_t seq = hidden_states.shape()[0];
  Qwen3LayerTrace r;
  Tensor residual = hidden_states;
  r.input_norm = cuda_rms_norm(hidden_states, w.input_norm, eps);
  r.q_linear = cuda_linear(r.input_norm, w.q_proj);
  r.k_linear = cuda_linear(r.input_norm, w.k_proj);
  r.v_linear = cuda_linear(r.input_norm, w.v_proj);

  Tensor q = r.q_linear.reshape({seq, static_cast<int64_t>(shape.q_heads),
                                  static_cast<int64_t>(shape.head_dim)});
  Tensor k = r.k_linear.reshape({seq, static_cast<int64_t>(shape.kv_heads),
                                  static_cast<int64_t>(shape.head_dim)});
  Tensor v = r.v_linear.reshape({seq, static_cast<int64_t>(shape.kv_heads),
                                  static_cast<int64_t>(shape.head_dim)});
  r.q_norm_output = cuda_rms_norm(q, w.q_norm, eps);
  r.k_norm_output = cuda_rms_norm(k, w.k_norm, eps);
  r.q_normed = r.q_norm_output;
  r.k_normed = r.k_norm_output;
  r.q_rope = cuda_rope(r.q_norm_output, positions, theta);
  r.k_rope = cuda_rope(r.k_norm_output, positions, theta);

  Tensor attention_heads = cuda_gqa_attention(r.q_rope, r.k_rope, v);
  r.attention_output = attention_heads.reshape({seq, static_cast<int64_t>(shape.q_dim())});
  r.o_proj_output = cuda_linear(r.attention_output, w.o_proj);
  r.attention_residual = cuda_add(residual, r.o_proj_output);
  r.post_attention_norm = cuda_rms_norm(r.attention_residual, w.post_attention_norm, eps);
  r.gate_proj_output = cuda_linear(r.post_attention_norm, w.gate_proj);
  r.up_proj_output = cuda_linear(r.post_attention_norm, w.up_proj);
  r.swiglu_output = cuda_swiglu(r.gate_proj_output, r.up_proj_output);
  r.down_proj_output = cuda_linear(r.swiglu_output, w.down_proj);
  r.layer_output = cuda_add(r.attention_residual, r.down_proj_output);
  return r;
}

Qwen3LayerTrace qwen3_decoder_layer_cuda_trace(
    const Tensor& hidden_states, const std::vector<int32_t>& positions,
    const Qwen3CudaLayerWeights& w, float eps, float theta) {
  return run_trace(hidden_states, positions, w, eps, theta, DType::F32,
                   "qwen3_decoder_layer_cuda_trace");
}

Qwen3LayerTrace qwen3_decoder_layer_cuda_trace_fp16(
    const Tensor& hidden_states, const std::vector<int32_t>& positions,
    const Qwen3CudaLayerWeights& w, float eps, float theta) {
  return run_trace(hidden_states, positions, w, eps, theta, DType::F16,
                   "qwen3_decoder_layer_cuda_trace_fp16");
}

Tensor qwen3_decoder_layer_cuda_decode_fp16(
    const Tensor& hidden_one_cuda_f16, int32_t position_id,
    const Qwen3CudaLayerWeights& w, Tensor& k_cache_layer,
    Tensor& v_cache_layer, size_t previous_cache_length, float eps,
    float theta) {
  if (position_id < 0) invalid("qwen3_decoder_layer_cuda_decode_fp16", "position_id must be non-negative");
  const char* function = "qwen3_decoder_layer_cuda_decode_fp16";
  const auto shape = infer_layer_shape(w, function);
  require_tensor(hidden_one_cuda_f16, {1, static_cast<int64_t>(shape.hidden)}, "hidden_states", DType::F16, function);
  require_tensor(k_cache_layer, {k_cache_layer.shape().empty() ? 0 : k_cache_layer.shape()[0], static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)}, "k_cache", DType::F16, function);
  require_tensor(v_cache_layer, k_cache_layer.shape(), "v_cache", DType::F16, function);
  if (k_cache_layer.shape()[0] <= 0 || previous_cache_length >= size_t(k_cache_layer.shape()[0]))
    invalid(function, "previous_cache_length exceeds cache capacity");
  validate(hidden_one_cuda_f16, {position_id}, w, eps, theta, DType::F16, function);
  Tensor residual = hidden_one_cuda_f16;
  Tensor input_norm = cuda_rms_norm(hidden_one_cuda_f16, w.input_norm, eps);
  Tensor q_linear = cuda_linear(input_norm, w.q_proj).reshape({1, static_cast<int64_t>(shape.q_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor k_linear = cuda_linear(input_norm, w.k_proj).reshape({1, static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor v_linear = cuda_linear(input_norm, w.v_proj).reshape({1, static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor q_norm = cuda_rms_norm(q_linear, w.q_norm, eps);
  Tensor k_norm = cuda_rms_norm(k_linear, w.k_norm, eps);
  Tensor q_rope = cuda_rope(q_norm, {position_id}, theta);
  Tensor k_rope = cuda_rope(k_norm, {position_id}, theta);
  const size_t row_bytes = shape.kv_dim() * dtype_size(DType::F16);
  CUDA_CHECK(cudaMemcpy(static_cast<char*>(k_cache_layer.data()) + previous_cache_length * row_bytes,
                       k_rope.data(), row_bytes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(static_cast<char*>(v_cache_layer.data()) + previous_cache_length * row_bytes,
                       v_linear.data(), row_bytes, cudaMemcpyDeviceToDevice));
  Tensor attention = cuda_gqa_decode_attention(q_rope, k_cache_layer, v_cache_layer,
                                               previous_cache_length + 1).reshape({1, static_cast<int64_t>(shape.q_dim())});
  Tensor o_proj = cuda_linear(attention, w.o_proj);
  Tensor attention_residual = cuda_add(residual, o_proj);
  Tensor post_norm = cuda_rms_norm(attention_residual, w.post_attention_norm, eps);
  Tensor gate = cuda_linear(post_norm, w.gate_proj);
  Tensor up = cuda_linear(post_norm, w.up_proj);
  Tensor swiglu = cuda_swiglu(gate, up);
  Tensor down = cuda_linear(swiglu, w.down_proj);
  return cuda_add(attention_residual, down);
}

Tensor qwen3_decoder_layer_cuda_decode_fp16_paged(
    const Tensor& hidden_one_cuda_f16, int32_t position_id,
    const Qwen3CudaLayerWeights& w, Qwen3PagedKvCache& cache,
    size_t layer_index, const Tensor& device_block_table_i32, float eps,
    float theta) {
  const char* function = "qwen3_decoder_layer_cuda_decode_fp16_paged";
  if (position_id < 0) invalid(function, "position_id must be non-negative");
  require_tensor(hidden_one_cuda_f16, {1, 1024}, "hidden_states", DType::F16, function);
  if (layer_index >= 28) invalid(function, "layer_index must be less than 28");
  if (cache.length() == 0 || cache.length() >= cache.capacity())
    invalid(function, "cache must be non-empty and have capacity for one more token");
  if (!cache.in_decode_transaction())
    invalid(function, "cache must have an active decode transaction");
  if (device_block_table_i32.device() != DeviceType::CUDA ||
      device_block_table_i32.dtype() != DType::I32 ||
      !device_block_table_i32.is_contiguous() ||
      device_block_table_i32.shape().size() != 1 ||
      device_block_table_i32.shape()[0] !=
          static_cast<int64_t>(cache.block_table().size()))
    invalid(function, "device_block_table must be CUDA/I32/contiguous with cache block count");
  validate(hidden_one_cuda_f16, {position_id}, w, eps, theta, DType::F16, function);

  Tensor residual = hidden_one_cuda_f16;
  Tensor input_norm = cuda_rms_norm(hidden_one_cuda_f16, w.input_norm, eps);
  Tensor q_linear = cuda_linear(input_norm, w.q_proj).reshape({1, 16, 128});
  Tensor k_linear = cuda_linear(input_norm, w.k_proj).reshape({1, 8, 128});
  Tensor v_linear = cuda_linear(input_norm, w.v_proj).reshape({1, 8, 128});
  Tensor q_norm = cuda_rms_norm(q_linear, w.q_norm, eps);
  Tensor k_norm = cuda_rms_norm(k_linear, w.k_norm, eps);
  Tensor q_rope = cuda_rope(q_norm, {position_id}, theta);
  Tensor k_rope = cuda_rope(k_norm, {position_id}, theta);
  cache.append_layer_kv(layer_index, k_rope.reshape({8, 128}),
                        v_linear.reshape({8, 128}));
  Tensor attention = cuda_paged_gqa_attention_decode(
      q_rope.reshape({16, 128}), cache.pool(), layer_index,
      device_block_table_i32, cache.length() + 1,
      AttentionConfig{16, 8, 128, cache.pool().config().block_size, 512, true})
      .reshape({1, 2048});
  Tensor o_proj = cuda_linear(attention, w.o_proj);
  Tensor attention_residual = cuda_add(residual, o_proj);
  Tensor post_norm = cuda_rms_norm(attention_residual, w.post_attention_norm, eps);
  Tensor gate = cuda_linear(post_norm, w.gate_proj);
  Tensor up = cuda_linear(post_norm, w.up_proj);
  Tensor swiglu = cuda_swiglu(gate, up);
  Tensor down = cuda_linear(swiglu, w.down_proj);
  return cuda_add(attention_residual, down);
}

void qwen3_decoder_layer_cuda_decode_fp16_paged_into(
    const Tensor& hidden_one_cuda_f16, int32_t position_id,
    const Qwen3CudaLayerWeights& w, Qwen3PagedKvCache& cache,
    size_t layer_index, const Tensor& device_block_table_i32,
    DecodeWorkspace& ws, Tensor& output, float eps, float theta) {
  const char* function = "qwen3_decoder_layer_cuda_decode_fp16_paged_into";
  const auto shape = infer_layer_shape(w, function);
  if (position_id < 0) invalid(function, "position_id must be non-negative");
  require_tensor(hidden_one_cuda_f16, {1, static_cast<int64_t>(shape.hidden)},
                 "hidden_states", DType::F16, function);
  require_tensor(output, {1, static_cast<int64_t>(shape.hidden)}, "output",
                 DType::F16, function);
  if (layer_index >= 28) invalid(function, "layer_index must be less than 28");
  if (cache.length() == 0 || cache.length() >= cache.capacity())
    invalid(function, "cache must be non-empty and have capacity for one more token");
  if (!cache.in_decode_transaction())
    invalid(function, "cache must have an active decode transaction");
  if (device_block_table_i32.device() != DeviceType::CUDA ||
      device_block_table_i32.dtype() != DType::I32 ||
      !device_block_table_i32.is_contiguous() ||
      device_block_table_i32.shape().size() != 1 ||
      device_block_table_i32.shape()[0] !=
          static_cast<int64_t>(cache.block_table().size()))
    invalid(function, "device_block_table must be CUDA/I32/contiguous with cache block count");
  validate(hidden_one_cuda_f16, {position_id}, w, eps, theta, DType::F16,
           function);

  Tensor input_norm = ws.input_norm.prefix_first_dim(1);
  Tensor q_linear = ws.q_linear.prefix_first_dim(1);
  Tensor k_linear = ws.k_linear.prefix_first_dim(1);
  Tensor v_linear = ws.v_linear.prefix_first_dim(1);
  Tensor q = q_linear.reshape({1, static_cast<int64_t>(shape.q_heads),
                               static_cast<int64_t>(shape.head_dim)});
  Tensor k = k_linear.reshape({1, static_cast<int64_t>(shape.kv_heads),
                               static_cast<int64_t>(shape.head_dim)});
  Tensor v = v_linear.reshape({1, static_cast<int64_t>(shape.kv_heads),
                               static_cast<int64_t>(shape.head_dim)});
  Tensor q_norm = ws.q_norm.prefix_first_dim(1).reshape(
      {1, static_cast<int64_t>(shape.q_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor k_norm = ws.k_norm.prefix_first_dim(1).reshape(
      {1, static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor position_ids = ws.variable_position_ids.prefix_first_dim(1);
  CUDA_CHECK(cudaMemcpy(position_ids.data(), &position_id, sizeof(position_id),
                        cudaMemcpyHostToDevice));

  cuda_rms_norm_out(hidden_one_cuda_f16, w.input_norm, eps, input_norm);
  cuda_linear_out(input_norm, w.q_proj, q_linear);
  cuda_linear_out(input_norm, w.k_proj, k_linear);
  cuda_linear_out(input_norm, w.v_proj, v_linear);
  cuda_rms_norm_out(q, w.q_norm, eps, q_norm);
  cuda_rms_norm_out(k, w.k_norm, eps, k_norm);
  cuda_rope_token_positions_out(q_norm, position_ids, theta, q);
  cuda_rope_token_positions_out(k_norm, position_ids, theta, k);

  cache.append_layer_kv(
      layer_index,
      k.reshape({static_cast<int64_t>(shape.kv_heads),
                 static_cast<int64_t>(shape.head_dim)}),
      v.reshape({static_cast<int64_t>(shape.kv_heads),
                 static_cast<int64_t>(shape.head_dim)}));

  Tensor attention = ws.attention.prefix_first_dim(1).reshape(
      {static_cast<int64_t>(shape.q_heads), static_cast<int64_t>(shape.head_dim)});
  cuda_paged_gqa_attention_decode_out(
      q.reshape({static_cast<int64_t>(shape.q_heads),
                 static_cast<int64_t>(shape.head_dim)}),
      cache.pool(), layer_index, device_block_table_i32, cache.length() + 1,
      shape.q_heads, shape.kv_heads, shape.head_dim, attention);
  Tensor attention_flat = attention.reshape(
      {1, static_cast<int64_t>(shape.q_dim())});
  Tensor o_proj = ws.o_proj.prefix_first_dim(1);
  Tensor attention_residual = ws.attention_residual.prefix_first_dim(1);
  Tensor post_norm = ws.post_attention_norm.prefix_first_dim(1);
  Tensor gate = ws.gate.prefix_first_dim(1);
  Tensor up = ws.up.prefix_first_dim(1);
  Tensor down = ws.down.prefix_first_dim(1);
  cuda_linear_out(attention_flat, w.o_proj, o_proj);
  cuda_add_out(hidden_one_cuda_f16, o_proj, attention_residual);
  cuda_rms_norm_out(attention_residual, w.post_attention_norm, eps, post_norm);
  cuda_linear_out(post_norm, w.gate_proj, gate);
  cuda_linear_out(post_norm, w.up_proj, up);
  cuda_swiglu_out(gate, up, gate);
  cuda_linear_out(gate, w.down_proj, down);
  cuda_add_out(attention_residual, down, output);
}

void qwen3_set_paged_batch_fault_for_testing(size_t batch_row,
                                             size_t layer_index) {
  g_paged_batch_fault_row = batch_row;
  g_paged_batch_fault_layer = layer_index;
}

void qwen3_clear_paged_batch_fault_for_testing() {
  g_paged_batch_fault_row = static_cast<std::size_t>(-1);
  g_paged_batch_fault_layer = static_cast<std::size_t>(-1);
}

Tensor qwen3_decoder_layer_cuda_decode_fp16_paged_batch(
    const Tensor& hidden_bh, const Tensor& positions_cuda,
    const std::vector<int32_t>& positions,
    const Qwen3CudaLayerWeights& w,
    const std::vector<Qwen3PagedKvCache*>& caches, size_t layer_index,
    const Tensor& block_tables_bm_i32, float eps,
  float theta) {
  const char* function = "qwen3_decoder_layer_cuda_decode_fp16_paged_batch";
  const auto shape = infer_layer_shape(w, function);
  if (hidden_bh.shape().size() != 2) invalid(function, "hidden must be [B,1024]");
  const size_t batch = static_cast<size_t>(hidden_bh.shape()[0]);
  if (batch != 1 && batch != 2 && batch != 4) invalid(function, "batch must be 1, 2, or 4");
  require_tensor(hidden_bh, {static_cast<int64_t>(batch), static_cast<int64_t>(shape.hidden)}, "hidden", DType::F16, function);
  if (positions.size() != batch || caches.size() != batch)
    invalid(function, "positions and caches must match batch");
  if (block_tables_bm_i32.device() != DeviceType::CUDA ||
      block_tables_bm_i32.dtype() != DType::I32 ||
      !block_tables_bm_i32.is_contiguous() ||
      block_tables_bm_i32.shape().size() != 2 ||
      block_tables_bm_i32.shape()[0] != static_cast<int64_t>(batch) ||
      block_tables_bm_i32.shape()[1] <= 0)
    invalid(function, "block_tables must be CUDA/I32/contiguous [B,max_blocks]");
  if (positions_cuda.device() != DeviceType::CUDA || positions_cuda.dtype() != DType::I32 ||
      !positions_cuda.is_contiguous() || positions_cuda.shape() !=
          std::vector<int64_t>{static_cast<int64_t>(batch)})
    invalid(function, "positions_cuda must be CUDA/I32/contiguous [B]");
  if (layer_index >= 28 || !(eps > 0.0f) || !(theta > 0.0f))
    invalid(function, "invalid layer index, epsilon, or theta");
  require_tensor(w.input_norm, {static_cast<int64_t>(shape.hidden)}, "input_norm", DType::F16, function);
  require_tensor(w.q_proj, {static_cast<int64_t>(shape.q_dim()),static_cast<int64_t>(shape.hidden)}, "q_proj", DType::F16, function);
  require_tensor(w.k_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "k_proj", DType::F16, function);
  require_tensor(w.v_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "v_proj", DType::F16, function);
  require_tensor(w.q_norm, {static_cast<int64_t>(shape.head_dim)}, "q_norm", DType::F16, function);
  require_tensor(w.k_norm, {static_cast<int64_t>(shape.head_dim)}, "k_norm", DType::F16, function);
  require_tensor(w.o_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.q_dim())}, "o_proj", DType::F16, function);
  require_tensor(w.post_attention_norm, {static_cast<int64_t>(shape.hidden)}, "post_attention_norm", DType::F16, function);
  require_tensor(w.gate_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "gate_proj", DType::F16, function);
  require_tensor(w.up_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "up_proj", DType::F16, function);
  require_tensor(w.down_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.intermediate)}, "down_proj", DType::F16, function);
  for (size_t b = 0; b < batch; ++b) {
    if (!caches[b]) invalid(function, "null cache");
    if (!caches[b]->in_decode_transaction() || caches[b]->length() == 0 ||
        caches[b]->length() >= caches[b]->capacity() ||
        caches[b]->length() != static_cast<size_t>(positions[b]))
      invalid(function, "cache state does not match position");
  }
  Tensor residual = hidden_bh;
  Tensor norm = cuda_rms_norm(hidden_bh, w.input_norm, eps);
  Tensor q_linear = cuda_linear(norm, w.q_proj).reshape({static_cast<int64_t>(batch),static_cast<int64_t>(shape.q_heads),static_cast<int64_t>(shape.head_dim)});
  Tensor k_linear = cuda_linear(norm, w.k_proj).reshape({static_cast<int64_t>(batch),static_cast<int64_t>(shape.kv_heads),static_cast<int64_t>(shape.head_dim)});
  Tensor v_linear = cuda_linear(norm, w.v_proj).reshape({static_cast<int64_t>(batch),static_cast<int64_t>(shape.kv_heads),static_cast<int64_t>(shape.head_dim)});
  Tensor q_norm = cuda_rms_norm(q_linear, w.q_norm, eps);
  Tensor k_norm = cuda_rms_norm(k_linear, w.k_norm, eps);
  Tensor q_rope = q_norm.contiguous();
  Tensor k_rope = k_norm.contiguous();
  cuda_rope_batched_positions_out(q_norm, positions_cuda, theta, q_rope);
  cuda_rope_batched_positions_out(k_norm, positions_cuda, theta, k_rope);
  Tensor attention_heads(DType::F16,
                         {static_cast<int64_t>(batch), static_cast<int64_t>(shape.q_heads), static_cast<int64_t>(shape.head_dim)},
                         DeviceType::CUDA);
  for (size_t b = 0; b < batch; ++b) {
    if (g_paged_batch_fault_row == b && g_paged_batch_fault_layer == layer_index)
      throw std::runtime_error("paged batch injected fault");
    Tensor k_row = k_rope.slice_first_dim(b, 1).reshape({static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
    Tensor v_row = v_linear.slice_first_dim(b, 1).reshape({static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
    caches[b]->append_layer_kv(layer_index, k_row, v_row);
  }
  cuda_paged_gqa_attention_decode_batch_out(
      q_rope, caches[0]->pool(), layer_index, block_tables_bm_i32,
      positions_cuda,
      AttentionConfig{shape.q_heads, shape.kv_heads, shape.head_dim, caches[0]->pool().config().block_size, w.model_spec.attention.max_seq_len, true},
      attention_heads);
  Tensor attention = attention_heads.reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.q_dim())});
  Tensor o_proj = cuda_linear(attention, w.o_proj);
  Tensor attention_residual = cuda_add(residual, o_proj);
  Tensor post_norm = cuda_rms_norm(attention_residual, w.post_attention_norm, eps);
  Tensor gate = cuda_linear(post_norm, w.gate_proj);
  Tensor up = cuda_linear(post_norm, w.up_proj);
  Tensor swiglu = cuda_swiglu(gate, up);
  Tensor down = cuda_linear(swiglu, w.down_proj);
  return cuda_add(attention_residual, down);
}

void qwen3_decoder_layer_cuda_decode_fp16_paged_batch_into(
    const Tensor& hidden_bh, const Tensor& positions_cuda,
    const std::vector<int32_t>& positions,
    const Qwen3CudaLayerWeights& w,
    const std::vector<Qwen3PagedKvCache*>& caches, size_t layer_index,
    const Tensor& block_tables_bm_i32, DecodeWorkspace& ws, Tensor& output_bh,
  float eps, float theta) {
  const char* function = "qwen3_decoder_layer_cuda_decode_fp16_paged_batch_into";
  const auto shape = infer_layer_shape(w, function);
  if (hidden_bh.shape().size() != 2) invalid(function, "hidden must be [B,1024]");
  const size_t batch = static_cast<size_t>(hidden_bh.shape()[0]);
  if (batch != 1 && batch != 2 && batch != 4) invalid(function, "batch must be 1, 2, or 4");
  require_tensor(hidden_bh, {static_cast<int64_t>(batch), static_cast<int64_t>(shape.hidden)}, "hidden", DType::F16, function);
  require_tensor(output_bh, {static_cast<int64_t>(batch), static_cast<int64_t>(shape.hidden)}, "output", DType::F16, function);
  if (positions.size() != batch || caches.size() != batch) invalid(function, "positions and caches must match batch");
  if (positions_cuda.device() != DeviceType::CUDA || positions_cuda.dtype() != DType::I32 ||
      !positions_cuda.is_contiguous() || positions_cuda.shape() != std::vector<int64_t>{static_cast<int64_t>(batch)})
    invalid(function, "positions_cuda must be CUDA/I32/contiguous [B]");
  if (block_tables_bm_i32.device() != DeviceType::CUDA || block_tables_bm_i32.dtype() != DType::I32 ||
      !block_tables_bm_i32.is_contiguous() || block_tables_bm_i32.shape().size() != 2 ||
      block_tables_bm_i32.shape()[0] != static_cast<int64_t>(batch) || block_tables_bm_i32.shape()[1] <= 0)
    invalid(function, "block_tables must be CUDA/I32/contiguous [B,max_blocks]");
  if (layer_index >= 28 || !(eps > 0.0f) || !(theta > 0.0f)) invalid(function, "invalid layer index, epsilon, or theta");
  require_tensor(w.input_norm, {static_cast<int64_t>(shape.hidden)}, "input_norm", DType::F16, function);
  require_tensor(w.q_proj, {static_cast<int64_t>(shape.q_dim()),static_cast<int64_t>(shape.hidden)}, "q_proj", DType::F16, function);
  require_tensor(w.k_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "k_proj", DType::F16, function);
  require_tensor(w.v_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "v_proj", DType::F16, function);
  require_tensor(w.q_norm, {static_cast<int64_t>(shape.head_dim)}, "q_norm", DType::F16, function);
  require_tensor(w.k_norm, {static_cast<int64_t>(shape.head_dim)}, "k_norm", DType::F16, function);
  require_tensor(w.o_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.q_dim())}, "o_proj", DType::F16, function);
  require_tensor(w.post_attention_norm, {static_cast<int64_t>(shape.hidden)}, "post_attention_norm", DType::F16, function);
  require_tensor(w.gate_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "gate_proj", DType::F16, function);
  require_tensor(w.up_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "up_proj", DType::F16, function);
  require_tensor(w.down_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.intermediate)}, "down_proj", DType::F16, function);
  for (size_t b = 0; b < batch; ++b) {
    if (!caches[b]) invalid(function, "cache pointer must be non-null");
    if (!caches[b]->in_decode_transaction() || caches[b]->length() == 0 ||
        caches[b]->length() >= caches[b]->capacity() ||
        caches[b]->length() != static_cast<size_t>(positions[b]))
      invalid(function, "cache state does not match position");
  }
  Tensor input_norm = ws.input_norm.prefix_first_dim(batch);
  Tensor q_linear = ws.q_linear.prefix_first_dim(batch);
  Tensor k_linear = ws.k_linear.prefix_first_dim(batch);
  Tensor v_linear = ws.v_linear.prefix_first_dim(batch);
  Tensor q = q_linear.reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.q_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor k = k_linear.reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor v = v_linear.reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor q_norm = ws.q_norm.prefix_first_dim(batch).reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.q_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor k_norm = ws.k_norm.prefix_first_dim(batch).reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
  cuda_rms_norm_out(hidden_bh, w.input_norm, eps, input_norm);
  cuda_linear_out(input_norm, w.q_proj, q_linear);
  cuda_linear_out(input_norm, w.k_proj, k_linear);
  cuda_linear_out(input_norm, w.v_proj, v_linear);
  cuda_rms_norm_out(q, w.q_norm, eps, q_norm);
  cuda_rms_norm_out(k, w.k_norm, eps, k_norm);
  cuda_rope_batched_positions_out(q_norm, positions_cuda, theta, q);
  cuda_rope_batched_positions_out(k_norm, positions_cuda, theta, k);
  for (size_t b = 0; b < batch; ++b) {
    if (g_paged_batch_fault_row == b && g_paged_batch_fault_layer == layer_index)
      throw std::runtime_error("paged batch injected fault");
    Tensor k_row = k.slice_first_dim(b, 1).reshape({static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
    Tensor v_row = v.slice_first_dim(b, 1).reshape({static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
    caches[b]->append_layer_kv(layer_index, k_row, v_row);
  }
  Tensor attention = ws.attention.prefix_first_dim(batch).reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.q_heads), static_cast<int64_t>(shape.head_dim)});
  cuda_paged_gqa_attention_decode_batch_out(
      q, caches[0]->pool(), layer_index, block_tables_bm_i32,
      positions_cuda,
      AttentionConfig{shape.q_heads, shape.kv_heads, shape.head_dim, caches[0]->pool().config().block_size, w.model_spec.attention.max_seq_len, true},
      attention);
  Tensor attention_flat = attention.reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.q_dim())});
  Tensor o_proj = ws.o_proj.prefix_first_dim(batch);
  Tensor attention_residual = ws.attention_residual.prefix_first_dim(batch);
  Tensor post_norm = ws.post_attention_norm.prefix_first_dim(batch);
  Tensor gate = ws.gate.prefix_first_dim(batch);
  Tensor up = ws.up.prefix_first_dim(batch);
  Tensor down = ws.down.prefix_first_dim(batch);
  cuda_linear_out(attention_flat, w.o_proj, o_proj);
  cuda_add_out(hidden_bh, o_proj, attention_residual);
  cuda_rms_norm_out(attention_residual, w.post_attention_norm, eps, post_norm);
  cuda_linear_out(post_norm, w.gate_proj, gate);
  cuda_linear_out(post_norm, w.up_proj, up);
  cuda_swiglu_out(gate, up, gate);
  cuda_linear_out(gate, w.down_proj, down);
  cuda_add_out(attention_residual, down, output_bh);
}

static Qwen3BatchLayerTrace qwen3_decoder_layer_cuda_trace_fp16_batch_impl(
    const Tensor& hidden_bsh, const std::vector<int32_t>& positions,
    const Tensor* valid_lengths_cuda, const Qwen3CudaLayerWeights& w,
    float eps, float theta) {
  const char* function = "qwen3_decoder_layer_cuda_trace_fp16_batch";
  const auto shape = infer_layer_shape(w, function);
  if (hidden_bsh.shape().size() != 3)
    invalid(function, "hidden_bsh must have shape [B,S,hidden]");
  const int64_t batch = hidden_bsh.shape()[0], seq = hidden_bsh.shape()[1];
  require_tensor(hidden_bsh, {batch, seq, static_cast<int64_t>(shape.hidden)}, "hidden_bsh", DType::F16, function);
  if (batch < 1 || batch > 4) invalid(function, "batch size must be in {1,2,4}");
  if (seq < 1 || seq > 32) invalid(function, "seq_len must be in [1,32]");
  if (positions.size() != static_cast<size_t>(seq)) invalid(function, "position_ids length mismatch");
  if (valid_lengths_cuda &&
      (valid_lengths_cuda->device() != DeviceType::CUDA ||
       valid_lengths_cuda->dtype() != DType::I32 ||
       !valid_lengths_cuda->is_contiguous() ||
       valid_lengths_cuda->shape() != std::vector<int64_t>{batch}))
    invalid(function, "valid_lengths must be CUDA/I32/contiguous shape [B]");
  if (!(eps > 0.0f) || !(theta > 0.0f)) invalid(function, "epsilon and theta must be > 0");
  require_tensor(w.input_norm, {static_cast<int64_t>(shape.hidden)}, "input_norm", DType::F16, function);
  require_tensor(w.q_proj, {static_cast<int64_t>(shape.q_dim()),static_cast<int64_t>(shape.hidden)}, "q_proj", DType::F16, function);
  require_tensor(w.k_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "k_proj", DType::F16, function);
  require_tensor(w.v_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "v_proj", DType::F16, function);
  require_tensor(w.q_norm, {static_cast<int64_t>(shape.head_dim)}, "q_norm", DType::F16, function);
  require_tensor(w.k_norm, {static_cast<int64_t>(shape.head_dim)}, "k_norm", DType::F16, function);
  require_tensor(w.o_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.q_dim())}, "o_proj", DType::F16, function);
  require_tensor(w.post_attention_norm, {static_cast<int64_t>(shape.hidden)}, "post_attention_norm", DType::F16, function);
  require_tensor(w.gate_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "gate_proj", DType::F16, function);
  require_tensor(w.up_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "up_proj", DType::F16, function);
  require_tensor(w.down_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.intermediate)}, "down_proj", DType::F16, function);
  Tensor residual = hidden_bsh;
  Tensor norm = cuda_rms_norm(hidden_bsh, w.input_norm, eps);
  Tensor q_linear = cuda_linear(norm, w.q_proj).reshape({batch,seq,static_cast<int64_t>(shape.q_heads),static_cast<int64_t>(shape.head_dim)});
  Tensor k_linear = cuda_linear(norm, w.k_proj).reshape({batch,seq,static_cast<int64_t>(shape.kv_heads),static_cast<int64_t>(shape.head_dim)});
  Tensor v_linear = cuda_linear(norm, w.v_proj).reshape({batch,seq,static_cast<int64_t>(shape.kv_heads),static_cast<int64_t>(shape.head_dim)});
  Tensor q_norm = cuda_rms_norm(q_linear, w.q_norm, eps);
  Tensor k_norm = cuda_rms_norm(k_linear, w.k_norm, eps);
  Tensor q_rope = cuda_rope_batched(q_norm, positions, theta);
  Tensor k_rope = cuda_rope_batched(k_norm, positions, theta);
  Tensor attention = (valid_lengths_cuda
      ? cuda_gqa_attention_batched_valid_lengths_checked(q_rope, k_rope, v_linear, *valid_lengths_cuda)
      : cuda_gqa_attention_batched(q_rope, k_rope, v_linear)).reshape({batch,seq,static_cast<int64_t>(shape.q_dim())});
  Tensor o_proj = cuda_linear(attention, w.o_proj);
  Tensor attention_residual = cuda_add(residual, o_proj);
  Tensor post_norm = cuda_rms_norm(attention_residual, w.post_attention_norm, eps);
  Tensor gate = cuda_linear(post_norm, w.gate_proj);
  Tensor up = cuda_linear(post_norm, w.up_proj);
  Tensor swiglu = cuda_swiglu(gate, up);
  Tensor down = cuda_linear(swiglu, w.down_proj);
  Qwen3BatchLayerTrace result;
  result.k_rope = std::move(k_rope);
  result.v_linear = std::move(v_linear);
  result.layer_output = cuda_add(attention_residual, down);
  return result;
}

Qwen3BatchLayerTrace qwen3_decoder_layer_cuda_trace_fp16_batch(
    const Tensor& hidden_bsh, const std::vector<int32_t>& positions,
    const Qwen3CudaLayerWeights& w, float eps, float theta) {
  return qwen3_decoder_layer_cuda_trace_fp16_batch_impl(
      hidden_bsh, positions, nullptr, w, eps, theta);
}

Qwen3BatchLayerTrace qwen3_decoder_layer_cuda_trace_fp16_batch_valid_lengths(
    const Tensor& hidden_bsh, const std::vector<int32_t>& positions,
    const Tensor& valid_lengths_cuda, const Qwen3CudaLayerWeights& w,
    float eps, float theta) {
  return qwen3_decoder_layer_cuda_trace_fp16_batch_impl(
      hidden_bsh, positions, &valid_lengths_cuda, w, eps, theta);
}

Tensor qwen3_decoder_layer_cuda_decode_fp16_batch(
    const Tensor& hidden_bh, int32_t shared_position,
    const Qwen3CudaLayerWeights& w, const std::vector<Qwen3KvCache*>& caches,
    size_t layer_index, float eps, float theta) {
  const char* function = "qwen3_decoder_layer_cuda_decode_fp16_batch";
  const auto shape = infer_layer_shape(w, function);
  if (shared_position < 0) invalid(function, "shared_position must be non-negative");
  if (hidden_bh.shape().size() != 2) invalid(function, "hidden_bh must have shape [B,1024]");
  const int64_t batch = hidden_bh.shape()[0];
  require_tensor(hidden_bh, {batch, static_cast<int64_t>(shape.hidden)}, "hidden_bh", DType::F16, function);
  if (batch != static_cast<int64_t>(caches.size()) ||
      (batch != 1 && batch != 2 && batch != 4))
    invalid(function, "batch size must match caches and be 1, 2, or 4");
  if (!(eps > 0.0f) || !(theta > 0.0f)) invalid(function, "epsilon and theta must be > 0");
  require_tensor(w.input_norm, {static_cast<int64_t>(shape.hidden)}, "input_norm", DType::F16, function);
  require_tensor(w.q_proj, {static_cast<int64_t>(shape.q_dim()),static_cast<int64_t>(shape.hidden)}, "q_proj", DType::F16, function);
  require_tensor(w.k_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "k_proj", DType::F16, function);
  require_tensor(w.v_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "v_proj", DType::F16, function);
  require_tensor(w.q_norm, {static_cast<int64_t>(shape.head_dim)}, "q_norm", DType::F16, function);
  require_tensor(w.k_norm, {static_cast<int64_t>(shape.head_dim)}, "k_norm", DType::F16, function);
  require_tensor(w.o_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.q_dim())}, "o_proj", DType::F16, function);
  require_tensor(w.post_attention_norm, {static_cast<int64_t>(shape.hidden)}, "post_attention_norm", DType::F16, function);
  require_tensor(w.gate_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "gate_proj", DType::F16, function);
  require_tensor(w.up_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "up_proj", DType::F16, function);
  require_tensor(w.down_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.intermediate)}, "down_proj", DType::F16, function);
  size_t previous_length = 0;
  for (size_t b = 0; b < caches.size(); ++b) {
    if (caches[b] == nullptr) invalid(function, "cache pointer must be non-null");
    if (layer_index >= caches[b]->num_layers()) invalid(function, "layer index exceeds cache layers");
    if (b == 0) previous_length = caches[b]->length();
    if (caches[b]->length() != previous_length || previous_length == 0 ||
        previous_length >= caches[b]->capacity() || caches[b]->num_kv_heads() != shape.kv_heads ||
        caches[b]->head_dim() != shape.head_dim)
      invalid(function, "caches must have equal non-empty Qwen3-compatible length and capacity");
  }

  Tensor residual = hidden_bh;
  Tensor norm = cuda_rms_norm(hidden_bh, w.input_norm, eps);
  Tensor q = cuda_linear(norm, w.q_proj).reshape({batch, static_cast<int64_t>(shape.q_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor k = cuda_linear(norm, w.k_proj).reshape({batch, static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor v = cuda_linear(norm, w.v_proj).reshape({batch, static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor q_norm = cuda_rms_norm(q, w.q_norm, eps);
  Tensor k_norm = cuda_rms_norm(k, w.k_norm, eps);
  Tensor q_rope = cuda_rope_batched(q_norm.reshape({batch,1,static_cast<int64_t>(shape.q_heads),static_cast<int64_t>(shape.head_dim)}), {shared_position}, theta)
                     .reshape({batch,static_cast<int64_t>(shape.q_heads),static_cast<int64_t>(shape.head_dim)});
  Tensor k_rope = cuda_rope_batched(k_norm.reshape({batch,1,static_cast<int64_t>(shape.kv_heads),static_cast<int64_t>(shape.head_dim)}), {shared_position}, theta)
                     .reshape({batch,static_cast<int64_t>(shape.kv_heads),static_cast<int64_t>(shape.head_dim)});
  for (size_t b = 0; b < caches.size(); ++b)
    caches[b]->append_decode_layer_batched_slice(layer_index, k_rope, v, b, previous_length);
  std::vector<const Tensor*> keys;
  std::vector<const Tensor*> values;
  keys.reserve(caches.size()); values.reserve(caches.size());
  for (auto* cache : caches) {
    keys.push_back(&cache->key_cache(layer_index));
    values.push_back(&cache->value_cache(layer_index));
  }
  Tensor attention = cuda_gqa_decode_attention_batched(q_rope, keys, values,
                                                       previous_length + 1)
                         .reshape({batch, static_cast<int64_t>(shape.q_dim())});
  Tensor o_proj = cuda_linear(attention, w.o_proj);
  Tensor attention_residual = cuda_add(residual, o_proj);
  Tensor post_norm = cuda_rms_norm(attention_residual, w.post_attention_norm, eps);
  Tensor gate = cuda_linear(post_norm, w.gate_proj);
  Tensor up = cuda_linear(post_norm, w.up_proj);
  Tensor swiglu = cuda_swiglu(gate, up);
  Tensor down = cuda_linear(swiglu, w.down_proj);
  return cuda_add(attention_residual, down);
}

void qwen3_decoder_layer_cuda_decode_fp16_batch_into(
    const Tensor& hidden_bh, int32_t shared_position,
    const Qwen3CudaLayerWeights& w, const std::vector<Qwen3KvCache*>& caches,
    size_t layer_index, DecodeWorkspace& ws, Tensor& output, float eps,
    float theta) {
  const char* function = "qwen3_decoder_layer_cuda_decode_fp16_batch_into";
  const auto shape = infer_layer_shape(w, function);
  if (shared_position < 0) invalid(function, "shared_position must be non-negative");
  if (hidden_bh.shape().size() != 2) invalid(function, "hidden must be [B,1024]");
  const size_t batch = static_cast<size_t>(hidden_bh.shape()[0]);
  if (batch != 1 && batch != 2 && batch != 4)
    invalid(function, "batch must be 1, 2, or 4");
  require_tensor(hidden_bh, {static_cast<int64_t>(batch), static_cast<int64_t>(shape.hidden)}, "hidden", DType::F16, function);
  require_tensor(output, {static_cast<int64_t>(batch), static_cast<int64_t>(shape.hidden)}, "output", DType::F16, function);
  if (caches.size() != batch) invalid(function, "cache count mismatch");
  require_tensor(w.input_norm, {static_cast<int64_t>(shape.hidden)}, "input_norm", DType::F16, function);
  require_tensor(w.q_proj, {static_cast<int64_t>(shape.q_dim()),static_cast<int64_t>(shape.hidden)}, "q_proj", DType::F16, function);
  require_tensor(w.k_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "k_proj", DType::F16, function);
  require_tensor(w.v_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "v_proj", DType::F16, function);
  require_tensor(w.q_norm, {static_cast<int64_t>(shape.head_dim)}, "q_norm", DType::F16, function);
  require_tensor(w.k_norm, {static_cast<int64_t>(shape.head_dim)}, "k_norm", DType::F16, function);
  require_tensor(w.o_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.q_dim())}, "o_proj", DType::F16, function);
  require_tensor(w.post_attention_norm, {static_cast<int64_t>(shape.hidden)}, "post_attention_norm", DType::F16, function);
  require_tensor(w.gate_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "gate_proj", DType::F16, function);
  require_tensor(w.up_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "up_proj", DType::F16, function);
  require_tensor(w.down_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.intermediate)}, "down_proj", DType::F16, function);
  size_t previous_length = 0;
  for (size_t b = 0; b < batch; ++b) {
    if (!caches[b]) invalid(function, "cache pointer must be non-null");
    if (layer_index >= caches[b]->num_layers()) invalid(function, "layer index exceeds cache layers");
    if (b == 0) previous_length = caches[b]->length();
    if (caches[b]->length() != previous_length || previous_length == 0 ||
        previous_length >= caches[b]->capacity() || caches[b]->num_kv_heads() != shape.kv_heads ||
        caches[b]->head_dim() != shape.head_dim)
      invalid(function, "caches must have equal non-empty Qwen3-compatible length and capacity");
  }
  CUDA_CHECK(cudaMemcpy(ws.position_ids.data(), &shared_position, sizeof(shared_position), cudaMemcpyHostToDevice));
  Tensor input_norm = ws.input_norm.prefix_first_dim(batch);
  Tensor q_linear = ws.q_linear.prefix_first_dim(batch);
  Tensor k_linear = ws.k_linear.prefix_first_dim(batch);
  Tensor v_linear = ws.v_linear.prefix_first_dim(batch);
  Tensor q = q_linear.reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.q_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor k = k_linear.reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor v = v_linear.reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
  cuda_rms_norm_out(hidden_bh, w.input_norm, eps, input_norm);
  cuda_linear_out(input_norm, w.q_proj, q_linear);
  cuda_linear_out(input_norm, w.k_proj, k_linear);
  cuda_linear_out(input_norm, w.v_proj, v_linear);
  cuda_rms_norm_out(q, w.q_norm, eps, q);
  cuda_rms_norm_out(k, w.k_norm, eps, k);
  Tensor q_rope_view = q.reshape({static_cast<int64_t>(batch),1,static_cast<int64_t>(shape.q_heads),static_cast<int64_t>(shape.head_dim)});
  Tensor k_rope_view = k.reshape({static_cast<int64_t>(batch),1,static_cast<int64_t>(shape.kv_heads),static_cast<int64_t>(shape.head_dim)});
  cuda_rope_batched_out(q_rope_view, ws.position_ids, theta, q_rope_view);
  cuda_rope_batched_out(k_rope_view, ws.position_ids, theta, k_rope_view);
  for (size_t b = 0; b < batch; ++b)
    caches[b]->append_decode_layer_batched_slice(layer_index, k, v, b, previous_length);
  std::vector<const Tensor*> keys, values;
  keys.reserve(batch); values.reserve(batch);
  for (auto* cache : caches) {
    keys.push_back(&cache->key_cache(layer_index));
    values.push_back(&cache->value_cache(layer_index));
  }
  Tensor attention = ws.attention.prefix_first_dim(batch).reshape({static_cast<int64_t>(batch),static_cast<int64_t>(shape.q_heads),static_cast<int64_t>(shape.head_dim)});
  cuda_gqa_decode_attention_batched_out(q, keys, values, previous_length + 1, attention);
  Tensor attention_flat = attention.reshape({static_cast<int64_t>(batch),static_cast<int64_t>(shape.q_dim())});
  Tensor o_proj = ws.o_proj.prefix_first_dim(batch);
  Tensor attention_residual = ws.attention_residual.prefix_first_dim(batch);
  Tensor post_norm = ws.post_attention_norm.prefix_first_dim(batch);
  Tensor gate = ws.gate.prefix_first_dim(batch);
  Tensor up = ws.up.prefix_first_dim(batch);
  Tensor down = ws.down.prefix_first_dim(batch);
  cuda_linear_out(attention_flat, w.o_proj, o_proj);
  cuda_add_out(hidden_bh, o_proj, attention_residual);
  cuda_rms_norm_out(attention_residual, w.post_attention_norm, eps, post_norm);
  cuda_linear_out(post_norm, w.gate_proj, gate);
  cuda_linear_out(post_norm, w.up_proj, up);
  cuda_swiglu_out(gate, up, gate);
  cuda_linear_out(gate, w.down_proj, down);
  cuda_add_out(attention_residual, down, output);
}

void qwen3_decoder_layer_cuda_decode_fp16_batch_variable_into(
    const Tensor& hidden_bh, const Tensor& position_ids_cuda,
    const Tensor& cache_lengths_after_append_cuda,
    const Qwen3CudaLayerWeights& w, const std::vector<Qwen3KvCache*>& caches,
    size_t layer_index, DecodeWorkspace& ws, Tensor& output, float eps,
  float theta) {
  const char* function = "qwen3_decoder_layer_cuda_decode_fp16_batch_variable_into";
  const auto shape = infer_layer_shape(w, function);
  if (hidden_bh.shape().size() != 2)
    invalid(function, "hidden must be [B,1024]");
  const size_t batch = static_cast<size_t>(hidden_bh.shape()[0]);
  if (batch != 1 && batch != 2 && batch != 4)
    invalid(function, "batch must be 1, 2, or 4");
  require_tensor(hidden_bh, {static_cast<int64_t>(batch), static_cast<int64_t>(shape.hidden)}, "hidden", DType::F16, function);
  require_tensor(output, {static_cast<int64_t>(batch), static_cast<int64_t>(shape.hidden)}, "output", DType::F16, function);
  if (position_ids_cuda.device() != DeviceType::CUDA ||
      position_ids_cuda.dtype() != DType::I32 || !position_ids_cuda.is_contiguous() ||
      position_ids_cuda.shape() != std::vector<int64_t>{static_cast<int64_t>(batch)} ||
      cache_lengths_after_append_cuda.device() != DeviceType::CUDA ||
      cache_lengths_after_append_cuda.dtype() != DType::I32 ||
      !cache_lengths_after_append_cuda.is_contiguous() ||
      cache_lengths_after_append_cuda.shape() != std::vector<int64_t>{static_cast<int64_t>(batch)})
    invalid(function, "position/length metadata mismatch: position device=" +
            std::string(position_ids_cuda.device() == DeviceType::CUDA ? "CUDA" : "CPU") +
            " dtype=" + dtype_name(position_ids_cuda.dtype()) +
            " shape=" + shape_string(position_ids_cuda.shape()) +
            " length device=" +
            std::string(cache_lengths_after_append_cuda.device() == DeviceType::CUDA ? "CUDA" : "CPU") +
            " dtype=" + dtype_name(cache_lengths_after_append_cuda.dtype()) +
            " shape=" + shape_string(cache_lengths_after_append_cuda.shape()));
  if (caches.size() != batch) invalid(function, "cache count mismatch");
  require_tensor(w.input_norm, {static_cast<int64_t>(shape.hidden)}, "input_norm", DType::F16, function);
  require_tensor(w.q_proj, {static_cast<int64_t>(shape.q_dim()),static_cast<int64_t>(shape.hidden)}, "q_proj", DType::F16, function);
  require_tensor(w.k_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "k_proj", DType::F16, function);
  require_tensor(w.v_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "v_proj", DType::F16, function);
  require_tensor(w.q_norm, {static_cast<int64_t>(shape.head_dim)}, "q_norm", DType::F16, function);
  require_tensor(w.k_norm, {static_cast<int64_t>(shape.head_dim)}, "k_norm", DType::F16, function);
  require_tensor(w.o_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.q_dim())}, "o_proj", DType::F16, function);
  require_tensor(w.post_attention_norm, {static_cast<int64_t>(shape.hidden)}, "post_attention_norm", DType::F16, function);
  require_tensor(w.gate_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "gate_proj", DType::F16, function);
  require_tensor(w.up_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "up_proj", DType::F16, function);
  require_tensor(w.down_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.intermediate)}, "down_proj", DType::F16, function);
  std::vector<size_t> previous_lengths(batch);
  for (size_t b = 0; b < batch; ++b) {
    if (!caches[b]) invalid(function, "cache pointer must be non-null");
    if (layer_index >= caches[b]->num_layers()) invalid(function, "layer index exceeds cache layers");
    previous_lengths[b] = caches[b]->length();
    if (previous_lengths[b] == 0 || previous_lengths[b] >= caches[b]->capacity() ||
        caches[b]->num_kv_heads() != shape.kv_heads || caches[b]->head_dim() != shape.head_dim)
      invalid(function, "each cache must have non-empty Qwen3-compatible length and capacity");
  }
  Tensor input_norm = ws.input_norm.prefix_first_dim(batch);
  Tensor q_linear = ws.q_linear.prefix_first_dim(batch);
  Tensor k_linear = ws.k_linear.prefix_first_dim(batch);
  Tensor v_linear = ws.v_linear.prefix_first_dim(batch);
  Tensor q = q_linear.reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.q_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor k = k_linear.reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
  Tensor v = v_linear.reshape({static_cast<int64_t>(batch), static_cast<int64_t>(shape.kv_heads), static_cast<int64_t>(shape.head_dim)});
  cuda_rms_norm_out(hidden_bh, w.input_norm, eps, input_norm);
  cuda_linear_out(input_norm, w.q_proj, q_linear);
  cuda_linear_out(input_norm, w.k_proj, k_linear);
  cuda_linear_out(input_norm, w.v_proj, v_linear);
  cuda_rms_norm_out(q, w.q_norm, eps, q);
  cuda_rms_norm_out(k, w.k_norm, eps, k);
  cuda_rope_batched_positions_out(q, position_ids_cuda, theta, q);
  cuda_rope_batched_positions_out(k, position_ids_cuda, theta, k);
  for (size_t b = 0; b < batch; ++b)
    caches[b]->append_decode_layer_batched_slice(layer_index, k, v, b, previous_lengths[b]);
  std::vector<const Tensor*> keys, values;
  keys.reserve(batch); values.reserve(batch);
  for (auto* cache : caches) {
    keys.push_back(&cache->key_cache(layer_index));
    values.push_back(&cache->value_cache(layer_index));
  }
  Tensor attention = ws.attention.prefix_first_dim(batch).reshape({static_cast<int64_t>(batch),static_cast<int64_t>(shape.q_heads),static_cast<int64_t>(shape.head_dim)});
  cuda_gqa_decode_attention_batched_variable_lengths_out(
      q, keys, values, cache_lengths_after_append_cuda, attention);
  Tensor attention_flat = attention.reshape({static_cast<int64_t>(batch),static_cast<int64_t>(shape.q_dim())});
  Tensor o_proj = ws.o_proj.prefix_first_dim(batch);
  Tensor attention_residual = ws.attention_residual.prefix_first_dim(batch);
  Tensor post_norm = ws.post_attention_norm.prefix_first_dim(batch);
  Tensor gate = ws.gate.prefix_first_dim(batch);
  Tensor up = ws.up.prefix_first_dim(batch);
  Tensor down = ws.down.prefix_first_dim(batch);
  cuda_linear_out(attention_flat, w.o_proj, o_proj);
  cuda_add_out(hidden_bh, o_proj, attention_residual);
  cuda_rms_norm_out(attention_residual, w.post_attention_norm, eps, post_norm);
  cuda_linear_out(post_norm, w.gate_proj, gate);
  cuda_linear_out(post_norm, w.up_proj, up);
  cuda_swiglu_out(gate, up, gate);
  cuda_linear_out(gate, w.down_proj, down);
  cuda_add_out(attention_residual, down, output);
}

Qwen3PackedLayerTrace qwen3_decoder_layer_cuda_trace_fp16_packed(
    const Tensor& hidden_th, const Tensor& positions_cuda,
    const Tensor& offsets_cuda, const Qwen3CudaLayerWeights& w,
    float eps, float theta) {
  const char* function = "qwen3_decoder_layer_cuda_trace_fp16_packed";
  const auto shape = infer_layer_shape(w, function);
  if (hidden_th.device() != DeviceType::CUDA || hidden_th.dtype() != DType::F16 ||
      hidden_th.shape().size() != 2 || hidden_th.shape()[1] != static_cast<int64_t>(shape.hidden))
    invalid(function, "hidden_th must be CUDA/F16 [T,hidden]");
  const int64_t tokens = hidden_th.shape()[0];
  if (tokens < 1 || tokens > 512)
    invalid(function, "total packed token count must be in [1,512]");
  if (positions_cuda.device() != DeviceType::CUDA ||
      positions_cuda.dtype() != DType::I32 || !positions_cuda.is_contiguous() ||
      positions_cuda.shape() != std::vector<int64_t>{tokens})
    invalid(function, "positions must be CUDA/I32/contiguous [T]");
  if (offsets_cuda.device() != DeviceType::CUDA ||
      offsets_cuda.dtype() != DType::I32 || !offsets_cuda.is_contiguous())
    invalid(function, "offsets must be CUDA/I32/contiguous [B+1]");
  require_tensor(w.input_norm, {static_cast<int64_t>(shape.hidden)}, "input_norm", DType::F16, function);
  require_tensor(w.q_proj, {static_cast<int64_t>(shape.q_dim()),static_cast<int64_t>(shape.hidden)}, "q_proj", DType::F16, function);
  require_tensor(w.k_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "k_proj", DType::F16, function);
  require_tensor(w.v_proj, {static_cast<int64_t>(shape.kv_dim()),static_cast<int64_t>(shape.hidden)}, "v_proj", DType::F16, function);
  require_tensor(w.q_norm, {static_cast<int64_t>(shape.head_dim)}, "q_norm", DType::F16, function);
  require_tensor(w.k_norm, {static_cast<int64_t>(shape.head_dim)}, "k_norm", DType::F16, function);
  require_tensor(w.o_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.q_dim())}, "o_proj", DType::F16, function);
  require_tensor(w.post_attention_norm, {static_cast<int64_t>(shape.hidden)}, "post_attention_norm", DType::F16, function);
  require_tensor(w.gate_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "gate_proj", DType::F16, function);
  require_tensor(w.up_proj, {static_cast<int64_t>(shape.intermediate),static_cast<int64_t>(shape.hidden)}, "up_proj", DType::F16, function);
  require_tensor(w.down_proj, {static_cast<int64_t>(shape.hidden),static_cast<int64_t>(shape.intermediate)}, "down_proj", DType::F16, function);
  if (!(eps > 0.0f) || !(theta > 0.0f))
    invalid(function, "epsilon and theta must be > 0");
  Tensor residual = hidden_th;
  Tensor norm = cuda_rms_norm(hidden_th, w.input_norm, eps);
  Tensor q = cuda_linear(norm, w.q_proj).reshape({tokens,static_cast<int64_t>(shape.q_heads),static_cast<int64_t>(shape.head_dim)});
  Tensor k = cuda_linear(norm, w.k_proj).reshape({tokens,static_cast<int64_t>(shape.kv_heads),static_cast<int64_t>(shape.head_dim)});
  Tensor v = cuda_linear(norm, w.v_proj).reshape({tokens,static_cast<int64_t>(shape.kv_heads),static_cast<int64_t>(shape.head_dim)});
  q = cuda_rms_norm(q, w.q_norm, eps);
  k = cuda_rms_norm(k, w.k_norm, eps);
  Tensor q_rope(DType::F16, q.shape(), DeviceType::CUDA);
  Tensor k_rope(DType::F16, k.shape(), DeviceType::CUDA);
  cuda_rope_token_positions_out(q, positions_cuda, theta, q_rope);
  cuda_rope_token_positions_out(k, positions_cuda, theta, k_rope);
  const AttentionConfig attention_config{shape.q_heads, shape.kv_heads, shape.head_dim, 16, w.model_spec.attention.max_seq_len, true};
  Tensor attention = cuda_gqa_attention_packed(
      q_rope, k_rope, v, offsets_cuda, attention_config).reshape({tokens,static_cast<int64_t>(shape.q_dim())});
  Tensor o_proj = cuda_linear(attention, w.o_proj);
  Tensor attention_residual = cuda_add(residual, o_proj);
  Tensor post_norm = cuda_rms_norm(attention_residual, w.post_attention_norm, eps);
  Tensor gate = cuda_linear(post_norm, w.gate_proj);
  Tensor up = cuda_linear(post_norm, w.up_proj);
  Tensor swiglu = cuda_swiglu(gate, up);
  Tensor down = cuda_linear(swiglu, w.down_proj);
  Qwen3PackedLayerTrace result;
  result.k_rope = std::move(k_rope);
  result.v_linear = std::move(v);
  result.layer_output = cuda_add(attention_residual, down);
  return result;
}
}
