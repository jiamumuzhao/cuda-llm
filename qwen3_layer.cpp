#include "llm/qwen3_layer.h"
#include "llm/ops.h"
#include <stdexcept>
#include <string>

namespace llm {
namespace {
void require_f32_2d(const Tensor& t, const std::vector<int64_t>& shape, const char* name) {
  if (t.device() != DeviceType::CPU || t.dtype() != DType::F32 || !t.is_contiguous() || t.shape() != shape)
    throw std::invalid_argument(std::string("qwen3 layer: ") + name + " must be contiguous CPU F32 with shape " +
                                std::to_string(shape[0]) + "x" + std::to_string(shape[1]));
}
void require_weight(const Tensor& t, const std::vector<int64_t>& shape, const char* name) {
  require_f32_2d(t, shape, name);
}
void require_weight_1d(const Tensor& t, int64_t n, const char* name) {
  if (t.device() != DeviceType::CPU || t.dtype() != DType::F32 || !t.is_contiguous() || t.shape() != std::vector<int64_t>{n})
    throw std::invalid_argument(std::string("qwen3 layer: ") + name + " must be contiguous CPU F32 with shape [" + std::to_string(n) + "]");
}
void validate(const Tensor& h, const std::vector<int32_t>& positions, const Qwen3LayerWeights& w,
              float eps, float theta) {
  require_f32_2d(h, {h.shape().size() == 2 ? h.shape()[0] : -1, 1024}, "hidden_states");
  if (h.shape().size() != 2) throw std::invalid_argument("qwen3 layer: hidden_states must have shape [T,1024]");
  if (positions.size() != static_cast<size_t>(h.shape()[0])) throw std::invalid_argument("qwen3 layer: position_ids length does not match T");
  if (!(eps >= 0.0f) || !(theta > 0.0f)) throw std::invalid_argument("qwen3 layer: eps must be non-negative and rope_theta positive");
  require_weight_1d(w.input_layernorm,1024,"input_layernorm"); require_weight(w.q_proj,{2048,1024},"q_proj");
  require_weight(w.k_proj,{1024,1024},"k_proj"); require_weight(w.v_proj,{1024,1024},"v_proj");
  require_weight_1d(w.q_norm,128,"q_norm"); require_weight_1d(w.k_norm,128,"k_norm");
  require_weight(w.o_proj,{1024,2048},"o_proj"); require_weight_1d(w.post_attention_layernorm,1024,"post_attention_layernorm");
  require_weight(w.gate_proj,{3072,1024},"gate_proj"); require_weight(w.up_proj,{3072,1024},"up_proj"); require_weight(w.down_proj,{1024,3072},"down_proj");
}
}

Qwen3LayerTrace qwen3_decoder_layer_trace(const Tensor& hidden_states, const std::vector<int32_t>& positions,
                                          const Qwen3LayerWeights& w, float eps, float theta) {
  validate(hidden_states, positions, w, eps, theta);
  Qwen3LayerTrace r;
  r.input_norm = rms_norm(hidden_states, w.input_layernorm, eps);
  r.q_linear = linear(r.input_norm, w.q_proj); r.k_linear = linear(r.input_norm, w.k_proj); r.v_linear = linear(r.input_norm, w.v_proj);
  auto q = r.q_linear.reshape({hidden_states.shape()[0],16,128}); auto k = r.k_linear.reshape({hidden_states.shape()[0],8,128});
  auto v = r.v_linear.reshape({hidden_states.shape()[0],8,128});
  r.q_normed = rms_norm(q.reshape({hidden_states.shape()[0]*16,128}), w.q_norm, eps).reshape(q.shape());
  r.k_normed = rms_norm(k.reshape({hidden_states.shape()[0]*8,128}), w.k_norm, eps).reshape(k.shape());
  r.q_rope = rope(r.q_normed, positions, theta); r.k_rope = rope(r.k_normed, positions, theta);
  r.attention_output = gqa_attention(r.q_rope, r.k_rope, v);
  r.o_proj_output = linear(r.attention_output.reshape({hidden_states.shape()[0],2048}), w.o_proj);
  r.attention_residual = add(hidden_states, r.o_proj_output);
  r.post_attention_norm = rms_norm(r.attention_residual, w.post_attention_layernorm, eps);
  r.gate_proj_output = linear(r.post_attention_norm, w.gate_proj); r.up_proj_output = linear(r.post_attention_norm, w.up_proj);
  r.swiglu_output = swiglu(r.gate_proj_output, r.up_proj_output); r.down_proj_output = linear(r.swiglu_output, w.down_proj);
  r.layer_output = add(r.attention_residual, r.down_proj_output);
  return r;
}
Tensor qwen3_decoder_layer(const Tensor& h, const std::vector<int32_t>& p, const Qwen3LayerWeights& w, float eps, float theta) {
  return qwen3_decoder_layer_trace(h,p,w,eps,theta).layer_output;
}
}
