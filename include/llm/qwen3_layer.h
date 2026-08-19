#pragma once
#include "tensor.h"
#include <cstdint>
#include <vector>

namespace llm {
struct Qwen3LayerWeights {
  Tensor input_layernorm, q_proj, k_proj, v_proj, q_norm, k_norm;
  Tensor o_proj, post_attention_layernorm, gate_proj, up_proj, down_proj;
};

struct Qwen3LayerTrace {
  Tensor input_norm, q_linear, k_linear, v_linear, q_normed, k_normed;
  Tensor q_rope, k_rope, attention_output, o_proj_output;
  Tensor attention_residual, post_attention_norm, gate_proj_output;
  Tensor up_proj_output, swiglu_output, down_proj_output, layer_output;
  // Explicit aliases used by the CUDA trace while retaining the Phase 1 names.
  Tensor q_norm_output, k_norm_output;
};

Tensor qwen3_decoder_layer(const Tensor& hidden_states,
                           const std::vector<int32_t>& position_ids,
                           const Qwen3LayerWeights& weights,
                           float rms_norm_eps, float rope_theta);

Qwen3LayerTrace qwen3_decoder_layer_trace(const Tensor& hidden_states,
                                          const std::vector<int32_t>& position_ids,
                                          const Qwen3LayerWeights& weights,
                                          float rms_norm_eps, float rope_theta);
}
