#pragma once

#include "tensor.h"
#include "decoder_model_spec.h"

namespace llm {

struct DecoderLayerWeights {
  Tensor input_norm;
  Tensor q_proj;
  Tensor k_proj;
  Tensor v_proj;
  Tensor q_norm;
  Tensor k_norm;
  Tensor o_proj;
  Tensor post_attention_norm;
  Tensor gate_proj;
  Tensor up_proj;
  Tensor down_proj;
  // Runtime shape/configuration carried with the adapter-normalized weights.
  // Defaults preserve the current Qwen3 test fixtures.
  DecoderModelSpec model_spec{
      AttentionConfig{16, 8, 128, 16, 512, true},
      1024, 3072, 151936, 28, 1.0e-6f, 1.0e6f, true};
};

}  // namespace llm
