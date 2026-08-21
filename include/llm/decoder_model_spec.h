#pragma once

#include "attention_config.h"

#include <cstddef>

namespace llm {

// Architecture description consumed by model-independent decoder runtime
// components. A model adapter populates this structure from its package.
struct DecoderModelSpec {
  AttentionConfig attention;
  std::size_t hidden_size = 0;
  std::size_t intermediate_size = 0;
  std::size_t vocab_size = 0;
  std::size_t num_layers = 0;
  float rms_norm_eps = 0.0f;
  float rope_theta = 0.0f;
  bool tie_word_embeddings = false;
};

}  // namespace llm
