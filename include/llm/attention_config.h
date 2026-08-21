#pragma once
#include <cstddef>

namespace llm {

// Model-independent attention metadata shared by decoder-only LLM kernels.
// num_q_heads may be larger than num_kv_heads for GQA.
struct AttentionConfig {
  std::size_t num_q_heads = 0;
  std::size_t num_kv_heads = 0;
  std::size_t head_dim = 0;
  std::size_t page_size = 0;
  std::size_t max_seq_len = 0;
  bool causal = true;

  bool is_gqa() const noexcept {
    return num_kv_heads != 0 && num_q_heads != 0 &&
           num_q_heads % num_kv_heads == 0;
  }
};

}  // namespace llm
