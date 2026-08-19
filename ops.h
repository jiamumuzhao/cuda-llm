#pragma once
#include "tensor.h"
#include <vector>
#include <cstddef>
#include <cstdint>
namespace llm {
Tensor embedding_lookup(const Tensor&, const std::vector<int32_t>&);
Tensor linear(const Tensor&, const Tensor&); 
Tensor rms_norm(const Tensor&, const Tensor&, float eps);
Tensor rope(const Tensor&, const std::vector<int32_t>&, float theta);
Tensor causal_mask(size_t); 
Tensor softmax_last_dim(const Tensor&);
Tensor gqa_attention(const Tensor&, const Tensor&, const Tensor&); 
Tensor swiglu(const Tensor&, const Tensor&);
Tensor add(const Tensor&, const Tensor&); 
Tensor lm_head(const Tensor&, const Tensor&);
}
