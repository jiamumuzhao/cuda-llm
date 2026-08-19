#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
namespace llm {
enum class DType { F32, F16, BF16 };
inline size_t dtype_size(DType d) { return d == DType::F32 ? 4 : 2; }
inline const char* dtype_name(DType d) { return d == DType::F32 ? "F32" : d == DType::F16 ? "F16" : "BF16"; }
float decode_f16(uint16_t); uint16_t encode_f16(float);
float decode_bf16(uint16_t); uint16_t encode_bf16(float);
}
