#include "llm/cuda_check.h"
#include "llm/ops_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
static void fail(const std::string& s) { throw std::runtime_error("test_gqa_valid_lengths_cuda: " + s); }
static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); } catch (const std::exception& e) { std::cout << "rejected " << name << ": " << e.what() << "\n"; return; }
  fail(name + " was accepted");
}

static Tensor lengths_cuda(const std::vector<int32_t>& values) {
  Tensor host(DType::I32, {static_cast<int64_t>(values.size())});
  std::copy(values.begin(), values.end(), host.data_i32());
  return host.to(DeviceType::CUDA);
}

static void run_case(DType dtype) {
  constexpr size_t B = 2, S = 4, QH = 16, KH = 8, D = 128;
  Tensor q_host(dtype, {static_cast<int64_t>(B), static_cast<int64_t>(S), static_cast<int64_t>(QH), static_cast<int64_t>(D)});
  Tensor k_host(dtype, {static_cast<int64_t>(B), static_cast<int64_t>(S), static_cast<int64_t>(KH), static_cast<int64_t>(D)});
  Tensor v_host(dtype, {static_cast<int64_t>(B), static_cast<int64_t>(S), static_cast<int64_t>(KH), static_cast<int64_t>(D)});
  for (size_t i = 0; i < q_host.numel(); ++i) q_host.set_f32(i, 0.0f);
  for (size_t i = 0; i < k_host.numel(); ++i) k_host.set_f32(i, 0.0f);
  for (size_t b = 0; b < B; ++b)
    for (size_t s = 0; s < S; ++s)
      for (size_t h = 0; h < KH; ++h)
        for (size_t d = 0; d < D; ++d)
          v_host.set_f32(((b * S + s) * KH + h) * D + d,
                         float(100 * b + 10 * s + d) / 100.0f);
  Tensor q = q_host.to(DeviceType::CUDA), k = k_host.to(DeviceType::CUDA), v = v_host.to(DeviceType::CUDA);
  Tensor lengths = lengths_cuda({2, 4});
  Tensor masked = cuda_gqa_attention_batched_valid_lengths(q, k, v, lengths);
  Tensor causal = cuda_gqa_attention_batched(q, k, v);
  Tensor masked_cpu = masked.to(DeviceType::CPU), causal_cpu = causal.to(DeviceType::CPU);
  for (size_t b = 0; b < B; ++b)
    for (size_t s = 0; s < S; ++s)
      for (size_t h = 0; h < QH; ++h)
        for (size_t d = 0; d < D; ++d) {
          const size_t index = ((b * S + s) * QH + h) * D + d;
          const float actual = masked_cpu.get_f32(index);
          const float expected = s < (b == 0 ? 2 : 4) ? causal_cpu.get_f32(index) : 0.0f;
          if (std::fabs(actual - expected) > (dtype == DType::F16 ? 5e-3f : 1e-5f))
            fail("masked output mismatch dtype=" + std::string(dtype_name(dtype)) + " index=" + std::to_string(index));
        }

  // Row 0's invalid tail is never read; row 1 remains isolated.
  v_host.set_f32(((0 * S + 2) * KH + 0) * D, 999.0f);
  v_host.set_f32(((0 * S + 3) * KH + 0) * D, -999.0f);
  Tensor changed_v = v_host.to(DeviceType::CUDA);
  Tensor changed = cuda_gqa_attention_batched_valid_lengths(q, k, changed_v, lengths).to(DeviceType::CPU);
  for (size_t s = 0; s < S; ++s)
    for (size_t h = 0; h < QH; ++h)
      for (size_t d = 0; d < D; ++d) {
        const size_t row0 = ((0 * S + s) * QH + h) * D + d;
        const size_t row1 = ((1 * S + s) * QH + h) * D + d;
        if (s < 2 && std::fabs(changed.get_f32(row0) - masked_cpu.get_f32(row0)) > (dtype == DType::F16 ? 5e-3f : 1e-5f)) fail("row 0 tail affected valid query");
        if (std::fabs(changed.get_f32(row1) - masked_cpu.get_f32(row1)) > (dtype == DType::F16 ? 5e-3f : 1e-5f)) fail("row isolation failed");
        if (s >= 2 && changed.get_f32(row0) != 0.0f) fail("invalid query output is not zero");
      }
  std::cout << "masked GQA dtype=" << dtype_name(dtype) << " B=2 valid_lengths=[2,4] query/key mask, zero invalid query, isolation passed\n";

  expect_throw("valid length zero", [&] { cuda_gqa_attention_batched_valid_lengths(q, k, v, lengths_cuda({0, 4})); });
  expect_throw("valid length exceeds S", [&] { cuda_gqa_attention_batched_valid_lengths(q, k, v, lengths_cuda({2, 5})); });
  expect_throw("valid length batch mismatch", [&] { cuda_gqa_attention_batched_valid_lengths(q, k, v, lengths_cuda({2})); });
  expect_throw("valid length CPU metadata", [&] { Tensor h(DType::I32, {2}); cuda_gqa_attention_batched_valid_lengths(q, k, v, h); });
  expect_throw("valid length wrong dtype", [&] { cuda_gqa_attention_batched_valid_lengths(q, k, v, Tensor(DType::F32, {2}, DeviceType::CUDA)); });
}

int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability=" << prop.major << "." << prop.minor << "\n";
    run_case(DType::F32); run_case(DType::F16);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_gqa_valid_lengths_cuda passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
