#include "llm/cuda_check.h"
#include "llm/ops_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& message) { throw std::runtime_error("test_argmax_cuda: " + message); }

static Tensor f16(const Tensor& input) {
  Tensor result(DType::F16, input.shape());
  for (size_t i = 0; i < input.numel(); ++i) result.set_f32(i, input.get_f32(i));
  return result;
}

static void expect_result(const std::string& name, const Tensor& logits, int32_t expected) {
  int32_t actual = cuda_argmax_last_row(logits.to(DeviceType::CUDA));
  std::cout << name << " device=CUDA dtype=" << dtype_name(logits.dtype())
            << " argmax=" << actual << " expected=" << expected << "\n";
  if (actual != expected) fail(name + " actual=" + std::to_string(actual) +
                                " expected=" + std::to_string(expected));
}

static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); }
  catch (const std::exception& error) { std::cout << "rejected " << name << ": " << error.what() << "\n"; return; }
  fail(name + " was accepted");
}

int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability=" << prop.major << "." << prop.minor << "\n";
    expect_result("f32_normal", Tensor::from_f32({2, 5}, {-1, 2, 3, 4, 5, 1, 9, 2, 8, 3}), 1);
    expect_result("f16_normal", f16(Tensor::from_f32({2, 5}, {-1, 2, 3, 4, 5, 1, 9, 2, 8, 3})), 1);
    expect_result("f32_tie_min_id", Tensor::from_f32({1, 6}, {4, 7, 7, 1, 7, 2}), 1);
    expect_result("f16_negative", f16(Tensor::from_f32({1, 4}, {-1, -2, -3, -4})), 0);
    std::vector<float> large(151936, -10.0f); large[11] = 100.0f; large[151935] = 101.0f;
    std::vector<float> two_rows(2 * 151936, -10.0f);
    two_rows[11] = 100.0f; two_rows[151936 + 12345] = 101.0f;
    expect_result("f32_large_vocab", Tensor::from_f32({2, 151936}, two_rows), 12345);
    Tensor large_last = Tensor::from_f32({1, 151936}, large);
    large_last.set_f32(12345, 101.0f); large_last.set_f32(151935, 101.0f);
    expect_result("f16_large_vocab_tie", f16(large_last), 12345);
    expect_throw("CPU input", [&] { cuda_argmax_last_row(Tensor::from_f32({1, 2}, {1, 2})); });
    expect_throw("rank mismatch", [&] { cuda_argmax_last_row(Tensor(DType::F32, {2, 2, 1}, DeviceType::CUDA)); });
    expect_throw("BF16 input", [&] { cuda_argmax_last_row(Tensor(DType::BF16, {1, 2}, DeviceType::CUDA)); });
    expect_throw("zero shape", [&] { cuda_argmax_last_row(Tensor(DType::F32, {0, 2}, DeviceType::CUDA)); });
    expect_throw("empty shape", [&] { cuda_argmax_last_row(Tensor(DType::F32, {}, DeviceType::CUDA)); });
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_argmax_cuda passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << "\n"; return 1; }
}
