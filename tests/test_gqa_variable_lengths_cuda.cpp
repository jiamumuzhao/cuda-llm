#include "llm/cuda_check.h"
#include "llm/ops_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& message) {
  throw std::runtime_error("test_gqa_variable_lengths_cuda: " + message);
}
static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); }
  catch (const std::exception& e) { std::cout << "rejected " << name << ": " << e.what() << "\n"; return; }
  fail(name + " was accepted");
}
static Tensor f16_tensor(const std::vector<int64_t>& shape, float value) {
  Tensor t(DType::F16, shape, DeviceType::CPU);
  for (size_t i = 0; i < t.numel(); ++i) t.set_f32(i, value);
  return t.to(DeviceType::CUDA);
}
static Tensor lengths_tensor(const std::vector<int32_t>& values) {
  Tensor t(DType::I32, {static_cast<int64_t>(values.size())}, DeviceType::CPU);
  for (size_t i = 0; i < values.size(); ++i) t.data_i32()[i] = values[i];
  return t.to(DeviceType::CUDA);
}

int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability="
              << prop.major << "." << prop.minor << "\n";
    Tensor q = f16_tensor({2, 16, 128}, 0.0f);
    Tensor k0 = f16_tensor({4, 8, 128}, 0.0f);
    Tensor k1 = f16_tensor({4, 8, 128}, 0.0f);
    Tensor v0 = f16_tensor({4, 8, 128}, 0.0f);
    Tensor v1 = f16_tensor({4, 8, 128}, 0.0f);
    Tensor v0_cpu(DType::F16, {4, 8, 128}, DeviceType::CPU);
    Tensor v1_cpu(DType::F16, {4, 8, 128}, DeviceType::CPU);
    for (size_t i = 0; i < v0_cpu.numel(); ++i) {
      const float row = static_cast<float>(i / (8 * 128));
      v0_cpu.set_f32(i, row + 1.0f); v1_cpu.set_f32(i, row + 10.0f);
    }
    v0 = v0_cpu.to(DeviceType::CUDA); v1 = v1_cpu.to(DeviceType::CUDA);
    std::vector<const Tensor*> keys{&k0, &k1};
    std::vector<const Tensor*> values{&v0, &v1};
    Tensor lengths = lengths_tensor({2, 4});
    Tensor actual = cuda_gqa_decode_attention_batched_variable_lengths(
        q, keys, values, lengths).to(DeviceType::CPU);
    auto changed_tail = v0_cpu;
    for (size_t i = 2 * 8 * 128; i < changed_tail.numel(); ++i) changed_tail.set_f32(i, 999.0f);
    Tensor changed = changed_tail.to(DeviceType::CUDA);
    std::vector<const Tensor*> changed_values{&changed, &v1};
    Tensor after = cuda_gqa_decode_attention_batched_variable_lengths(
        q, keys, changed_values, lengths).to(DeviceType::CPU);
    for (size_t i = 0; i < 16 * 128; ++i)
      if (actual.get_f32(i) != after.get_f32(i)) fail("row 0 read invalid cache tail");
    for (size_t i = 16 * 128; i < actual.numel(); ++i)
      if (actual.get_f32(i) != after.get_f32(i)) fail("row 1 changed when row 0 changed");
    std::cout << "variable GQA lengths=[2,4] tail masking and row isolation passed\n";

    expect_throw("length zero", [&] { cuda_gqa_decode_attention_batched_variable_lengths(
        q, keys, values, lengths_tensor({0, 4})); });
    expect_throw("length exceeds capacity", [&] { cuda_gqa_decode_attention_batched_variable_lengths(
        q, keys, values, lengths_tensor({2, 5})); });
    expect_throw("batch mismatch", [&] { cuda_gqa_decode_attention_batched_variable_lengths(
        q, keys, values, lengths_tensor({2})); });
    Tensor cpu_q(DType::F16, {2, 16, 128}, DeviceType::CPU);
    expect_throw("CPU q", [&] { cuda_gqa_decode_attention_batched_variable_lengths(
        cpu_q, keys, values, lengths); });
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_gqa_variable_lengths_cuda passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
