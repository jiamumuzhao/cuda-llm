#include "llm/cuda_check.h"
#include "llm/ops_cuda.h"
#include "llm/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace llm;

static Tensor make_input(const std::vector<int64_t>& shape, float base,
                         DType dtype) {
  Tensor x(dtype, shape);
  for (size_t i = 0; i < x.numel(); ++i) {
    const size_t d = i % 128;
    const size_t h = (i / 128) % static_cast<size_t>(shape[1]);
    const size_t t = i / (static_cast<size_t>(shape[1]) * 128);
    x.set_f32(i, base + 0.0007f * static_cast<float>(t) +
                    0.0011f * static_cast<float>(h) +
                    0.0003f * static_cast<float>(d));
  }
  return x.to(DeviceType::CUDA);
}

static void run_case(size_t seq_len, size_t batch, DType dtype,
                      const char* path, int warmup, int repeats) {
  constexpr size_t q_heads = 16;
  constexpr size_t kv_heads = 8;
  constexpr size_t head_dim = 128;
  const size_t total_tokens = seq_len * batch;
  Tensor q = make_input({static_cast<int64_t>(total_tokens), q_heads, head_dim},
                        0.03f, dtype);
  Tensor k = make_input({static_cast<int64_t>(total_tokens), kv_heads, head_dim},
                        0.07f, dtype);
  Tensor v = make_input({static_cast<int64_t>(total_tokens), kv_heads, head_dim},
                        0.11f, dtype);
  Tensor offsets(DType::I32, {static_cast<int64_t>(batch + 1)});
  offsets.data_i32()[0] = 0;
  for (size_t b = 0; b < batch; ++b)
    offsets.data_i32()[b + 1] = static_cast<int32_t>(seq_len * (b + 1));
  Tensor offsets_cuda = offsets.to(DeviceType::CUDA);
  Tensor output(dtype,
                {static_cast<int64_t>(total_tokens), q_heads, head_dim},
                DeviceType::CUDA);

  for (int i = 0; i < warmup; ++i)
    cuda_gqa_attention_packed_out(q, k, v, offsets_cuda, output);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeats; ++i)
    cuda_gqa_attention_packed_out(q, k, v, offsets_cuda, output);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  const double avg_ms = elapsed_ms / repeats;
  const double tokens_per_second =
      static_cast<double>(total_tokens) * 1000.0 / avg_ms;
  const double theoretical_flops =
      2.0 * static_cast<double>(batch) * seq_len * seq_len *
      q_heads * head_dim;
  const double tflops = theoretical_flops / (avg_ms * 1.0e9);
  std::cout << path
            << " T=" << seq_len
            << " B=" << batch
            << " dtype=" << dtype_name(dtype)
            << " avg_ms=" << std::fixed << std::setprecision(3) << avg_ms
            << " tokens_per_sec=" << std::setprecision(1) << tokens_per_second
            << " qk_tflops=" << std::setprecision(4) << tflops << "\n";
}

int main() {
  int devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&devices));
  if (devices == 0) {
    std::cout << "SKIP benchmark_packed_attention: no CUDA device\n";
    return 0;
  }
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "device=" << prop.name
            << " compute_capability=" << prop.major << "." << prop.minor << "\n";
  for (size_t seq_len : {size_t(512), size_t(1024)}) {
    const int repeats = seq_len == 1024 ? 5 : 10;
    run_case(seq_len, 1, DType::F32, "packed_attention_cuda_core",
             3, repeats);
    run_case(seq_len, 1, DType::F16, "packed_attention_tensor_core",
             3, repeats);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
