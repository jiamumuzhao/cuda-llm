#include "llm/cuda_check.h"
#include "llm/ops_cuda.h"
#include "llm/paged_kv_cache_pool.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using namespace llm;

int main(int argc, char** argv) {
  const int iterations = argc > 1 ? std::atoi(argv[1]) : 1000;
  const std::size_t sequence_length =
      argc > 2 ? static_cast<std::size_t>(std::atoi(argv[2])) : 256;
  constexpr std::size_t batch = 4;
  constexpr std::size_t q_heads = 16;
  constexpr std::size_t kv_heads = 8;
  constexpr std::size_t head_dim = 128;
  constexpr std::size_t block_size = 16;
  const std::size_t max_blocks =
      (sequence_length + block_size - 1) / block_size;

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "SKIP benchmark_paged_decode_kernel: no CUDA device\n";
    return 0;
  }

  PagedKvCachePoolConfig pool_config;
  pool_config.total_blocks = 128;
  pool_config.num_layers = 1;
  pool_config.num_kv_heads = kv_heads;
  pool_config.block_size = block_size;
  pool_config.head_dim = head_dim;
  pool_config.dtype = DType::F16;
  PagedKvCachePool pool(pool_config);

  std::vector<std::unique_ptr<PagedSequenceKvCache>> sequences;
  sequences.reserve(batch);
  Tensor q_host(DType::F16, {static_cast<int64_t>(batch),
                             static_cast<int64_t>(q_heads),
                             static_cast<int64_t>(head_dim)});
  Tensor tables_host(DType::I32, {static_cast<int64_t>(batch),
                                  static_cast<int64_t>(max_blocks)});
  Tensor positions_host(DType::I32, {static_cast<int64_t>(batch)});

  for (std::size_t b = 0; b < batch; ++b) {
    sequences.push_back(
        std::make_unique<PagedSequenceKvCache>(pool, sequence_length));
    sequences.back()->append_tokens(sequence_length);
    positions_host.data_i32()[b] =
        static_cast<int32_t>(sequence_length - 1);
    for (std::size_t i = 0; i < q_heads * head_dim; ++i)
      q_host.set_f32(b * q_heads * head_dim + i,
                     0.01f * static_cast<float>((i + 13 * b) % 97));
    for (std::size_t i = 0; i < max_blocks; ++i)
      tables_host.data_i32()[b * max_blocks + i] =
          static_cast<int32_t>(sequences.back()->block_table()[i]);

    Tensor key_token(DType::F16,
                     {static_cast<int64_t>(kv_heads),
                      static_cast<int64_t>(head_dim)});
    Tensor value_token(DType::F16, key_token.shape());
    for (std::size_t t = 0; t < sequence_length; ++t) {
      for (std::size_t i = 0; i < kv_heads * head_dim; ++i) {
        key_token.set_f32(i, 0.001f * static_cast<float>((i + t) % 251));
        value_token.set_f32(i, 0.001f * static_cast<float>((3 * i + t) % 251));
      }
      sequences.back()->copy_token_from_host(
          0, t, PagedKvKind::Key, key_token.data_f16(), kv_heads * head_dim);
      sequences.back()->copy_token_from_host(
          0, t, PagedKvKind::Value, value_token.data_f16(),
          kv_heads * head_dim);
    }
  }

  Tensor q_cuda = q_host.to(DeviceType::CUDA);
  Tensor tables_cuda = tables_host.to(DeviceType::CUDA);
  Tensor positions_cuda = positions_host.to(DeviceType::CUDA);
  Tensor output(DType::F16,
                {static_cast<int64_t>(batch),
                 static_cast<int64_t>(q_heads),
                 static_cast<int64_t>(head_dim)},
                DeviceType::CUDA);

  for (int i = 0; i < 20; ++i)
    cuda_paged_gqa_attention_decode_batch_out(
        q_cuda, pool, 0, tables_cuda, positions_cuda, q_heads, kv_heads,
        head_dim, output);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i)
    cuda_paged_gqa_attention_decode_batch_out(
        q_cuda, pool, 0, tables_cuda, positions_cuda, q_heads, kv_heads,
        head_dim, output);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  const double kernel_ms = elapsed_ms / iterations;
  const double tokens_per_sec =
      1000.0 * static_cast<double>(batch) / kernel_ms;
  std::cout << "iterations=" << iterations
            << " batch=" << batch
            << " kv_len=" << sequence_length
            << " avg_ms=" << kernel_ms
            << " tokens_per_sec=" << tokens_per_sec
            << " output_workspace=reused\n";
  return 0;
}
