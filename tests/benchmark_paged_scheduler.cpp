#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_paged_scheduler.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using namespace llm;

int main(int argc, char** argv) {
  const std::string package =
      argc > 1 ? argv[1] : "artifacts/phase15/qwen3-0.6b-f32";
  Qwen3CudaModel model(package);
  PagedKvCachePool pool({128, 28, 8, 16, 128, DType::F16});

  PagedSchedulerConfig config;
  config.max_seq_len = 512;
  config.max_active_requests = 4;
  config.max_prefill_batch_size = 4;
  config.max_decode_batch_size = 4;
  config.prefer_packed_prefill = true;
  Qwen3PagedRequestScheduler scheduler(model, pool, config);

  const std::vector<std::size_t> lengths = {32, 64, 128, 256};
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    InferenceRequestConfig request;
    request.request_id = i + 1;
    request.prompt_ids.resize(lengths[i]);
    for (std::size_t j = 0; j < lengths[i]; ++j)
      request.prompt_ids[j] = static_cast<int32_t>((j * 17 + i * 31) % 32000);
    request.max_new_tokens = 8;
    request.max_seq_len = 512;
    request.sampling.temperature = 0.0f;
    scheduler.submit(std::move(request));
  }

  cudaDeviceSynchronize();
  const auto started = std::chrono::steady_clock::now();
  while (scheduler.has_unfinished()) {
    if (!scheduler.step()) {
      std::cerr << "scheduler made no progress\n";
      return 2;
    }
  }
  cudaDeviceSynchronize();
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();

  std::size_t prefill_tokens = 0;
  std::size_t decode_tokens = 0;
  std::size_t decode_batches = 0;
  std::size_t max_decode_batch = 0;
  for (const auto& metric : scheduler.metrics()) {
    prefill_tokens += metric.valid_prefill_tokens;
    decode_tokens += metric.decode_tokens;
    if (metric.action == "decode") {
      ++decode_batches;
      max_decode_batch = std::max(max_decode_batch, metric.decode_batch_size);
    }
  }
  const auto finished = scheduler.take_finished();
  std::cout << "requests=" << finished.size()
            << " prompt_tokens=" << prefill_tokens
            << " decode_tokens=" << decode_tokens
            << " decode_batches=" << decode_batches
            << " max_decode_batch=" << max_decode_batch
            << " elapsed_ms=" << elapsed_ms
            << " total_tokens_per_sec="
            << (1000.0 * (prefill_tokens + decode_tokens) / elapsed_ms)
            << " pool_used_blocks=" << pool.used_block_count() << "\n";
  return finished.size() == lengths.size() && pool.used_block_count() == 0 ? 0 : 3;
}
