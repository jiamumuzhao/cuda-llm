#include "llm/qwen3_cuda_model.h"
#include "llm/cuda_check.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

using namespace llm;

int main(int argc, char** argv) {
  const std::size_t iterations = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 20;
  const std::size_t context = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 32;
  const bool flash = argc > 4 && std::string(argv[4]) == "flash";
  const bool graph = argc > 4 && std::string(argv[4]) == "graph";
  const bool graph_flash = argc > 4 && std::string(argv[4]) == "graph_flash";
  const bool compare = argc > 4 && std::string(argv[4]) == "compare";
  const bool batch_graph = argc > 4 && std::string(argv[4]) == "batch_graph";
  const bool batch_graph_flash = argc > 4 && std::string(argv[4]) == "batch_graph_flash";
  const bool batch_eager = argc > 4 && std::string(argv[4]) == "batch_eager";
  const bool batch_compare = argc > 4 && std::string(argv[4]) == "batch_compare";
  const std::string package = argc > 1 ? argv[1] : "artifacts/phase15/qwen3-0.6b-f32";
  if (iterations == 0 || context == 0 || context + iterations > 512) {
    std::cerr << "usage: benchmark_paged_decode_e2e [package] [iterations] [context] [flash]\n";
    return 2;
  }
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "SKIP benchmark_paged_decode_e2e: no CUDA device\n";
    return 0;
  }

  Qwen3CudaModel model(package);
  if (compare) {
    PagedKvCachePool eager_pool({32, 28, 8, 16, 128, DType::F16});
    PagedKvCachePool graph_pool({32, 28, 8, 16, 128, DType::F16});
    PagedKvCachePool flash_pool({32, 28, 8, 16, 128, DType::F16});
    Qwen3PagedKvCache eager_cache(eager_pool, 512);
    Qwen3PagedKvCache graph_cache(graph_pool, 512);
    Qwen3PagedKvCache flash_cache(flash_pool, 512);
    std::vector<int32_t> prompt(context);
    for (std::size_t i = 0; i < context; ++i)
      prompt[i] = static_cast<int32_t>((17 + i * 37) % 150000);
    model.prefill_logits_paged(prompt, eager_cache);
    model.prefill_logits_paged(prompt, graph_cache);
    model.prefill_logits_paged(prompt, flash_cache);
    Tensor eager_logits(DType::F16,
                        {1, static_cast<int64_t>(model.model_spec().vocab_size)},
                        DeviceType::CUDA);
    int32_t token = 700;
    for (std::size_t step = 0; step < iterations; ++step) {
      model.decode_logits_paged_into(token, eager_cache, eager_logits);
      Tensor graph_logits = model.decode_logits_paged_graph(token, graph_cache);
      Tensor flash_logits = model.decode_logits_paged_flash_graph(token, flash_cache);
      const int32_t eager_token = cuda_argmax_last_row(eager_logits);
      const int32_t graph_token = cuda_argmax_last_row(graph_logits);
      const int32_t flash_token = cuda_argmax_last_row(flash_logits);
      std::cout << "compare_step=" << step
                << " eager_token=" << eager_token
                << " graph_token=" << graph_token
                << " flash_graph_token=" << flash_token << "\n";
      if (eager_token != graph_token || eager_token != flash_token) {
        std::cerr << "multi-step token comparison failed at step=" << step << "\n";
        return 1;
      }
      token = eager_token;
    }
    std::cout << "eager/reference-graph/flash-graph token equivalence passed\n";
    return 0;
  }
  if (batch_compare) {
    const std::size_t batch = argc > 5 ? std::strtoul(argv[5], nullptr, 10) : 2;
    if (batch != 2 && batch != 4) { std::cerr << "batch_compare requires batch=2 or 4\n"; return 2; }
    PagedKvCachePool eager_pool({256, 28, 8, 16, 128, DType::F16});
    PagedKvCachePool graph_pool({256, 28, 8, 16, 128, DType::F16});
    PagedKvCachePool flash_pool({256, 28, 8, 16, 128, DType::F16});
    std::vector<std::unique_ptr<Qwen3PagedKvCache>> eager, graph_c, flash_c;
    std::vector<Qwen3PagedKvCache*> ep, gp, fp;
    std::vector<int32_t> ids(batch, 700);
    for (std::size_t b = 0; b < batch; ++b) {
      const std::size_t row_context = context + b * 3;
      std::vector<int32_t> prompt(row_context);
      for (std::size_t i = 0; i < row_context; ++i)
        prompt[i] = static_cast<int32_t>((17 + b * 11 + i * 37) % 150000);
      eager.push_back(std::make_unique<Qwen3PagedKvCache>(eager_pool, 512));
      graph_c.push_back(std::make_unique<Qwen3PagedKvCache>(graph_pool, 512));
      flash_c.push_back(std::make_unique<Qwen3PagedKvCache>(flash_pool, 512));
      ep.push_back(eager.back().get()); gp.push_back(graph_c.back().get()); fp.push_back(flash_c.back().get());
      model.prefill_logits_paged(prompt, *ep.back());
      model.prefill_logits_paged(prompt, *gp.back());
      model.prefill_logits_paged(prompt, *fp.back());
    }
    auto row_argmax = [](const Tensor& x, std::size_t row) {
      Tensor cpu = x.to(DeviceType::CPU);
      const std::size_t vocab = static_cast<std::size_t>(x.shape()[1]);
      int32_t best = 0; float value = cpu.get_f32(row * vocab);
      for (std::size_t i = 1; i < vocab; ++i) { const float v = cpu.get_f32(row * vocab + i); if (v > value) { value = v; best = static_cast<int32_t>(i); } }
      return best;
    };
    for (std::size_t step = 0; step < iterations; ++step) {
      for (std::size_t b = 0; b < batch; ++b) ids[b] = 700 + static_cast<int32_t>(b * 17 + step);
      Tensor eager_logits = model.decode_logits_paged_batch(ids, ep);
      Tensor graph_logits = model.decode_logits_paged_batch_graph(ids, gp);
      Tensor flash_logits = model.decode_logits_paged_flash_batch_graph(ids, fp);
      for (std::size_t b = 0; b < batch; ++b) {
        const int32_t e = row_argmax(eager_logits, b), g = row_argmax(graph_logits, b), f = row_argmax(flash_logits, b);
        std::cout << "batch_compare_step=" << step << " row=" << b << " eager_token=" << e << " graph_token=" << g << " flash_graph_token=" << f << "\n";
        if (e != g || e != f) { std::cerr << "batch graph token mismatch at step=" << step << " row=" << b << "\n"; return 1; }
      }
    }
    std::cout << "batch=" << batch << " mixed-length eager/reference-graph/flash-graph token equivalence passed\n";
    return 0;
  }
  if (batch_eager || batch_graph || batch_graph_flash) {
    const std::size_t batch = argc > 5 ? std::strtoul(argv[5], nullptr, 10) : 2;
    if (batch != 2 && batch != 4) {
      std::cerr << "batch_graph requires batch=2 or 4\n";
      return 2;
    }
    PagedKvCachePool pool({128, 28, 8, 16, 128, DType::F16});
    std::vector<std::unique_ptr<Qwen3PagedKvCache>> owned;
    std::vector<Qwen3PagedKvCache*> caches;
    owned.reserve(batch); caches.reserve(batch);
    std::vector<int32_t> prompt(context);
    for (std::size_t i = 0; i < context; ++i)
      prompt[i] = static_cast<int32_t>((17 + i * 37) % 150000);
    for (std::size_t b = 0; b < batch; ++b) {
      owned.push_back(std::make_unique<Qwen3PagedKvCache>(pool, 512));
      caches.push_back(owned.back().get());
      model.prefill_logits_paged(prompt, *caches.back());
    }
    std::vector<int32_t> tokens(batch, 700);
    const std::size_t warmup = std::min<std::size_t>(3, 512 - context - iterations);
    for (std::size_t i = 0; i < warmup; ++i) {
      for (std::size_t b = 0; b < batch; ++b) tokens[b] = 700 + static_cast<int32_t>(i);
      if (batch_graph_flash) model.decode_logits_paged_flash_batch_graph(tokens, caches);
      else if (batch_graph) model.decode_logits_paged_batch_graph(tokens, caches);
      else model.decode_logits_paged_batch(tokens, caches);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    reset_cuda_paged_gqa_attention_decode_batch_launch_count();
    reset_cuda_allocation_stats();
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
      for (std::size_t b = 0; b < batch; ++b) tokens[b] = 900 + static_cast<int32_t>(i);
      if (batch_graph_flash) model.decode_logits_paged_flash_batch_graph(tokens, caches);
      else if (batch_graph) model.decode_logits_paged_batch_graph(tokens, caches);
      else model.decode_logits_paged_batch(tokens, caches);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    const auto stats = cuda_allocation_stats();
    const double avg_ms = elapsed_ms / static_cast<double>(iterations);
    std::cout << "batch=" << batch << " iterations=" << iterations
              << " context_start=" << context + warmup
              << " avg_ms=" << avg_ms
              << " tokens_per_sec=" << (1000.0 * batch / avg_ms)
              << " steady_state_cuda_malloc_calls=" << stats.cuda_malloc_calls
              << " cuda_free_calls=" << stats.cuda_free_calls
              << " allocated_bytes=" << stats.cuda_allocated_bytes_total
              << " freed_bytes=" << stats.cuda_freed_bytes_total
              << " batch_attention_launches=" << cuda_paged_gqa_attention_decode_batch_launch_count()
              << " attention_kernel=" << (batch_graph_flash ? "flash_online" : "reference")
              << " execution=" << (batch_eager ? "eager" : "cuda_graph") << "\n";
    return 0;
  }
  Tensor logits(DType::F16,
                {1, static_cast<int64_t>(model.model_spec().vocab_size)},
                DeviceType::CUDA);
  PagedKvCachePool pool({64, 28, 8, 16, 128, DType::F16});
  Qwen3PagedKvCache cache(pool, 512);
  std::vector<int32_t> prompt(context);
  for (std::size_t i = 0; i < context; ++i)
    prompt[i] = static_cast<int32_t>((17 + i * 37) % 150000);
  model.prefill_logits_paged(prompt, cache);
  const std::size_t warmup = std::min<std::size_t>(3, 512 - context - iterations);
  for (std::size_t i = 0; i < warmup; ++i) {
    if (graph_flash) model.decode_logits_paged_flash_graph(700 + i, cache);
    else if (graph) model.decode_logits_paged_graph(700 + i, cache);
    else if (flash) model.decode_logits_paged_flash_into(700 + i, cache, logits);
    else model.decode_logits_paged_into(700 + i, cache, logits);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  reset_cuda_paged_gqa_attention_decode_batch_launch_count();
  reset_cuda_allocation_stats();
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    if (graph_flash) model.decode_logits_paged_flash_graph(900 + static_cast<int32_t>(i), cache);
    else if (graph) model.decode_logits_paged_graph(900 + static_cast<int32_t>(i), cache);
    else if (flash) model.decode_logits_paged_flash_into(900 + static_cast<int32_t>(i), cache, logits);
    else model.decode_logits_paged_into(900 + static_cast<int32_t>(i), cache, logits);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  const auto stats = cuda_allocation_stats();
  const double avg_ms = elapsed_ms / static_cast<double>(iterations);
  std::cout << "iterations=" << iterations
            << " context_start=" << context + 3
            << " avg_ms=" << avg_ms
            << " tokens_per_sec=" << (1000.0 / avg_ms)
            << " steady_state_cuda_malloc_calls=" << stats.cuda_malloc_calls
            << " cuda_free_calls=" << stats.cuda_free_calls
            << " allocated_bytes=" << stats.cuda_allocated_bytes_total
            << " freed_bytes=" << stats.cuda_freed_bytes_total
            << " batch_attention_launches=" << cuda_paged_gqa_attention_decode_batch_launch_count()
            << " metadata_workspace_bytes=" << model.paged_decode_metadata_workspace_bytes()
            << " attention_kernel=" << ((flash || graph_flash) ? "flash_online" : "reference")
            << " execution=" << ((graph || graph_flash) ? "cuda_graph" : "eager")
            << "\n";
  cache.release_all();
  return 0;
}
