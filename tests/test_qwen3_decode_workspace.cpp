#include "llm/cuda_check.h"
#include "llm/ops_cuda.h"
#include "llm/qwen3_cuda_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& message) {
  throw std::runtime_error("test_qwen3_decode_workspace: " + message);
}
static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); } catch (const std::exception& error) {
    std::cout << "rejected " << name << ": " << error.what() << "\n";
    return;
  }
  fail(name + " was accepted");
}
static std::vector<int32_t> prompt(size_t b, size_t n) {
  std::vector<int32_t> ids(n);
  for (size_t i = 0; i < n; ++i) ids[i] = static_cast<int32_t>(11 + b * 97 + i * 31);
  return ids;
}
static std::vector<float> row_cpu(const Tensor& t, size_t row) {
  Tensor cpu = t.to(DeviceType::CPU);
  const size_t width = static_cast<size_t>(cpu.shape().back());
  std::vector<float> result(width);
  for (size_t i = 0; i < width; ++i) result[i] = cpu.get_f32(row * width + i);
  return result;
}
static std::vector<int32_t> top5(const std::vector<float>& row) {
  std::vector<int32_t> ids(row.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + 5, ids.end(),
                    [&](int32_t a, int32_t b) {
                      return row[a] != row[b] ? row[a] > row[b] : a < b;
                    });
  ids.resize(5);
  return ids;
}
static void compare(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    const std::string& name) {
  float max_abs = 0.0f;
  double dot = 0.0, aa = 0.0, ee = 0.0;
  for (size_t i = 0; i < actual.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(actual[i] - expected[i]));
    dot += double(actual[i]) * expected[i];
    aa += double(actual[i]) * actual[i];
    ee += double(expected[i]) * expected[i];
  }
  const double cosine = dot / std::sqrt(aa * ee);
  std::cout << name << " max_abs_error=" << max_abs
            << " cosine_similarity=" << cosine << " tolerance=1e-4\n";
  if (max_abs > 1e-4f || cosine < 0.99999) fail(name + " numeric mismatch");
}

int main() {
  try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability="
              << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    if (model.decode_workspace_bytes() == 0) fail("workspace is not resident");
    SamplingConfig sampled;
    sampled.temperature = 0.8f;
    sampled.top_k = 50;
    sampled.top_p = 0.9f;
    sampled.seed = 20260727;

    for (size_t batch : {size_t(1), size_t(2), size_t(4)}) {
      constexpr size_t S = 4;
      std::vector<std::vector<int32_t>> prompts;
      std::vector<std::vector<int32_t>> one_prompt;
      std::vector<Qwen3KvCache> caches, one_cache;
      std::vector<Qwen3KvCache*> cache_ptrs, one_ptrs;
      prompts.reserve(batch); one_prompt.reserve(batch);
      caches.reserve(batch); one_cache.reserve(batch);
      cache_ptrs.reserve(batch); one_ptrs.reserve(batch);
      for (size_t b = 0; b < batch; ++b) {
        prompts.push_back(prompt(b, S));
        one_prompt.push_back(prompts.back());
        caches.emplace_back(32);
        one_cache.emplace_back(32);
        cache_ptrs.push_back(&caches.back());
        one_ptrs.push_back(&one_cache.back());
      }
      model.prefill_logits_batch_with_caches(prompts, cache_ptrs);
      for (size_t b = 0; b < batch; ++b)
        model.prefill_logits_with_cache(one_prompt[b], one_cache[b]);
      std::vector<int32_t> next(batch, 701);
      CudaSampler sampler(151936);
      for (size_t step = 0; step < 3; ++step) {
        reset_cuda_allocation_stats();
        Tensor logits = model.decode_logits_batch_with_caches(next, cache_ptrs);
        CUDA_CHECK(cudaDeviceSynchronize());
        const CudaAllocationStats stats = cuda_allocation_stats();
        if (stats.cuda_malloc_calls > 2)
          fail("batch=" + std::to_string(batch) + " step=" + std::to_string(step) +
               " allocation count=" + std::to_string(stats.cuda_malloc_calls));
        for (size_t b = 0; b < batch; ++b) {
          Tensor reference = model.decode_logits(next[b], one_cache[b]);
          compare(row_cpu(logits, b), row_cpu(reference, 0),
                  "workspace batch=" + std::to_string(batch) +
                  " step=" + std::to_string(step) + " row=" + std::to_string(b));
          if (top5(row_cpu(logits, b)) != top5(row_cpu(reference, 0)))
            fail("top-5 mismatch");
          const int32_t sampled_batch = sampler.sample_row(logits, b, sampled, step);
          const int32_t sampled_one = sampler.sample_last_row(reference, sampled, step);
          if (sampled_batch != sampled_one) fail("sampled token mismatch");
          next[b] = sampler.sample_row(logits, b, SamplingConfig{}, step);
        }
        logits = Tensor();
        CUDA_CHECK(cudaDeviceSynchronize());
      }
      for (const auto& cache : caches)
        if (cache.length() != S + 3) fail("batched cache length mismatch");
      for (const auto& cache : one_cache)
        if (cache.length() != S + 3) fail("reference cache length mismatch");
      std::cout << "workspace batch=" << batch
                << " resident_bytes=" << model.decode_workspace_bytes()
                << " warmup/steady decode allocation <=2 passed\n";
    }
    expect_throw("invalid batch", [&] {
      model.decode_logits_batch_with_caches(std::vector<int32_t>(3, 1),
                                             std::vector<Qwen3KvCache*>(3, nullptr));
    });
    expect_throw("duplicate cache", [&] {
      Qwen3KvCache cache(32);
      model.prefill_logits_with_cache(prompt(0, 4), cache);
      model.decode_logits_batch_with_caches({1, 2}, {&cache, &cache});
    });
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_decode_workspace passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
