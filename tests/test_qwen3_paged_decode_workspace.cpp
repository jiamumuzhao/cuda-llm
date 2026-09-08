#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& s) {
  throw std::runtime_error("test_qwen3_paged_decode_workspace: " + s);
}
static void expect(bool ok, const std::string& s) { if (!ok) fail(s); }
static std::vector<int32_t> prompt(size_t n, int salt) {
  std::vector<int32_t> ids(n);
  for (size_t i = 0; i < n; ++i) ids[i] = 1 + (salt + static_cast<int>(i * 31)) % 150000;
  return ids;
}
static PagedKvCachePoolConfig config(size_t blocks) {
  return {blocks, 28, 8, 16, 128, DType::F16};
}
static void compare_row(const Tensor& actual, size_t row, const Tensor& expected) {
  Tensor a = actual.to(DeviceType::CPU), b = expected.to(DeviceType::CPU);
  expect(a.shape() == std::vector<int64_t>{actual.shape()[0], 151936}, "unexpected output shape");
  float max_abs = 0.0f;
  for (size_t i = 0; i < 151936; ++i)
    max_abs = std::max(max_abs, std::fabs(a.get_f32(row * 151936 + i) - b.get_f32(i)));
  expect(max_abs <= 5e-3f, "workspace output differs from B=1 reference, max_abs=" + std::to_string(max_abs));
}

int main() {
  int devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&devices));
  if (!devices) { std::cout << "SKIP test_qwen3_paged_decode_workspace: no CUDA device\n"; return 0; }
  try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability=" << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    // Initialize the model-owned workspace on an isolated cache so the
    // per-batch measurements below compare equal cache positions.
    {
      PagedKvCachePool warm_pool(config(8));
      Qwen3KvCache warm_source(32);
      model.prefill_logits_with_cache(prompt(4, 77), warm_source);
      Qwen3PagedKvCache warm_cache(warm_pool, 32);
      warm_cache.seed_from_contiguous_cache(warm_source);
      Tensor warm = model.decode_logits_paged(901, warm_cache);
      (void)warm;
      warm_cache.release_all();
    }
    for (size_t batch : {size_t(1), size_t(2), size_t(4)}) {
      PagedKvCachePool pool(config(64));
      std::vector<Qwen3KvCache> sources;
      std::vector<std::unique_ptr<Qwen3PagedKvCache>> paged;
      std::vector<std::unique_ptr<Qwen3PagedKvCache>> refs;
      std::vector<Qwen3PagedKvCache*> pp;
      sources.reserve(batch); paged.reserve(batch); refs.reserve(batch); pp.reserve(batch);
      for (size_t b = 0; b < batch; ++b) {
        sources.emplace_back(32);
        model.prefill_logits_with_cache(prompt(4 + b, 100 + static_cast<int>(b)), sources.back());
        paged.emplace_back(std::make_unique<Qwen3PagedKvCache>(pool, 32));
        refs.emplace_back(std::make_unique<Qwen3PagedKvCache>(pool, 32));
        paged.back()->seed_from_contiguous_cache(sources.back());
        refs.back()->seed_from_contiguous_cache(sources.back());
        pp.push_back(paged.back().get());
      }
      std::vector<int32_t> ids(batch);
      for (size_t b = 0; b < batch; ++b) ids[b] = 900 + static_cast<int32_t>(b);
      reset_cuda_allocation_stats();
      Tensor actual = model.decode_logits_paged_batch(ids, pp);
      const CudaAllocationStats stats = cuda_allocation_stats();
      expect(stats.cuda_malloc_calls <= 2, "steady-state allocation count > 2 for B=" + std::to_string(batch));
      expect(model.paged_decode_metadata_workspace_bytes() == 528, "metadata workspace resident bytes changed");
      for (size_t b = 0; b < batch; ++b) {
        Tensor reference = model.decode_logits_paged(ids[b], *refs[b]);
        compare_row(actual, b, reference);
      }
      std::cout << "paged_decode_workspace B=" << batch
                << " cuda_malloc_calls=" << stats.cuda_malloc_calls
                << " cuda_free_calls=" << stats.cuda_free_calls
                << " metadata_resident_bytes=528 passed\n";
      for (auto& c : paged) c->release_all();
      for (auto& c : refs) c->release_all();
      expect(pool.used_block_count() == 0, "workspace test leaked paged blocks");
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_paged_decode_workspace passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
