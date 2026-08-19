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
static void fail(const std::string& m) { throw std::runtime_error("test_qwen3_variable_length_decode: " + m); }
static std::vector<int32_t> ids(size_t row, size_t n) {
  std::vector<int32_t> r(n);
  for (size_t i = 0; i < n; ++i) r[i] = int32_t(1 + ((row * 173 + i * 37 + 19) % 150000));
  return r;
}
static std::vector<float> row(const Tensor& t, size_t b, size_t width) {
  std::vector<float> r(width); for (size_t i = 0; i < width; ++i) r[i] = t.get_f32(b * width + i); return r;
}
static std::vector<int32_t> top5(const std::vector<float>& r) {
  std::vector<int32_t> x(r.size()); std::iota(x.begin(), x.end(), 0);
  std::partial_sort(x.begin(), x.begin() + 5, x.end(), [&](int32_t a, int32_t b) {
    return r[a] != r[b] ? r[a] > r[b] : a < b;
  }); x.resize(5); return x;
}
static void compare(const std::vector<float>& a, const std::vector<float>& b, const std::string& name) {
  float max_abs = 0.0f; double dot = 0.0, aa = 0.0, bb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) { max_abs = std::max(max_abs, std::fabs(a[i] - b[i])); dot += double(a[i]) * b[i]; aa += double(a[i]) * a[i]; bb += double(b[i]) * b[i]; }
  const double cosine = dot / std::sqrt(aa * bb);
  std::cout << name << " max_abs_error=" << max_abs << " cosine_similarity=" << cosine << " tolerance=1e-4\n";
  if (max_abs > 1e-4f || cosine < 0.99999) fail(name + " numeric mismatch");
}
static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); } catch (const std::exception& e) { std::cout << "rejected " << name << ": " << e.what() << "\n"; return; }
  fail(name + " was accepted");
}

int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability=" << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    CudaSampler sampler(151936); SamplingConfig sampled; sampled.temperature = 0.8f; sampled.top_k = 50; sampled.top_p = 0.9f; sampled.seed = 20260727;
    for (const std::vector<size_t>& lengths : {std::vector<size_t>{4, 6}, std::vector<size_t>{3, 4, 6, 8}}) {
      const size_t batch = lengths.size();
      std::vector<std::vector<int32_t>> prompts; std::vector<Qwen3KvCache> variable_caches, serial_caches;
      std::vector<Qwen3KvCache*> pointers; std::vector<Tensor> serial_logits;
      std::vector<int32_t> next_tokens;
      for (size_t b = 0; b < batch; ++b) {
        prompts.push_back(ids(b, lengths[b])); next_tokens.push_back(int32_t(700 + b * 13));
        variable_caches.emplace_back(lengths[b] + 3); serial_caches.emplace_back(lengths[b] + 3);
        model.prefill_logits_with_cache(prompts.back(), variable_caches.back());
        model.prefill_logits_with_cache(prompts.back(), serial_caches.back());
        Tensor serial = model.decode_logits(next_tokens.back(), serial_caches.back());
        serial_logits.push_back(std::move(serial));
      }
      reset_variable_decode_transfer_stats();
      std::vector<Qwen3KvCache*> variable_ptrs; for (auto& c : variable_caches) variable_ptrs.push_back(&c);
      Tensor actual = model.decode_logits_variable_length_batch_with_caches(next_tokens, variable_ptrs);
      const auto transfers = variable_decode_transfer_stats();
      if (transfers.h2d_uploads != 1 || transfers.d2h_validation_copies != 0) fail("per-row metadata transfer count mismatch");
      Tensor cpu = actual.to(DeviceType::CPU);
      for (size_t b = 0; b < batch; ++b) {
        const auto a = row(cpu, b, 151936), e = row(serial_logits[b].to(DeviceType::CPU), 0, 151936);
        compare(a, e, "variable batch=" + std::to_string(batch) + " request=" + std::to_string(b));
        if (top5(a) != top5(e)) fail("top-5 mismatch");
        if (sampler.sample_row(actual, b, SamplingConfig{}, 3) != sampler.sample_last_row(serial_logits[b], SamplingConfig{}, 3)) fail("greedy mismatch");
        if (sampler.sample_row(actual, b, sampled, 3) != sampler.sample_last_row(serial_logits[b], sampled, 3)) fail("sample mismatch");
        if (variable_caches[b].length() != lengths[b] + 1 || serial_caches[b].length() != lengths[b] + 1) fail("cache L->L+1 mismatch");
      }
      std::cout << "variable decode batch=" << batch << " cache_lengths=";
      for (size_t b = 0; b < batch; ++b) std::cout << (b ? "," : "[") << lengths[b] << "->" << lengths[b] + 1;
      std::cout << "] metadata_h2d=1 metadata_d2h=0 top5/greedy/sampling passed\n";
    }
    expect_throw("B=3", [&] { model.decode_logits_variable_length_batch_with_caches(std::vector<int32_t>(3, 1), std::vector<Qwen3KvCache*>(3, nullptr)); });
    expect_throw("empty cache", [&] { Qwen3KvCache c(8); model.decode_logits_variable_length_batch_with_caches({1}, {&c}); });
    expect_throw("duplicate cache", [&] { Qwen3KvCache c(8); model.prefill_logits_with_cache(ids(0, 4), c); model.decode_logits_variable_length_batch_with_caches({1, 2}, {&c, &c}); });
    expect_throw("invalid token", [&] { Qwen3KvCache c(8); model.prefill_logits_with_cache(ids(0, 4), c); model.decode_logits_variable_length_batch_with_caches({151936}, {&c}); });
    expect_throw("cache full", [&] { Qwen3KvCache c(4); model.prefill_logits_with_cache(ids(0, 4), c); model.decode_logits_variable_length_batch_with_caches({1}, {&c}); });
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_variable_length_decode passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
