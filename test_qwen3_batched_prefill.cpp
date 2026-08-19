#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& message) {
  throw std::runtime_error("test_qwen3_batched_prefill: " + message);
}

static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); }
  catch (const std::exception& e) {
    std::cout << "rejected " << name << ": " << e.what() << "\n";
    return;
  }
  fail(name + " was accepted");
}

static std::vector<int32_t> top5(const std::vector<float>& row) {
  std::vector<int32_t> ids(row.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + 5, ids.end(),
                    [&](int32_t a, int32_t b) {
                      if (row[a] != row[b]) return row[a] > row[b];
                      return a < b;
                    });
  ids.resize(5);
  return ids;
}

static void compare_row(const std::vector<float>& actual,
                        const std::vector<float>& expected,
                        const std::string& name) {
  if (actual.size() != expected.size()) fail(name + " size mismatch");
  double sum_abs = 0.0, dot = 0.0, na = 0.0, ne = 0.0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) fail(name + " non-finite");
    const float error = std::fabs(actual[i] - expected[i]);
    max_abs = std::max(max_abs, error);
    sum_abs += error;
    dot += double(actual[i]) * expected[i];
    na += double(actual[i]) * actual[i];
    ne += double(expected[i]) * expected[i];
  }
  const double cosine = dot / std::sqrt(na * ne);
  std::cout << name << " max_abs_error=" << max_abs
            << " mean_abs_error=" << sum_abs / actual.size()
            << " cosine_similarity=" << cosine << " tolerance=1e-4\n";
  if (max_abs > 1e-4f || cosine < 0.99999) fail(name + " numeric mismatch");
}

static std::vector<float> cpu_row(const Tensor& cpu, size_t row, size_t width) {
  std::vector<float> result(width);
  for (size_t i = 0; i < width; ++i) result[i] = cpu.get_f32(row * width + i);
  return result;
}

static std::vector<int32_t> prompt(size_t batch, size_t seq, size_t salt) {
  std::vector<int32_t> ids(seq);
  for (size_t i = 0; i < seq; ++i) ids[i] = int32_t(1 + ((salt + batch * 97 + i * 31) % 150000));
  return ids;
}

int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability="
              << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    constexpr size_t S = 4;
    constexpr size_t V = 151936;

    for (size_t batch_size : {size_t(1), size_t(2), size_t(4)}) {
      std::vector<std::vector<int32_t>> prompts;
      std::vector<Qwen3KvCache> serial_caches;
      std::vector<Qwen3KvCache*> batch_caches;
      std::vector<std::vector<float>> serial_rows;
      for (size_t b = 0; b < batch_size; ++b) {
        prompts.push_back(prompt(b, S, 11));
        serial_caches.emplace_back(32);
        Tensor serial = model.prefill_logits_with_cache(prompts.back(), serial_caches.back());
        Tensor serial_cpu = serial.to(DeviceType::CPU);
        serial_rows.push_back(cpu_row(serial_cpu, S - 1, V));
        batch_caches.push_back(nullptr);
      }
      std::vector<Qwen3KvCache> batched_storage;
      for (size_t b = 0; b < batch_size; ++b) {
        batched_storage.emplace_back(32);
        batch_caches[b] = &batched_storage[b];
      }
      Tensor batch_logits = model.prefill_logits_batch_with_caches(prompts, batch_caches);
      if (batch_logits.shape() != std::vector<int64_t>{int64_t(batch_size), int64_t(S), int64_t(V)} ||
          batch_logits.device() != DeviceType::CUDA || !batch_logits.is_contiguous())
        fail("batched logits metadata mismatch");
      Tensor batch_cpu = batch_logits.to(DeviceType::CPU);
      for (size_t b = 0; b < batch_size; ++b) {
        auto actual = cpu_row(batch_cpu, b * S + S - 1, V);
        compare_row(actual, serial_rows[b], "batch=" + std::to_string(batch_size) + " request=" + std::to_string(b));
        if (top5(actual) != top5(serial_rows[b])) fail("batch top5 mismatch");
        if (batched_storage[b].length() != S) fail("batch cache length mismatch");
      }
      CudaSampler sampler(V);
      SamplingConfig greedy; greedy.temperature = 0.0f;
      for (size_t b = 0; b < batch_size; ++b) {
        const int32_t sampled = sampler.sample_last_token_of_batch(batch_logits, b, greedy, 0);
        const auto expected = int32_t(std::distance(serial_rows[b].begin(),
          std::max_element(serial_rows[b].begin(), serial_rows[b].end())));
        if (sampled != expected) fail("batched greedy token mismatch");
      }
      std::cout << "batched prefill batch_size=" << batch_size << " seq_len=" << S
                << " layout=[B,S,1024], q=[B,S,16,128], kv=[B,S,8,128], logits=[B,S,151936] passed\n";

      for (size_t b = 0; b < batch_size; ++b) {
        const int32_t fixed_token = int32_t(500 + b);
        Tensor serial_decode = model.decode_logits(fixed_token, serial_caches[b]);
        Tensor batch_decode = model.decode_logits(fixed_token, batched_storage[b]);
        compare_row(cpu_row(batch_decode.to(DeviceType::CPU), 0, V),
                    cpu_row(serial_decode.to(DeviceType::CPU), 0, V),
                    "decode batch=" + std::to_string(batch_size) + " request=" + std::to_string(b));
      }
    }

    // Batch isolation: changing request 0 must not change requests 1..3.
    std::vector<std::vector<int32_t>> base_prompts;
    for (size_t b = 0; b < 4; ++b) base_prompts.push_back(prompt(b, S, 101));
    std::vector<Qwen3KvCache> base_caches, changed_caches;
    std::vector<Qwen3KvCache*> base_ptrs, changed_ptrs;
    for (size_t b = 0; b < 4; ++b) { base_caches.emplace_back(32); changed_caches.emplace_back(32); base_ptrs.push_back(&base_caches.back()); changed_ptrs.push_back(&changed_caches.back()); }
    Tensor base = model.prefill_logits_batch_with_caches(base_prompts, base_ptrs).to(DeviceType::CPU);
    base_prompts[0][0] += 1;
    Tensor changed = model.prefill_logits_batch_with_caches(base_prompts, changed_ptrs).to(DeviceType::CPU);
    for (size_t b = 1; b < 4; ++b) compare_row(cpu_row(changed, b*S+S-1, V), cpu_row(base, b*S+S-1, V), "batch isolation request=" + std::to_string(b));
    std::cout << "batch isolation passed\n";

    const auto good_prompts = std::vector<std::vector<int32_t>>{prompt(0, S, 7), prompt(1, S, 7)};
    expect_throw("empty batch", [&] { model.prefill_logits_batch_with_caches({}, {}); });
    expect_throw("unsupported batch size", [&] { std::vector<Qwen3KvCache*> c(3); model.prefill_logits_batch_with_caches({good_prompts[0], good_prompts[1], good_prompts[0]}, c); });
    expect_throw("mismatched cache count", [&] { std::vector<Qwen3KvCache> c(2); std::vector<Qwen3KvCache*> p{&c[0]}; model.prefill_logits_batch_with_caches(good_prompts, p); });
    expect_throw("unequal sequence lengths", [&] { std::vector<Qwen3KvCache> c(2); std::vector<Qwen3KvCache*> p{&c[0], &c[1]}; model.prefill_logits_batch_with_caches({good_prompts[0], prompt(1, 3, 7)}, p); });
    expect_throw("token out of range", [&] { auto bad=good_prompts; bad[0][0]=151936; std::vector<Qwen3KvCache> c(2); std::vector<Qwen3KvCache*> p{&c[0],&c[1]}; model.prefill_logits_batch_with_caches(bad,p); });
    expect_throw("capacity too small", [&] { std::vector<Qwen3KvCache> c(2); std::vector<Qwen3KvCache*> p{&c[0],&c[1]}; model.prefill_logits_batch_with_caches(good_prompts,p); });
    expect_throw("duplicate cache pointer", [&] { Qwen3KvCache c(32); std::vector<Qwen3KvCache*> p{&c,&c}; model.prefill_logits_batch_with_caches(good_prompts,p); });
    expect_throw("cache config mismatch", [&] { Qwen3KvCache c0(32,27), c1(32); std::vector<Qwen3KvCache*> p{&c0,&c1}; model.prefill_logits_batch_with_caches(good_prompts,p); });
    expect_throw("zero sequence", [&] { Qwen3KvCache c(32); std::vector<Qwen3KvCache*> p{&c}; model.prefill_logits_batch_with_caches({{}},p); });
    std::cout << "batched prefill rejection paths passed\n";
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_batched_prefill passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
