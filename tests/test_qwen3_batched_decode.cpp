#include "llm/cuda_check.h"
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
  throw std::runtime_error("test_qwen3_batched_decode: " + message);
}
static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); }
  catch (const std::exception& error) {
    std::cout << "rejected " << name << ": " << error.what() << "\n";
    return;
  }
  fail(name + " was accepted");
}
static std::vector<int32_t> ids_for(size_t index, size_t length, int salt) {
  std::vector<int32_t> result(length);
  for (size_t i = 0; i < length; ++i)
    result[i] = int32_t(1 + ((salt + int(index * 97) + int(i * 31)) % 150000));
  return result;
}
static std::vector<int32_t> top5(const std::vector<float>& row) {
  std::vector<int32_t> indices(row.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                    [&](int32_t a, int32_t b) {
                      return row[a] != row[b] ? row[a] > row[b] : a < b;
                    });
  indices.resize(5);
  return indices;
}
static std::vector<float> row_cpu(const Tensor& tensor, size_t row, size_t width) {
  std::vector<float> result(width);
  for (size_t i = 0; i < width; ++i) result[i] = tensor.get_f32(row * width + i);
  return result;
}
static void compare(const std::vector<float>& actual,
                    const std::vector<float>& expected,
                    const std::string& name) {
  float max_abs = 0.0f;
  double mean = 0.0, dot = 0.0, aa = 0.0, ee = 0.0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float error = std::fabs(actual[i] - expected[i]);
    max_abs = std::max(max_abs, error);
    mean += error;
    dot += double(actual[i]) * expected[i];
    aa += double(actual[i]) * actual[i];
    ee += double(expected[i]) * expected[i];
  }
  const double cosine = dot / std::sqrt(aa * ee);
  std::cout << name << " max_abs_error=" << max_abs
            << " mean_abs_error=" << mean / actual.size()
            << " cosine_similarity=" << cosine << " tolerance=1e-4\n";
  if (max_abs > 1e-4f || cosine < 0.99999) fail(name + " numeric mismatch");
}

int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability="
              << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    constexpr size_t S = 4;
    constexpr size_t V = 151936;
    SamplingConfig sampled;
    sampled.temperature = 0.8f; sampled.top_k = 50; sampled.top_p = 0.9f;
    sampled.seed = 20260727;

    for (size_t batch_size : {size_t(1), size_t(2), size_t(4)}) {
      std::vector<std::vector<int32_t>> prompts;
      std::vector<int32_t> next_tokens;
      std::vector<Qwen3KvCache> serial_caches;
      std::vector<Qwen3KvCache> batched_caches;
      std::vector<Qwen3KvCache*> batch_ptrs;
      std::vector<std::vector<float>> serial_logits;
      std::vector<Tensor> serial_decode_tensors;
      prompts.reserve(batch_size); next_tokens.reserve(batch_size);
      serial_caches.reserve(batch_size); batched_caches.reserve(batch_size);
      serial_decode_tensors.reserve(batch_size);
      batch_ptrs.reserve(batch_size);
      for (size_t b = 0; b < batch_size; ++b) {
        prompts.push_back(ids_for(b, S, 17));
        next_tokens.push_back(int32_t(500 + b * 13));
        serial_caches.emplace_back(32);
        Tensor prefill = model.prefill_logits_with_cache(prompts.back(), serial_caches.back());
        Tensor serial_decode = model.decode_logits(next_tokens.back(), serial_caches.back());
        serial_logits.push_back(row_cpu(serial_decode.to(DeviceType::CPU), 0, V));
        serial_decode_tensors.push_back(std::move(serial_decode));
        batched_caches.emplace_back(32);
        batch_ptrs.push_back(&batched_caches.back());
      }
      Tensor batched_prefill = model.prefill_logits_batch_with_caches(prompts, batch_ptrs);
      if (batched_prefill.shape() != std::vector<int64_t>{int64_t(batch_size), int64_t(S), int64_t(V)})
        fail("batched prefill shape mismatch");
      Tensor batched_logits = model.decode_logits_batch_with_caches(next_tokens, batch_ptrs);
      if (batched_logits.shape() != std::vector<int64_t>{int64_t(batch_size), int64_t(V)} ||
          batched_logits.dtype() != DType::F16 || batched_logits.device() != DeviceType::CUDA ||
          !batched_logits.is_contiguous()) fail("batched decode output metadata mismatch");
      Tensor batched_cpu = batched_logits.to(DeviceType::CPU);
      CudaSampler sampler(V);
      for (size_t b = 0; b < batch_size; ++b) {
        auto actual = row_cpu(batched_cpu, b, V);
        compare(actual, serial_logits[b], "decode batch=" + std::to_string(batch_size) +
                                         " request=" + std::to_string(b));
        if (top5(actual) != top5(serial_logits[b])) fail("decode top-5 mismatch");
        const int32_t greedy = sampler.sample_row(batched_logits, b, SamplingConfig{}, 0);
        const int32_t expected_greedy = int32_t(std::distance(
            serial_logits[b].begin(), std::max_element(serial_logits[b].begin(), serial_logits[b].end())));
        if (greedy != expected_greedy) fail("decode greedy token mismatch");
        const int32_t sampled_batch = sampler.sample_row(batched_logits, b, sampled, 3);
        const int32_t sampled_serial = sampler.sample_last_row(serial_decode_tensors[b], sampled, 3);
        if (sampled_batch != sampled_serial) fail("sampled row result mismatch");
        if (batched_caches[b].length() != S + 1 || serial_caches[b].length() != S + 1)
          fail("decode cache length mismatch");
      }
      std::cout << "batched decode batch_size=" << batch_size
                << " input=[B,1024] q=[B,16,128] kv=[B,8,128] logits=[B,151936] passed\n";
    }

    std::vector<std::vector<int32_t>> isolation_prompts{ids_for(0, S, 71), ids_for(1, S, 71)};
    std::vector<Qwen3KvCache> isolation_base, isolation_changed;
    std::vector<Qwen3KvCache*> base_ptrs, changed_ptrs;
    isolation_base.reserve(2); isolation_changed.reserve(2);
    base_ptrs.reserve(2); changed_ptrs.reserve(2);
    for (size_t b = 0; b < 2; ++b) {
      isolation_base.emplace_back(32); isolation_changed.emplace_back(32);
      base_ptrs.push_back(&isolation_base.back()); changed_ptrs.push_back(&isolation_changed.back());
    }
    model.prefill_logits_batch_with_caches(isolation_prompts, base_ptrs);
    model.prefill_logits_batch_with_caches(isolation_prompts, changed_ptrs);
    const std::vector<int32_t> isolation_tokens{701, 702};
    Tensor isolation_reference = model.decode_logits_batch_with_caches(isolation_tokens, base_ptrs);
    auto changed_tokens = isolation_tokens; changed_tokens[0] += 1;
    Tensor isolation_changed_logits = model.decode_logits_batch_with_caches(changed_tokens, changed_ptrs);
    compare(row_cpu(isolation_changed_logits.to(DeviceType::CPU), 1, V),
            row_cpu(isolation_reference.to(DeviceType::CPU), 1, V), "batch isolation request=1");
    if (isolation_base[1].length() != S + 1 || isolation_changed[1].length() != S + 1)
      fail("batch isolation cache length mismatch");
    std::cout << "batched decode isolation passed\n";

    expect_throw("batch size zero", [&] {
      model.decode_logits_batch_with_caches({}, {});
    });
    expect_throw("batch size three", [&] {
      model.decode_logits_batch_with_caches(std::vector<int32_t>(3, 1),
                                             std::vector<Qwen3KvCache*>(3, nullptr));
    });
    expect_throw("batch size five", [&] {
      model.decode_logits_batch_with_caches(std::vector<int32_t>(5, 1),
                                             std::vector<Qwen3KvCache*>(5, nullptr));
    });
    expect_throw("count mismatch", [&] {
      model.decode_logits_batch_with_caches({1, 2}, {});
    });
    expect_throw("duplicate cache", [&] {
      Qwen3KvCache cache(32);
      model.prefill_logits_with_cache(ids_for(0, S, 901), cache);
      model.decode_logits_batch_with_caches({1, 2}, {&cache, &cache});
    });
    expect_throw("empty cache", [&] {
      Qwen3KvCache cache(32);
      model.decode_logits_batch_with_caches({1}, {&cache});
    });
    expect_throw("full cache", [&] {
      Qwen3KvCache cache(S);
      model.prefill_logits_with_cache(ids_for(0, S, 902), cache);
      model.decode_logits_batch_with_caches({1}, {&cache});
    });
    expect_throw("different cache lengths", [&] {
      Qwen3KvCache first(32), second(32);
      model.prefill_logits_with_cache(ids_for(0, S, 903), first);
      model.prefill_logits_with_cache(ids_for(1, S + 1, 903), second);
      model.decode_logits_batch_with_caches({1, 2}, {&first, &second});
    });
    expect_throw("invalid token", [&] {
      Qwen3KvCache cache(32);
      model.prefill_logits_with_cache(ids_for(0, S, 904), cache);
      model.decode_logits_batch_with_caches({151936}, {&cache});
    });
    expect_throw("cache config mismatch", [&] {
      Qwen3KvCache cache(32, 27);
      model.decode_logits_batch_with_caches({1}, {&cache});
    });

    Tensor health_host = Tensor::from_f32({4}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor health_device = health_host.to(DeviceType::CUDA);
    Tensor health = cuda_add(health_device, health_device);
    Tensor health_cpu = health.to(DeviceType::CPU);
    for (size_t i = 0; i < 4; ++i)
      if (health_cpu.get_f32(i) != 2.0f * health_host.get_f32(i)) fail("CUDA health check mismatch");
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "post-batched-decode CUDA health check passed\n";
    std::cout << "test_qwen3_batched_decode passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
