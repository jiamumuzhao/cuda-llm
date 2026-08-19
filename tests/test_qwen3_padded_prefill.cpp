#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_padded_prefill.h"

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
static void fail(const std::string& message) { throw std::runtime_error("test_qwen3_padded_prefill: " + message); }
static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); } catch (const std::exception& e) { std::cout << "rejected " << name << ": " << e.what() << "\n"; return; }
  fail(name + " was accepted");
}
static std::vector<int32_t> prompt(size_t n, size_t salt) {
  std::vector<int32_t> result(n);
  for (size_t i = 0; i < n; ++i) result[i] = int32_t(1 + (salt + i * 37) % 150000);
  return result;
}
static std::vector<float> row(const Tensor& cpu, size_t index) {
  std::vector<float> result(151936);
  for (size_t i = 0; i < result.size(); ++i) result[i] = cpu.get_f32(index * result.size() + i);
  return result;
}
static std::vector<int32_t> top5(const std::vector<float>& values) {
  std::vector<int32_t> ids(values.size()); std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + 5, ids.end(), [&](int32_t a, int32_t b) {
    return values[a] != values[b] ? values[a] > values[b] : a < b;
  });
  ids.resize(5); return ids;
}
static void compare(const std::vector<float>& a, const std::vector<float>& b, const std::string& name) {
  float max_abs = 0.0f; double sum = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) fail(name + " non-finite");
    const float e = std::fabs(a[i] - b[i]); max_abs = std::max(max_abs, e); sum += e;
    dot += double(a[i]) * b[i]; na += double(a[i]) * a[i]; nb += double(b[i]) * b[i];
  }
  const double cosine = dot / std::sqrt(na * nb);
  std::cout << name << " shape=[1,151936] max_abs_error=" << max_abs
            << " mean_abs_error=" << sum / a.size() << " cosine_similarity=" << cosine << " tolerance=1e-4\n";
  if (max_abs > 1e-4f || cosine < .99999) fail(name + " mismatch");
}

int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability=" << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    CudaSampler sampler(151936);
    SamplingConfig sampled; sampled.temperature = .8f; sampled.top_k = 50; sampled.top_p = .9f; sampled.seed = 20260727;

    const std::vector<std::vector<size_t>> test_lengths{{4, 6}, {3, 4, 6, 8}};
    for (const std::vector<size_t>& lengths : test_lengths) {
      std::vector<std::vector<int32_t>> prompts;
      std::vector<Qwen3KvCache> serial_caches;
    std::vector<Qwen3KvCache> padded_caches;
      std::vector<Qwen3KvCache*> pointers;
      for (size_t b = 0; b < lengths.size(); ++b) {
        prompts.push_back(prompt(lengths[b], 17 + b * 100));
        serial_caches.emplace_back(lengths[b]);
        padded_caches.emplace_back(lengths[b]);
      }
      for (auto& cache : padded_caches) pointers.push_back(&cache);
      PaddedPrefillBatch metadata = build_padded_prefill_batch(prompts, pointers);
      if (metadata.max_seq_len != *std::max_element(lengths.begin(), lengths.end()) ||
          metadata.valid_lengths != lengths || metadata.padded_token_ids.size() != lengths.size() * metadata.max_seq_len)
        fail("metadata lengths/layout mismatch");
      for (size_t b = 0; b < lengths.size(); ++b)
        for (size_t s = 0; s < metadata.max_seq_len; ++s) {
          const int32_t expected = s < lengths[b] ? prompts[b][s] : 0;
          if (metadata.padded_token_ids[b * metadata.max_seq_len + s] != expected) fail("right padding mismatch");
        }

      std::vector<Tensor> serial_logits;
      std::vector<Tensor> serial_logits_cuda;
      for (size_t b = 0; b < lengths.size(); ++b) {
        serial_logits_cuda.push_back(model.prefill_logits_with_cache(prompts[b], serial_caches[b]));
        serial_logits.push_back(serial_logits_cuda.back().to(DeviceType::CPU));
      }
      reset_valid_length_transfer_stats();
      Tensor padded = model.prefill_logits_padded_batch_with_caches(metadata);
      const ValidLengthTransferStats transfer_stats = valid_length_transfer_stats();
      if (transfer_stats.h2d_uploads != 1 || transfer_stats.d2h_validation_copies != 0)
        fail("padded valid-length transfer count mismatch: H2D=" +
             std::to_string(transfer_stats.h2d_uploads) + " D2H=" +
             std::to_string(transfer_stats.d2h_validation_copies));
      std::cout << "valid-length transfers batch=" << lengths.size()
                << " h2d_uploads=" << transfer_stats.h2d_uploads
                << " d2h_validation_copies=" << transfer_stats.d2h_validation_copies << " passed\n";
      if (padded.shape() != std::vector<int64_t>{int64_t(lengths.size()), int64_t(metadata.max_seq_len), 151936} ||
          padded.device() != DeviceType::CUDA || !padded.is_contiguous()) fail("padded logits metadata mismatch");
      Tensor selected = select_last_valid_logits(padded, metadata.valid_lengths);
      Tensor selected_cpu = selected.to(DeviceType::CPU);
      Tensor padded_cpu = padded.to(DeviceType::CPU);
      for (size_t b = 0; b < lengths.size(); ++b) {
        compare(row(selected_cpu, b), row(serial_logits[b], lengths[b] - 1), "padded batch=" + std::to_string(lengths.size()) + " request=" + std::to_string(b));
        if (top5(row(selected_cpu, b)) != top5(row(serial_logits[b], lengths[b] - 1)))
          fail("top-5 last-valid mismatch");
        const int32_t greedy_a = sampler.sample_row(selected, b, SamplingConfig{0.0f, 0, 1.0f, 0}, 0);
        const int32_t greedy_b = sampler.sample_last_row(serial_logits_cuda[b], SamplingConfig{0.0f, 0, 1.0f, 0}, 0);
        if (greedy_a != greedy_b) fail("greedy last-valid mismatch");
        const int32_t sampled_a = sampler.sample_row(selected, b, sampled, b);
        const int32_t sampled_b = sampler.sample_last_row(serial_logits_cuda[b], sampled, b);
        if (sampled_a != sampled_b) fail("sampled last-valid mismatch");
        if (padded_caches[b].length() != lengths[b]) fail("cache committed padded valid length");
        (void)padded_cpu;
      }
      std::cout << "padded prefill batch=" << lengths.size() << " lengths=" << lengths[0] << ",... right-padding/greedy/sampling/cache lengths passed\n";
    }
    std::cout << "padded prefill attention_mask=valid_lengths_cuda passed\n";

    // Changing request 0 must not change the valid logits of the other rows.
    std::vector<std::vector<int32_t>> isolation_prompts;
    const size_t isolation_lengths[4] = {3, 4, 6, 8};
    for (size_t b = 0; b < 4; ++b) isolation_prompts.push_back(prompt(isolation_lengths[b], 701 + b));
    std::vector<Qwen3KvCache> isolation_a, isolation_b;
    std::vector<Qwen3KvCache*> isolation_ptrs_a, isolation_ptrs_b;
    isolation_a.reserve(4); isolation_b.reserve(4);
    for (size_t b = 0; b < 4; ++b) { isolation_a.emplace_back(isolation_prompts[b].size()); isolation_ptrs_a.push_back(&isolation_a.back()); }
    PaddedPrefillBatch isolation_meta_a = build_padded_prefill_batch(isolation_prompts, isolation_ptrs_a);
    Tensor isolation_a_logits = model.prefill_logits_padded_batch_with_caches(isolation_meta_a).to(DeviceType::CPU);
    isolation_prompts[0][0] += 1;
    for (size_t b = 0; b < 4; ++b) { isolation_b.emplace_back(isolation_prompts[b].size()); isolation_ptrs_b.push_back(&isolation_b.back()); }
    PaddedPrefillBatch isolation_meta_b = build_padded_prefill_batch(isolation_prompts, isolation_ptrs_b);
    Tensor isolation_b_logits = model.prefill_logits_padded_batch_with_caches(isolation_meta_b).to(DeviceType::CPU);
    for (size_t b = 1; b < 4; ++b)
      compare(row(isolation_b_logits, b * isolation_meta_b.max_seq_len + isolation_meta_b.valid_lengths[b] - 1),
              row(isolation_a_logits, b * isolation_meta_a.max_seq_len + isolation_meta_a.valid_lengths[b] - 1),
              "padded isolation request=" + std::to_string(b));
    std::cout << "padded batch isolation passed\n";

    // The public metadata can be hand-constructed; the model must re-check
    // right padding instead of relying only on the builder.
    std::vector<std::vector<int32_t>> metadata_prompts{prompt(4, 901), prompt(6, 902)};
    std::vector<Qwen3KvCache> metadata_caches;
    metadata_caches.emplace_back(4); metadata_caches.emplace_back(6);
    std::vector<Qwen3KvCache*> metadata_cache_ptrs{&metadata_caches[0], &metadata_caches[1]};
    PaddedPrefillBatch invalid_metadata = build_padded_prefill_batch(metadata_prompts, metadata_cache_ptrs);
    invalid_metadata.padded_token_ids[4] = 123;
    expect_throw("model rejects non-zero right padding", [&] {
      model.prefill_logits_padded_batch_with_caches(invalid_metadata);
    });
    if (metadata_caches[0].length() != 0 || metadata_caches[1].length() != 0)
      fail("invalid metadata changed cache length");
    std::cout << "model-level right-padding rejection row=0 position=4 actual_token=123 cache_lengths=[0,0] passed\n";

    expect_throw("empty batch", [] { build_padded_prefill_batch({}, {}); });
    expect_throw("batch size 3", [&] { std::vector<Qwen3KvCache*> c(3); build_padded_prefill_batch({prompt(2, 1), prompt(2, 2), prompt(2, 3)}, c); });
    expect_throw("empty prompt", [&] { Qwen3KvCache c(8); build_padded_prefill_batch({{}}, {&c}); });
    expect_throw("token out of range", [&] { Qwen3KvCache c(8); build_padded_prefill_batch({{151936}}, {&c}); });
    expect_throw("long prompt", [&] { Qwen3KvCache c(64); build_padded_prefill_batch({prompt(33, 1)}, {&c}); });
    expect_throw("null cache", [&] { build_padded_prefill_batch({prompt(2, 1)}, {nullptr}); });
    expect_throw("duplicate cache", [&] { Qwen3KvCache c(8); build_padded_prefill_batch({prompt(2, 1), prompt(3, 2)}, {&c, &c}); });
    expect_throw("capacity", [&] { Qwen3KvCache c(1); build_padded_prefill_batch({prompt(2, 1)}, {&c}); });
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "padded prefill rejection paths and CUDA health check passed\n";
    std::cout << "test_qwen3_padded_prefill passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
