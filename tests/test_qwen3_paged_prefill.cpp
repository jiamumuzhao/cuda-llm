#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& s) { throw std::runtime_error("test_qwen3_paged_prefill: " + s); }
static void expect(bool ok, const std::string& s) { if (!ok) fail(s); }
static std::vector<int32_t> prompt(size_t n, int salt) {
  std::vector<int32_t> ids(n);
  for (size_t i = 0; i < n; ++i) ids[i] = 1 + (salt + static_cast<int>(i * 41)) % 150000;
  return ids;
}
static PagedKvCachePoolConfig pool_config(size_t blocks) {
  return {blocks, 28, 8, 16, 128, DType::F16};
}
static Tensor filled_kv(size_t length, std::uint16_t base) {
  Tensor host(DType::F16, {static_cast<int64_t>(length), 8, 128});
  for (size_t i = 0; i < host.numel(); ++i)
    host.data_f16()[i] = static_cast<std::uint16_t>(base + i * 7u);
  return host.to(DeviceType::CUDA);
}
static Tensor single_pattern(std::uint16_t base) {
  Tensor host(DType::F16, {8, 128});
  for (size_t i = 0; i < host.numel(); ++i)
    host.data_f16()[i] = static_cast<std::uint16_t>(base + i * 13u);
  return host.to(DeviceType::CUDA);
}
static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); } catch (const std::exception& e) {
    std::cout << "rejected " << name << ": " << e.what() << "\n"; return;
  }
  fail(name + " was accepted");
}
static std::vector<int32_t> top5(const Tensor& x, size_t row = 0) {
  Tensor cpu = x.to(DeviceType::CPU);
  std::vector<int32_t> ids(151936);
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + 5, ids.end(), [&](int a, int b) {
    const float av = cpu.get_f32(row * 151936 + a), bv = cpu.get_f32(row * 151936 + b);
    return av != bv ? av > bv : a < b;
  });
  ids.resize(5);
  return ids;
}
static void compare_logits(const Tensor& a, const Tensor& b, const std::string& name) {
  Tensor ac = a.to(DeviceType::CPU), bc = b.to(DeviceType::CPU);
  expect(ac.shape() == bc.shape(), name + " shape mismatch");
  float max_abs = 0.0f, mean_abs = 0.0f;
  for (size_t i = 0; i < ac.numel(); ++i) {
    const float e = std::fabs(ac.get_f32(i) - bc.get_f32(i));
    max_abs = std::max(max_abs, e); mean_abs += e;
  }
  mean_abs /= static_cast<float>(ac.numel());
  expect(max_abs <= 5e-3f, name + " max_abs_error=" + std::to_string(max_abs));
  std::cout << name << " max_abs_error=" << max_abs << " mean_abs_error=" << mean_abs << "\n";
}
static void compare_kv(const Qwen3PagedKvCache& paged, const Qwen3KvCache& contiguous,
                       size_t length, const std::string& name) {
  for (size_t layer = 0; layer < 28; ++layer) {
    Tensor k = contiguous.key_cache(layer).to(DeviceType::CPU);
    Tensor v = contiguous.value_cache(layer).to(DeviceType::CPU);
    const auto* kb = k.data_f16(); const auto* vb = v.data_f16();
    for (size_t token = 0; token < length; ++token) {
      std::vector<uint16_t> pk(1024), pv(1024);
      paged.copy_layer_kv_to_host(layer, token, pk.data(), pv.data(), 1024);
      for (size_t i = 0; i < 1024; ++i) {
        if (pk[i] != kb[token * 1024 + i] || pv[i] != vb[token * 1024 + i])
          fail(name + " bit mismatch layer=" + std::to_string(layer) +
               " token=" + std::to_string(token) + " index=" + std::to_string(i));
      }
    }
  }
}

int main() {
  int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
  if (!devices) { std::cout << "SKIP test_qwen3_paged_prefill: no CUDA device\n"; return 0; }
  try {
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    for (size_t length : {size_t(1), size_t(4), size_t(15), size_t(16), size_t(17), size_t(31)}) {
      const std::vector<int32_t> ids = prompt(length, static_cast<int>(length * 17));
      Qwen3KvCache contiguous(32);
      Tensor expected = model.prefill_logits_with_cache(ids, contiguous);
      PagedKvCachePool pool(pool_config(16));
      const BlockId fragmented_a = pool.block_manager().allocate();
      const BlockId fragmented_b = pool.block_manager().allocate();
      pool.block_manager().release(fragmented_a);
      Qwen3PagedKvCache direct(pool, 32);
      Tensor actual = model.prefill_logits_paged(ids, direct);
      expect(direct.length() == length, "direct prefill length mismatch");
      if (length >= 17) {
        const auto& table = direct.block_table();
        expect(table.size() >= 2, "fragmented direct table has fewer than two blocks");
        bool non_contiguous = false;
        for (size_t i = 0; i + 1 < table.size(); ++i)
          non_contiguous |= table[i + 1] != table[i] + 1;
        expect(non_contiguous, "fragmented direct table is accidentally contiguous");
        expect(std::find(table.begin(), table.end(), fragmented_b) == table.end(),
               "retained blocker block entered direct table");
        std::cout << "context=" << length << " fragmented table=[";
        for (size_t i = 0; i < table.size(); ++i) std::cout << (i ? "," : "") << table[i];
        std::cout << "]\n";
      }
      compare_logits(actual, expected, "prefill length=" + std::to_string(length));
      expect(top5(actual, length - 1) == top5(expected, length - 1),
             "prefill top5 mismatch length=" + std::to_string(length));
      CudaSampler sampler(151936); SamplingConfig sampling; sampling.temperature = .8f;
      sampling.top_k = 50; sampling.top_p = .9f; sampling.seed = 20260727;
      expect(sampler.sample_last_row(actual, sampling, 11) == sampler.sample_last_row(expected, sampling, 11),
             "prefill sampling mismatch length=" + std::to_string(length));
      compare_kv(direct, contiguous, length, "prefill length=" + std::to_string(length));

      Qwen3PagedKvCache seeded(pool, 32);
      seeded.seed_from_contiguous_cache(contiguous);
      Tensor direct_decode = model.decode_logits_paged(901, direct);
      Tensor seeded_decode = model.decode_logits_paged(901, seeded);
      compare_logits(direct_decode, seeded_decode, "decode after prefill length=" + std::to_string(length));
      expect(top5(direct_decode) == top5(seeded_decode), "decode top5 mismatch");
      expect(direct.length() == length + 1 && seeded.length() == length + 1, "decode length mismatch");
      direct.release_all(); seeded.release_all();
      pool.block_manager().release(fragmented_b);
      expect(pool.used_block_count() == 0, "paged prefill leaked blocks");
      std::cout << "context=" << length << " direct paged prefill/KV/decode passed\n";
    }

    // Long-context regression: packed paged prefill crosses the 32-block
    // boundary, then paged decode appends the 512th token.
    {
      const std::vector<int32_t> ids = prompt(511, 5121);
      PagedKvCachePool pool(pool_config(40));
      const BlockId fragmented = pool.block_manager().allocate();
      Qwen3PagedKvCache cache(pool, 512);
      Tensor prefill = model.prefill_logits_paged(ids, cache);
      expect(prefill.shape() == std::vector<int64_t>{511, 151936},
             "long packed prefill logits shape mismatch");
      expect(cache.length() == 511 && cache.block_table().size() == 32,
             "long packed prefill did not allocate 32 blocks");
      expect(cache.block_table().front() != fragmented,
             "long packed prefill reused released blocker unexpectedly");
      Tensor decode = model.decode_logits_paged(901, cache);
      expect(decode.shape() == std::vector<int64_t>{1, 151936},
             "long paged decode logits shape mismatch");
      expect(cache.length() == 512 && cache.block_table().size() == 32,
             "long paged decode did not reach max_seq_len=512");
      CUDA_CHECK(cudaDeviceSynchronize());
      cache.release_all();
      pool.block_manager().release(fragmented);
      expect(pool.used_block_count() == 0, "long paged prefill/decode leaked blocks");
      std::cout << "context=511->512 packed paged prefill/decode passed\n";
    }

    // Direct transaction API: full commit, explicit abort, conflicts, and limits.
    PagedKvCachePool pool(pool_config(4));
    Qwen3PagedKvCache cache(pool, 32);
    Tensor key = filled_kv(17, 0x1000), value = filled_kv(17, 0x4000);
    cache.begin_prefill(17);
    expect(cache.in_prefill_transaction() && cache.length() == 0 && cache.block_table().size() == 2,
           "prefill transaction state");
    for (size_t layer = 0; layer < 28; ++layer) cache.append_prefill_layer_kv(layer, key, value);
    cache.commit_prefill();
    expect(cache.length() == 17 && cache.block_table().size() == 2, "prefill commit state");
    cache.release_all();
    const size_t free_before_abort = pool.free_block_count();
    cache.begin_prefill(17);
    cache.append_prefill_layer_kv(0, key, value);
    expect_throw("prefill duplicate layer", [&] { cache.append_prefill_layer_kv(0, key, value); });
    cache.abort_prefill();
    expect(cache.length() == 0 && cache.block_table().empty() && pool.used_block_count() == 0 &&
               pool.free_block_count() == free_before_abort,
           "prefill abort did not restore state");
    cache.begin_prefill(17);
    cache.append_prefill_layer_kv(0, key, value);
    expect_throw("prefill incomplete commit", [&] { cache.commit_prefill(); });
    expect(cache.length() == 0 && cache.block_table().empty() && pool.used_block_count() == 0,
           "incomplete prefill commit did not auto-rollback");
    cache.begin_prefill(1);
    expect_throw("decode/prefill conflict", [&] { cache.begin_decode(); });
    cache.abort_prefill();
    cache.begin_decode();
    expect_throw("prefill/decode conflict", [&] { cache.begin_prefill(1); });
    cache.abort_decode();
    expect_throw("zero prefill tokens", [&] { cache.begin_prefill(0); });
    expect_throw("prefill over capacity", [&] { cache.begin_prefill(33); });

    PagedKvCachePool exhausted_pool(pool_config(2));
    Qwen3PagedKvCache blocker(exhausted_pool, 16), exhausted_target(exhausted_pool, 32);
    blocker.begin_decode();
    const Tensor blocker_key = single_pattern(0x5000), blocker_value = single_pattern(0x7000);
    for (size_t layer = 0; layer < 28; ++layer) blocker.append_layer_kv(layer, blocker_key, blocker_value);
    blocker.commit_decode();
    expect_throw("prefill begin pool exhaustion", [&] { exhausted_target.begin_prefill(17); });
    expect(exhausted_target.length() == 0 && exhausted_target.block_table().empty() &&
               exhausted_pool.used_block_count() == 1,
           "prefill pool exhaustion left partial blocks");
    blocker.release_all();
    exhausted_target.begin_prefill(17);
    for (size_t layer = 0; layer < 28; ++layer) exhausted_target.append_prefill_layer_kv(layer, key, value);
    exhausted_target.commit_prefill();
    expect(exhausted_target.length() == 17, "prefill did not recover after pool release");
    exhausted_target.release_all();

    // Model-level deterministic failure proves catch -> abort_prefill, then reuse.
    PagedKvCachePool fault_pool(pool_config(8));
    Qwen3PagedKvCache fault_cache(fault_pool, 32);
    const std::vector<int32_t> fault_ids = prompt(4, 991);
    qwen3_set_paged_prefill_fault_for_testing(9);
    expect_throw("model direct prefill injected fault", [&] { model.prefill_logits_paged(fault_ids, fault_cache); });
    qwen3_clear_paged_prefill_fault_for_testing();
    expect(fault_cache.length() == 0 && fault_cache.block_table().empty() && fault_pool.used_block_count() == 0,
           "model fault injection did not abort paged prefill");
    Qwen3KvCache fault_reference(32);
    Tensor fault_expected = model.prefill_logits_with_cache(fault_ids, fault_reference);
    Tensor fault_actual = model.prefill_logits_paged(fault_ids, fault_cache);
    compare_logits(fault_actual, fault_expected, "fault recovery prefill");
    fault_cache.release_all();

    // Model-level rejection paths preserve a reusable empty cache.
    PagedKvCachePool reject_pool(pool_config(8));
    Qwen3PagedKvCache reject_cache(reject_pool, 4);
    expect_throw("empty prompt", [&] { model.prefill_logits_paged({}, reject_cache); });
    expect_throw("prompt length 33", [&] { model.prefill_logits_paged(prompt(33, 1), reject_cache); });
    expect_throw("prompt exceeds capacity", [&] { model.prefill_logits_paged(prompt(5, 2), reject_cache); });
    auto negative = prompt(1, 3); negative[0] = -1;
    auto too_large = prompt(1, 4); too_large[0] = 151936;
    expect_throw("negative token", [&] { model.prefill_logits_paged(negative, reject_cache); });
    expect_throw("vocab upper bound token", [&] { model.prefill_logits_paged(too_large, reject_cache); });
    reject_cache.begin_prefill(1);
    expect_throw("model prefill while prefill transaction active", [&] { model.prefill_logits_paged(prompt(1, 5), reject_cache); });
    reject_cache.abort_prefill();
    reject_cache.begin_decode();
    expect_throw("model prefill while decode transaction active", [&] { model.prefill_logits_paged(prompt(1, 6), reject_cache); });
    reject_cache.abort_decode();
    Tensor reject_ok = model.prefill_logits_paged(prompt(4, 7), reject_cache);
    expect(reject_ok.shape() == std::vector<int64_t>{4, 151936}, "reusable reject cache prefill failed");
    expect_throw("non-empty cache repeated prefill", [&] { model.prefill_logits_paged(prompt(1, 8), reject_cache); });
    reject_cache.release_all();
    PagedKvCachePoolConfig bad_config = pool_config(4); bad_config.num_layers = 27;
    expect_throw("incompatible pool config", [&] { PagedKvCachePool bad_pool(bad_config); Qwen3PagedKvCache bad_cache(bad_pool, 4); });

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "prefill transactions, fragmented blocks, fault recovery and rejection paths passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
