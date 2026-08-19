#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_paged_kv_cache.h"

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
  throw std::runtime_error("test_qwen3_paged_decode: " + message);
}

static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); }
  catch (const std::exception& error) {
    std::cout << "rejected " << name << ": " << error.what() << "\n";
    return;
  }
  fail(name + " was accepted");
}

static std::vector<int32_t> prompt(size_t length, int salt) {
  std::vector<int32_t> ids(length);
  for (size_t i = 0; i < length; ++i)
    ids[i] = 1 + static_cast<int32_t>((salt + i * 37) % 150000);
  return ids;
}

static std::vector<int32_t> top5(const Tensor& logits) {
  Tensor cpu = logits.to(DeviceType::CPU);
  std::vector<int32_t> ids(151936);
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + 5, ids.end(),
                    [&](int32_t a, int32_t b) {
                      const float va = cpu.get_f32(static_cast<size_t>(a));
                      const float vb = cpu.get_f32(static_cast<size_t>(b));
                      return va != vb ? va > vb : a < b;
                    });
  ids.resize(5);
  return ids;
}

static void compare_logits(const Tensor& actual, const Tensor& expected,
                           const std::string& name) {
  Tensor a = actual.to(DeviceType::CPU);
  Tensor e = expected.to(DeviceType::CPU);
  double dot = 0.0, aa = 0.0, ee = 0.0, mean = 0.0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < a.numel(); ++i) {
    const float av = a.get_f32(i), ev = e.get_f32(i);
    const float error = std::fabs(av - ev);
    max_abs = std::max(max_abs, error);
    mean += error;
    dot += double(av) * ev;
    aa += double(av) * av;
    ee += double(ev) * ev;
  }
  const double cosine = dot / std::sqrt(aa * ee);
  std::cout << name << " max_abs_error=" << max_abs
            << " mean_abs_error=" << mean / a.numel()
            << " cosine_similarity=" << cosine << " tolerance=5e-3\n";
  if (max_abs > 5e-3f || cosine < 0.999)
    fail(name + " logits mismatch");
}

static PagedKvCachePoolConfig pool_config(size_t total_blocks) {
  PagedKvCachePoolConfig config;
  config.total_blocks = total_blocks;
  config.num_layers = 28;
  config.num_kv_heads = 8;
  config.block_size = 16;
  config.head_dim = 128;
  config.dtype = DType::F16;
  return config;
}

static Tensor dummy_kv(std::uint16_t bits) {
  Tensor host(DType::F16, {8, 128});
  for (size_t i = 0; i < host.numel(); ++i)
    host.data_f16()[i] = static_cast<std::uint16_t>(bits + i * 7);
  return host.to(DeviceType::CUDA);
}

static void fill_one(Qwen3PagedKvCache& cache, const Tensor& key,
                     const Tensor& value) {
  cache.begin_decode();
  for (size_t layer = 0; layer < 28; ++layer)
    cache.append_layer_kv(layer, key, value);
  cache.commit_decode();
}

static void compare_new_token_kv_bits(const Qwen3PagedKvCache& paged,
                                      const Qwen3KvCache& contiguous,
                                      size_t layer, size_t context) {
  std::vector<std::uint16_t> paged_k(1024), paged_v(1024);
  paged.copy_layer_kv_to_host(layer, context, paged_k.data(), paged_v.data(), 1024);
  Tensor contiguous_k = contiguous.key_cache(layer).to(DeviceType::CPU);
  Tensor contiguous_v = contiguous.value_cache(layer).to(DeviceType::CPU);
  for (size_t i = 0; i < 1024; ++i) {
    if (paged_k[i] != contiguous_k.data_f16()[context * 1024 + i])
      fail("new K bit mismatch context=" + std::to_string(context) +
           " layer=" + std::to_string(layer) + " kind=K element=" +
           std::to_string(i));
    if (paged_v[i] != contiguous_v.data_f16()[context * 1024 + i])
      fail("new V bit mismatch context=" + std::to_string(context) +
           " layer=" + std::to_string(layer) + " kind=V element=" +
           std::to_string(i));
  }
}

static void test_case(Qwen3CudaModel& model, size_t context) {
  const auto ids = prompt(context, 19 + static_cast<int>(context));
  Qwen3KvCache contiguous(32);
  model.prefill_logits_with_cache(ids, contiguous);

  PagedKvCachePool pool(pool_config(6));
  Tensor key = dummy_kv(0x1200), value = dummy_kv(0x5200);
  Qwen3PagedKvCache interposer_a(pool, 32), interposer_b(pool, 32);
  fill_one(interposer_a, key, value);
  fill_one(interposer_b, key, value);
  interposer_a.release_all();

  Qwen3PagedKvCache paged(pool, 32);
  paged.seed_from_contiguous_cache(contiguous);
  {
    std::vector<std::uint16_t> seeded_k(1024), seeded_v(1024);
    Tensor source_k = contiguous.key_cache(7).to(DeviceType::CPU);
    Tensor source_v = contiguous.value_cache(7).to(DeviceType::CPU);
    for (size_t token : {size_t(0), context - 1}) {
      paged.copy_layer_kv_to_host(7, token, seeded_k.data(), seeded_v.data(), 1024);
      for (size_t i = 0; i < 1024; ++i)
        if (seeded_k[i] != source_k.data_f16()[token * 1024 + i] ||
            seeded_v[i] != source_v.data_f16()[token * 1024 + i])
          fail("seed K/V mismatch before decode at context=" + std::to_string(context));
    }
  }
  const int32_t next = 700 + static_cast<int32_t>(context);
  Tensor contiguous_logits = model.decode_logits(next, contiguous);
  Tensor paged_logits = model.decode_logits_paged(next, paged);
  if (paged.length() != context + 1 || contiguous.length() != context + 1)
    fail("cache length mismatch at context=" + std::to_string(context) +
         " paged=" + std::to_string(paged.length()) +
         " contiguous=" + std::to_string(contiguous.length()));
  if (context >= 16 && paged.block_table().size() != 2)
    fail("context " + std::to_string(context) + " did not cross a block boundary");
  if (context >= 16 && paged.block_table()[0] == paged.block_table()[1])
    fail("context " + std::to_string(context) + " reused a physical block");
  compare_new_token_kv_bits(paged, contiguous, 7, context);
  compare_new_token_kv_bits(paged, contiguous, 27, context);
  compare_logits(paged_logits, contiguous_logits,
                 "paged decode context=" + std::to_string(context));
  if (top5(paged_logits) != top5(contiguous_logits))
    fail("top-5 mismatch at context=" + std::to_string(context));
  CudaSampler sampler(151936);
  SamplingConfig sampling;
  sampling.temperature = 0.8f; sampling.top_k = 50; sampling.top_p = 0.9f;
  sampling.seed = 20260727;
  if (sampler.sample_last_row(paged_logits, sampling, 11) !=
      sampler.sample_last_row(contiguous_logits, sampling, 11))
    fail("sampling mismatch at context=" + std::to_string(context));

  std::cout << "paged decode context=" << context
            << " non-contiguous physical table and layer 7/27 K/V bit-exact copy passed\n";
  interposer_b.release_all();
  paged.release_all();
}

static void test_attention_addressing() {
  Qwen3KvCache contiguous(8);
  Tensor key_host(DType::F16, {4, 8, 128});
  Tensor value_host(DType::F16, {4, 8, 128});
  for (size_t i = 0; i < key_host.numel(); ++i) {
    key_host.set_f32(i, 0.01f * float(static_cast<int>(i % 31) - 15));
    value_host.set_f32(i, 0.02f * float(static_cast<int>(i % 23) - 11));
  }
  Tensor key = key_host.to(DeviceType::CUDA), value = value_host.to(DeviceType::CUDA);
  for (size_t layer = 0; layer < 28; ++layer)
    contiguous.write_prefill_layer(layer, key, value, 4);
  contiguous.commit_prefill(4);
  Tensor q_host(DType::F16, {1, 16, 128});
  for (size_t i = 0; i < q_host.numel(); ++i)
    q_host.set_f32(i, 0.01f * float(static_cast<int>(i % 29) - 14));
  Tensor q = q_host.to(DeviceType::CUDA);
  Tensor expected = cuda_gqa_decode_attention(q, contiguous.key_cache(0),
                                              contiguous.value_cache(0), 4);
  PagedKvCachePool pool(pool_config(2));
  Qwen3PagedKvCache paged(pool, 8);
  paged.seed_from_contiguous_cache(contiguous);
  Tensor table = paged.make_device_block_table_i32();
  Tensor actual = cuda_paged_gqa_attention_decode(
      q.reshape({16, 128}), pool, 0, table, 4, 16, 8, 128);
  compare_logits(actual.reshape({1, 16, 128}), expected,
                 "direct paged attention addressing");
  paged.release_all();
}

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "SKIP test_qwen3_paged_decode: no CUDA device\n";
    return 0;
  }
  try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability="
              << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    test_attention_addressing();
    for (size_t context : {size_t(4), size_t(15), size_t(16), size_t(17), size_t(31)})
      test_case(model, context);

    expect_throw("invalid token", [&] {
      PagedKvCachePool pool(pool_config(2)); Qwen3PagedKvCache cache(pool, 32);
      Qwen3KvCache source(4); model.prefill_logits_with_cache(prompt(4, 91), source);
      cache.seed_from_contiguous_cache(source); model.decode_logits_paged(151936, cache);
    });
    expect_throw("empty cache", [&] {
      PagedKvCachePool pool(pool_config(2)); Qwen3PagedKvCache cache(pool, 32);
      model.decode_logits_paged(1, cache);
    });
    expect_throw("active transaction", [&] {
      PagedKvCachePool pool(pool_config(2)); Qwen3PagedKvCache cache(pool, 32);
      Qwen3KvCache source(4); model.prefill_logits_with_cache(prompt(4, 92), source);
      cache.seed_from_contiguous_cache(source); cache.begin_decode();
      model.decode_logits_paged(1, cache);
    });
    expect_throw("full cache", [&] {
      PagedKvCachePool pool(pool_config(2)); Qwen3PagedKvCache cache(pool, 32);
      Qwen3KvCache source(32); model.prefill_logits_with_cache(prompt(32, 93), source);
      cache.seed_from_contiguous_cache(source); model.decode_logits_paged(1, cache);
    });
    expect_throw("physical block exhaustion", [&] {
      PagedKvCachePool pool(pool_config(1)); Qwen3PagedKvCache cache(pool, 32);
      Qwen3KvCache source(16); model.prefill_logits_with_cache(prompt(16, 94), source);
      cache.seed_from_contiguous_cache(source); model.decode_logits_paged(1, cache);
    });

    // Multi-step decode compares the paged and contiguous state after every
    // append, including the 16->17 physical block transition.
    Qwen3KvCache serial(32);
    model.prefill_logits_with_cache(prompt(16, 101), serial);
    PagedKvCachePool pool(pool_config(5));
    Qwen3PagedKvCache paged(pool, 32);
    paged.seed_from_contiguous_cache(serial);
    for (size_t step = 0; step < 3; ++step) {
      const int32_t token = 800 + static_cast<int32_t>(step);
      Tensor a = model.decode_logits(token, serial);
      Tensor b = model.decode_logits_paged(token, paged);
      compare_logits(b, a, "multi-step decode step=" + std::to_string(step));
      if (serial.length() != paged.length()) fail("multi-step cache length mismatch");
    }
    paged.release_all();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "multi-step paged decode, rejection paths, and CUDA health check passed\n";
    std::cout << "test_qwen3_paged_decode passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
