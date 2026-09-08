#include "llm/cuda_check.h"
#include "llm/qwen3_kv_cache.h"
#include "llm/qwen3_paged_kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& message) {
  throw std::runtime_error("test_qwen3_paged_kv_cache: " + message);
}
static void expect_throw(const std::function<void()>& fn, const std::string& name) {
  try { fn(); } catch (const std::exception&) { return; }
  fail(name + " did not throw");
}
static void expect_throw_contains(const std::function<void()>& fn,
                                  const std::string& name,
                                  const std::string& expected_text) {
  try {
    fn();
  } catch (const std::exception& error) {
    if (std::string(error.what()).find(expected_text) == std::string::npos)
      fail(name + " wrong error: " + error.what());
    return;
  }
  fail(name + " did not throw");
}
static PagedKvCachePoolConfig config(std::size_t blocks = 4) {
  PagedKvCachePoolConfig c;
  c.total_blocks = blocks; c.num_layers = 28; c.num_kv_heads = 8;
  c.block_size = 16; c.head_dim = 128; c.dtype = DType::F16;
  return c;
}
static Tensor pattern(std::uint16_t base) {
  Tensor host(DType::F16, {8, 128});
  for (std::size_t i = 0; i < host.numel(); ++i)
    host.data_f16()[i] = static_cast<std::uint16_t>(base + i * 13u);
  return host.to(DeviceType::CUDA);
}
static Tensor contiguous_pattern(std::uint16_t base, std::size_t length) {
  Tensor host(DType::F16, {static_cast<int64_t>(length), 8, 128});
  for (std::size_t i = 0; i < host.numel(); ++i)
    host.data_f16()[i] = static_cast<std::uint16_t>(base + i * 7u);
  return host;
}
static void expect_bits(const std::vector<std::uint16_t>& actual,
                        std::uint16_t base, const std::string& name) {
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (actual[i] != static_cast<std::uint16_t>(base + i * 13u))
      fail(name + " bit mismatch at " + std::to_string(i));
}
static void append_all(Qwen3PagedKvCache& cache, const Tensor& key,
                       const Tensor& value) {
  cache.begin_decode();
  for (std::size_t layer = 0; layer < 28; ++layer)
    cache.append_layer_kv(layer, key, value);
  cache.commit_decode();
}

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "SKIP test_qwen3_paged_kv_cache: no CUDA device\n";
    return 0;
  }
  try {
    PagedKvCachePool pool(config());
    expect_throw([&] { Qwen3PagedKvCache bad(pool, 0); }, "zero capacity");
    expect_throw([&] { Qwen3PagedKvCache bad(pool, 513); }, "capacity above 512");
    PagedKvCachePoolConfig wrong = config(); wrong.num_layers = 27;
    PagedKvCachePool wrong_pool(wrong);
    expect_throw([&] { Qwen3PagedKvCache bad(wrong_pool, 4); }, "wrong Qwen dimensions");

    Qwen3PagedKvCache cache(pool, 32);
    Tensor key = pattern(0x1000), value = pattern(0x4000);
    reset_cuda_allocation_stats();
    const CudaAllocationStats baseline = cuda_allocation_stats();
    cache.begin_decode();
    if (!cache.in_decode_transaction() || cache.length() != 0 || cache.block_table().size() != 1)
      fail("begin_decode state mismatch");
    expect_throw([&] { cache.copy_layer_kv_to_host(0, 0, nullptr, nullptr, 1024); },
                 "read pending token");
    expect_throw([&] { cache.append_layer_kv(0, key, value); cache.append_layer_kv(0, key, value); },
                 "duplicate layer");
    expect_throw([&] { cache.commit_decode(); }, "partial commit");
    cache.abort_decode();
    if (cache.length() != 0 || !cache.block_table().empty() || cache.in_decode_transaction())
      fail("abort did not restore empty state");
    if (pool.used_block_count() != 0) fail("abort did not release pending block");
    expect_throw([&] { cache.append_layer_kv(0, key, value); }, "append outside transaction");
    expect_throw([&] { cache.begin_decode(); cache.begin_decode(); }, "duplicate begin");
    cache.abort_decode();

    std::vector<Tensor> transaction_keys;
    std::vector<Tensor> transaction_values;
    transaction_keys.reserve(32 * 28);
    transaction_values.reserve(32 * 28);
    for (std::size_t token = 0; token < 32; ++token)
      for (std::size_t layer = 0; layer < 28; ++layer) {
        transaction_keys.push_back(pattern(static_cast<std::uint16_t>(0x1000 + token * 17 + layer)));
        transaction_values.push_back(pattern(static_cast<std::uint16_t>(0x4000 + token * 19 + layer)));
      }
    reset_cuda_allocation_stats();
    const CudaAllocationStats transaction_baseline = cuda_allocation_stats();
    for (std::size_t token = 0; token < 17; ++token) {
      cache.begin_decode();
      for (std::size_t layer = 0; layer < 28; ++layer) {
        const std::size_t input = token * 28 + layer;
        cache.append_layer_kv(layer, transaction_keys[input], transaction_values[input]);
      }
      cache.commit_decode();
      if (cache.length() != token + 1 || cache.block_table().size() != (token + 16) / 16)
        fail("length/block transition mismatch at token " + std::to_string(token));
    }
    std::vector<std::uint16_t> read_k(1024), read_v(1024);
    cache.copy_layer_kv_to_host(27, 0, read_k.data(), read_v.data(), read_k.size());
    expect_bits(read_k, 0x1000 + 27, "token 0 key");
    expect_bits(read_v, 0x4000 + 27, "token 0 value");
    cache.copy_layer_kv_to_host(3, 16, read_k.data(), read_v.data(), read_k.size());
    expect_bits(read_k, static_cast<std::uint16_t>(0x1000 + 16 * 17 + 3), "token 16 key");
    expect_bits(read_v, static_cast<std::uint16_t>(0x4000 + 16 * 19 + 3), "token 16 value");
    const std::size_t used_before_full = pool.used_block_count();
    while (cache.length() < cache.capacity()) {
      cache.begin_decode();
      for (std::size_t layer = 0; layer < 28; ++layer) cache.append_layer_kv(layer, key, value);
      cache.commit_decode();
    }
    expect_throw([&] { cache.begin_decode(); }, "begin at capacity");
    if (cache.length() != 32 || pool.used_block_count() < used_before_full)
      fail("capacity state mismatch");
    cache.release_all();
    if (cache.length() != 0 || !cache.block_table().empty() || pool.used_block_count() != 0)
      fail("release_all did not return blocks");
    cache.release_all();

    const CudaAllocationStats steady = cuda_allocation_stats();
    if (steady.cuda_malloc_calls != transaction_baseline.cuda_malloc_calls ||
        steady.cuda_free_calls != transaction_baseline.cuda_free_calls)
      fail("decode transaction path allocated/freed CUDA Tensor storage");
    std::cout << "begin/append-all-28/commit/abort/capacity and zero transaction allocations passed\n";

    Qwen3KvCache source(4);
    Tensor source_k_host = contiguous_pattern(0x2200, 2);
    Tensor source_v_host = contiguous_pattern(0x6600, 2);
    Tensor source_k = source_k_host.to(DeviceType::CUDA);
    Tensor source_v = source_v_host.to(DeviceType::CUDA);
    for (std::size_t layer = 0; layer < 28; ++layer)
      source.write_prefill_layer(layer, source_k, source_v, 2);
    source.commit_prefill(2);
    Qwen3PagedKvCache seeded(pool, 8);
    const size_t source_length = source.length();
    seeded.seed_from_contiguous_cache(source);
    if (seeded.length() != source_length || source.length() != source_length)
      fail("seed length/source state mismatch");
    seeded.copy_layer_kv_to_host(4, 1, read_k.data(), read_v.data(), read_k.size());
    for (std::size_t i = 0; i < read_k.size(); ++i)
      if (read_k[i] != source_k_host.data_f16()[read_k.size() + i] ||
          read_v[i] != source_v_host.data_f16()[read_v.size() + i])
        fail("seed bit pattern mismatch");
    expect_throw([&] { seeded.seed_from_contiguous_cache(source); }, "seed into non-empty cache");
    seeded.release_all();
    Qwen3PagedKvCache active_seed_target(pool, 8);
    active_seed_target.begin_decode();
    expect_throw([&] { active_seed_target.seed_from_contiguous_cache(source); },
                 "seed with active transaction");
    active_seed_target.abort_decode();
    Qwen3PagedKvCache small_seed_target(pool, 2);
    expect_throw([&] { small_seed_target.seed_from_contiguous_cache(source); },
                 "source capacity exceeds target capacity");
    if (small_seed_target.length() != 0 || small_seed_target.block_table().size() != 0)
      fail("rejected seed changed target state");

    // Independent one-block pool: a committed blocker owns the only physical
    // block, so seed must fail while preserving both source and blocker.
    PagedKvCachePool exhausted_pool(config(1));
    Qwen3PagedKvCache blocker(exhausted_pool, 16);
    append_all(blocker, key, value);
    if (exhausted_pool.used_block_count() != 1 || exhausted_pool.free_block_count() != 0)
      fail("pool exhaustion setup did not commit the blocker block");
    const std::vector<BlockId> blocker_table = blocker.block_table();
    const std::size_t blocker_length = blocker.length();
    std::vector<std::uint16_t> blocker_k(1024), blocker_v(1024);
    blocker.copy_layer_kv_to_host(7, 0, blocker_k.data(), blocker_v.data(), blocker_k.size());
    expect_bits(blocker_k, 0x1000, "blocker key before failed seed");
    expect_bits(blocker_v, 0x4000, "blocker value before failed seed");
    Qwen3PagedKvCache seed_target(exhausted_pool, 4);
    expect_throw_contains([&] { seed_target.seed_from_contiguous_cache(source); },
                           "seed with exhausted physical pool", "block pool exhausted");
    if (seed_target.length() != 0 || !seed_target.block_table().empty() ||
        seed_target.in_decode_transaction())
      fail("pool-exhausted seed did not fully roll back target");
    if (exhausted_pool.used_block_count() != 1 || exhausted_pool.free_block_count() != 0 ||
        blocker.length() != blocker_length || blocker.block_table() != blocker_table)
      fail("pool-exhausted seed changed blocker or pool ownership");
    blocker.copy_layer_kv_to_host(7, 0, blocker_k.data(), blocker_v.data(), blocker_k.size());
    expect_bits(blocker_k, 0x1000, "blocker key after failed seed");
    expect_bits(blocker_v, 0x4000, "blocker value after failed seed");
    if (source.length() != source_length)
      fail("pool-exhausted seed changed source length");
    Tensor source_check = source.key_cache(4).to(DeviceType::CPU);
    for (std::size_t i = 0; i < read_k.size(); ++i)
      if (source_check.get_f32(read_k.size() + i) != source_k_host.get_f32(read_k.size() + i))
        fail("pool-exhausted seed changed source content");
    blocker.release_all();
    if (exhausted_pool.used_block_count() != 0 || exhausted_pool.free_block_count() != 1)
      fail("blocker release did not return exhausted pool block");
    Qwen3PagedKvCache recovered_target(exhausted_pool, 4);
    recovered_target.seed_from_contiguous_cache(source);
    if (recovered_target.length() != source_length || recovered_target.block_table().size() != 1)
      fail("seed did not succeed after pool block recovery");
    recovered_target.copy_layer_kv_to_host(4, 1, read_k.data(), read_v.data(), read_k.size());
    for (std::size_t i = 0; i < read_k.size(); ++i)
      if (read_k[i] != source_k_host.data_f16()[read_k.size() + i] ||
          read_v[i] != source_v_host.data_f16()[read_v.size() + i])
        fail("recovered seed bit pattern mismatch");
    recovered_target.release_all();
    if (exhausted_pool.used_block_count() != 0)
      fail("recovered seed did not release its block");
    std::cout << "pool exhaustion -> seed rollback -> blocker release -> seed recovery passed\n";
    std::cout << "contiguous Qwen3KvCache D2D seed and source preservation passed\n";
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_paged_kv_cache passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
