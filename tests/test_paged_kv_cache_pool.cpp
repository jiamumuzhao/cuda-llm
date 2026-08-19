#include "llm/cuda_check.h"
#include "llm/paged_kv_cache_pool.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& message) {
  throw std::runtime_error("test_paged_kv_cache_pool: " + message);
}

static void expect_throw(const std::function<void()>& fn, const std::string& name) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  fail(name + " did not throw");
}

static PagedKvCachePoolConfig small_config(std::size_t blocks = 8) {
  PagedKvCachePoolConfig config;
  config.total_blocks = blocks;
  config.num_layers = 2;
  config.num_kv_heads = 2;
  config.block_size = 4;
  config.head_dim = 4;
  config.dtype = DType::F16;
  return config;
}

static void check_pool_counts(const PagedKvCachePool& pool, std::size_t used) {
  if (pool.used_block_count() != used ||
      pool.free_block_count() != pool.total_blocks() - used)
    fail("pool free/used count mismatch");
}

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "SKIP test_paged_kv_cache_pool: no CUDA device\n";
    return 0;
  }
  try {
    PagedKvCachePool pool(small_config());
    const auto& storage = pool.storage();
    if (storage.device() != DeviceType::CUDA || storage.dtype() != DType::F16 ||
        !storage.is_contiguous())
      fail("small storage metadata mismatch");
    if (storage.shape() != std::vector<int64_t>({2, 8, 2, 4, 2, 4}))
      fail("small storage layout is not [layers,blocks,kv,block_size,heads,dim]");
    const std::size_t expected_token_bytes = 2 * 4 * sizeof(std::uint16_t);
    const std::size_t expected_plane_bytes = 4 * expected_token_bytes;
    const std::size_t expected_block_bytes = 2 * 2 * expected_plane_bytes;
    if (pool.token_bytes() != expected_token_bytes ||
        pool.kv_plane_bytes() != expected_plane_bytes ||
        pool.block_bytes() != expected_block_bytes ||
        pool.resident_bytes() != expected_block_bytes * 8)
      fail("small storage byte size mismatch");
    check_pool_counts(pool, 0);
    PagedKvCachePoolConfig qwen3 = small_config(2);
    qwen3.num_layers = 28;
    qwen3.num_kv_heads = 8;
    qwen3.block_size = 16;
    qwen3.head_dim = 128;
    PagedKvCachePool qwen3_pool(qwen3);
    if (qwen3_pool.block_bytes() != 1835008 ||
        qwen3_pool.resident_bytes() != qwen3.total_blocks * 1835008)
      fail("Qwen3 complete block_bytes/resident_bytes mismatch");
    PagedKvCachePool rollback_pool(small_config(2));
    std::cout << "storage shape=[2,8,2,4,2,4], Qwen3 complete block_bytes=1835008 passed\n";

    reset_cuda_allocation_stats();
    const CudaAllocationStats baseline = cuda_allocation_stats();
    PagedSequenceKvCache first(pool, 8);
    PagedSequenceKvCache second(pool, 8);
    first.append_tokens(5);
    second.append_tokens(5);
    check_pool_counts(pool, 4);
    std::set<BlockId> first_ids(first.block_table().begin(), first.block_table().end());
    std::set<BlockId> second_ids(second.block_table().begin(), second.block_table().end());
    if (first_ids.size() != first.block_count() || second_ids.size() != second.block_count())
      fail("sequence table contains duplicate blocks");
    for (BlockId id : first.block_table()) {
      if (second_ids.count(id) != 0 || !pool.block_manager().is_allocated(id) ||
          pool.block_manager().ref_count(id) != 1)
        fail("sequence physical blocks overlap");
    }
    std::cout << "two sequence physical block isolation passed\n";

    const std::size_t elements = pool.config().num_kv_heads * pool.config().head_dim;
    const std::vector<std::uint16_t> k_bits{0x0001, 0x03ff, 0x7c00, 0x8000,
                                            0x3555, 0x7e00, 0x0400, 0x1234};
    const std::vector<std::uint16_t> k_bits_offset1{0x9001, 0x13ff, 0x5c00, 0x0000,
                                                     0x4555, 0x3e00, 0x1400, 0x2234};
    const std::vector<std::uint16_t> v_bits{0x1111, 0x2222, 0x3333, 0x4444,
                                            0x5555, 0x6666, 0x7777, 0x8888};
    const std::vector<std::uint16_t> v_bits_offset1{0xaaaa, 0xbbbb, 0xcccc, 0xdddd,
                                                     0xeeee, 0xffff, 0x9999, 0x8888};
    const std::vector<std::uint16_t> layer1_bits(elements, 0x4a4a);
    const std::vector<std::uint16_t> second_bits(elements, 0x5b5b);
    first.copy_token_from_host(0, 0, PagedKvKind::Key, k_bits.data(), elements);
    first.copy_token_from_host(0, 1, PagedKvKind::Key, k_bits_offset1.data(), elements);
    first.copy_token_from_host(0, 0, PagedKvKind::Value, v_bits.data(), elements);
    first.copy_token_from_host(0, 1, PagedKvKind::Value, v_bits_offset1.data(), elements);
    first.copy_token_from_host(1, 0, PagedKvKind::Key, layer1_bits.data(), elements);
    second.copy_token_from_host(0, 0, PagedKvKind::Key, second_bits.data(), elements);
    first.copy_token_from_host(1, 4, PagedKvKind::Value, v_bits.data(), elements);
    std::vector<std::uint16_t> read_k(elements), read_k_offset1(elements), read_v(elements),
        read_v_offset1(elements), read_v_layer1(elements), read_layer1(elements),
        read_second(elements);
    first.copy_token_to_host(0, 0, PagedKvKind::Key, read_k.data(), elements);
    first.copy_token_to_host(0, 1, PagedKvKind::Key, read_k_offset1.data(), elements);
    first.copy_token_to_host(0, 0, PagedKvKind::Value, read_v.data(), elements);
    first.copy_token_to_host(0, 1, PagedKvKind::Value, read_v_offset1.data(), elements);
    first.copy_token_to_host(1, 0, PagedKvKind::Key, read_layer1.data(), elements);
    second.copy_token_to_host(0, 0, PagedKvKind::Key, read_second.data(), elements);
    first.copy_token_to_host(1, 4, PagedKvKind::Value, read_v_layer1.data(), elements);
    if (read_k != k_bits || read_k_offset1 != k_bits_offset1 || read_v != v_bits ||
        read_v_offset1 != v_bits_offset1 || read_v_layer1 != v_bits ||
        read_layer1 != layer1_bits ||
        read_second != second_bits)
      fail("FP16 K/V/layer/sequence copy was not bit-exact");
    if (static_cast<const char*>(first.device_token_ptr(0, 1, PagedKvKind::Key)) -
            static_cast<const char*>(first.device_token_ptr(0, 0, PagedKvKind::Key)) !=
        static_cast<std::ptrdiff_t>(pool.token_bytes()))
      fail("token pointer stride is not token_bytes");
    std::vector<std::uint16_t> untouched(elements, 0);
    first.copy_token_to_host(0, 0, PagedKvKind::Value, untouched.data(), elements);
    if (untouched != v_bits)
      fail("K write polluted V storage");
    if (read_layer1 != layer1_bits || read_second != second_bits)
      fail("layer or sequence write polluted another region");
    std::fill(untouched.begin(), untouched.end(), 0);
    first.copy_token_to_host(0, 0, PagedKvKind::Key, untouched.data(), elements);
    if (untouched != k_bits) fail("sequence write polluted another sequence");
    std::cout << "token stride=" << pool.token_bytes()
              << " and explicit K/V/layer/sequence sentinel isolation passed\n";

    expect_throw([&] { pool.device_block_ptr(2, first.block_table().front(), PagedKvKind::Key); }, "invalid layer");
    expect_throw([&] { pool.device_token_ptr(0, first.block_table().front(), PagedKvKind::Key, 4); }, "invalid offset");
    expect_throw([&] { first.copy_token_from_host(0, 0, PagedKvKind::Key, k_bits.data(), elements - 1); }, "wrong element count");
    expect_throw([&] { first.copy_token_to_host(0, 7, PagedKvKind::Key, read_k.data(), elements); }, "unappended token");
    const BlockId free_id = first.block_table().back();
    first.release_all();
    expect_throw([&] { pool.device_block_ptr(0, free_id, PagedKvKind::Key); }, "free block pointer");
    check_pool_counts(pool, 2);
    if (!first.block_table().empty())
      fail("released sequence retained a block table entry");
    for (BlockId id : second.block_table()) {
      if (!pool.block_manager().is_allocated(id) ||
          pool.block_manager().ref_count(id) != 1)
        fail("releasing first sequence affected second sequence block");
    }
    second.release_all();
    check_pool_counts(pool, 0);
    PagedSequenceKvCache reused(pool, 4);
    reused.append_tokens(1);
    std::vector<std::uint16_t> reuse_bits(elements, 0xabcd), reuse_read(elements);
    reused.copy_token_from_host(0, 0, PagedKvKind::Value, reuse_bits.data(), elements);
    reused.copy_token_to_host(0, 0, PagedKvKind::Value, reuse_read.data(), elements);
    if (reuse_read != reuse_bits) fail("reused block write/read mismatch");
    reused.release_all();
    std::cout << "release/reuse and pointer/copy rejection paths passed\n";

    PagedSequenceKvCache rollback(rollback_pool, 12);
    rollback.append_tokens(4);
    const auto old_table = rollback.block_table();
    const std::size_t old_free = rollback_pool.free_block_count();
    expect_throw([&] { rollback.append_tokens(5); }, "pool exhaustion append");
    if (rollback.token_count() != 4 || rollback.block_table() != old_table ||
        rollback_pool.free_block_count() != old_free || rollback_pool.used_block_count() != 1)
      fail("paged sequence append did not roll back atomically");
    rollback.release_all();
    std::cout << "GPU pool exhaustion transactional rollback passed\n";

    const CudaAllocationStats steady = cuda_allocation_stats();
    if (steady.cuda_malloc_calls != baseline.cuda_malloc_calls ||
        steady.cuda_free_calls != baseline.cuda_free_calls ||
        steady.cuda_allocated_bytes_total != baseline.cuda_allocated_bytes_total ||
        steady.cuda_freed_bytes_total != baseline.cuda_freed_bytes_total)
      fail("sequence append/release/copy caused extra CUDA allocations/frees");
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "steady-state allocation stats malloc/free=0 after pool reset passed\n";
    std::cout << "test_paged_kv_cache_pool passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
