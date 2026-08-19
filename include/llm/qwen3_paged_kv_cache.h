#pragma once

#include "llm/paged_kv_cache_pool.h"
#include "llm/qwen3_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llm {

// Correctness/transition wrapper for the Phase 6 paged pool. The pool must
// outlive this cache; this object is not copyable, movable, or thread-safe.
class Qwen3PagedKvCache {
 public:
  Qwen3PagedKvCache(PagedKvCachePool& pool, std::size_t max_seq_len);
  ~Qwen3PagedKvCache() noexcept = default;
  Qwen3PagedKvCache(const Qwen3PagedKvCache&) = delete;
  Qwen3PagedKvCache& operator=(const Qwen3PagedKvCache&) = delete;
  Qwen3PagedKvCache(Qwen3PagedKvCache&&) = delete;
  Qwen3PagedKvCache& operator=(Qwen3PagedKvCache&&) = delete;

  std::size_t length() const noexcept { return length_; }
  std::size_t capacity() const noexcept { return capacity_; }
  bool in_decode_transaction() const noexcept { return transaction_ == Transaction::Decode; }
  bool in_prefill_transaction() const noexcept { return transaction_ == Transaction::Prefill; }
  const std::vector<BlockId>& block_table() const noexcept {
    return sequence_.block_table();
  }
  PagedKvCachePool& pool() noexcept { return *pool_; }
  const PagedKvCachePool& pool() const noexcept { return *pool_; }
  Tensor make_device_block_table_i32() const {
    return sequence_.make_device_block_table_i32();
  }
  void copy_block_table_to_cuda(Tensor& output) const {
    sequence_.copy_block_table_to_cuda(output);
  }

  void begin_decode();
  void append_layer_kv(std::size_t layer, const Tensor& key, const Tensor& value);
  void commit_decode();
  void abort_decode() noexcept;
  void begin_prefill(std::size_t token_count);
  void append_prefill_layer_kv(std::size_t layer, const Tensor& key,
                               const Tensor& value);
  void append_prefill_layer_kv_batched_slice(
      std::size_t layer, const Tensor& key_batched,
      const Tensor& value_batched, std::size_t batch_index,
      std::size_t batch_size, std::size_t seq_len);
  void append_prefill_layer_kv_batched_valid_slice(
      std::size_t layer, const Tensor& key_batched,
      const Tensor& value_batched, std::size_t batch_index,
      std::size_t batch_size, std::size_t max_seq_len,
      std::size_t valid_length);
  void commit_prefill();
  void abort_prefill() noexcept;
  void release_all() noexcept;

  void copy_layer_kv_to_host(std::size_t layer, std::size_t token,
                             std::uint16_t* key_bits,
                             std::uint16_t* value_bits,
                             std::size_t element_count) const;

  // Test/migration helper only. It performs CUDA device-to-device copies and
  // never changes the source contiguous cache.
  void seed_from_contiguous_cache(const Qwen3KvCache& source);

 private:
  enum class Transaction { None, Decode, Prefill };
  static constexpr std::size_t kLayers = 28;
  static constexpr std::size_t kKvHeads = 8;
  static constexpr std::size_t kHeadDim = 128;
  static constexpr std::size_t kMaxSequence = 32;

  void error(const std::string& message) const;
  void require_pending(const char* function) const;
  void require_qwen3_pool() const;
  void require_kv_tensor(const Tensor& tensor, const char* name) const;
  void require_prefill_tensor(const Tensor& tensor, const char* name,
                              std::size_t token_count) const;

  PagedKvCachePool* pool_ = nullptr;
  PagedSequenceKvCache sequence_;
  std::size_t capacity_ = 0;
  std::size_t length_ = 0;
  Transaction transaction_ = Transaction::None;
  std::size_t pending_position_ = 0;
  std::vector<bool> written_;
  std::size_t begin_length_ = 0;
  std::size_t begin_block_count_ = 0;
  std::size_t prefill_token_count_ = 0;
};

}  // namespace llm
