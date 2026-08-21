#pragma once

#include "llm/paged_kv_block_manager.h"
#include "llm/tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llm {

enum class PagedKvKind { Key = 0, Value = 1 };

struct PagedKvCachePoolConfig {
  std::size_t total_blocks = 0;
  std::size_t num_layers = 0;
  std::size_t num_kv_heads = 0;
  std::size_t block_size = 0;
  std::size_t head_dim = 0;
  DType dtype = DType::F16;
};

// Owns one stable CUDA allocation and the CPU block metadata. This object is
// CPU-metadata-thread-unsafe by design; it is not part of the inference path.
class PagedKvCachePool {
 public:
  explicit PagedKvCachePool(const PagedKvCachePoolConfig& config);
  ~PagedKvCachePool() = default;
  PagedKvCachePool(const PagedKvCachePool&) = delete;
  PagedKvCachePool& operator=(const PagedKvCachePool&) = delete;
  PagedKvCachePool(PagedKvCachePool&&) = delete;
  PagedKvCachePool& operator=(PagedKvCachePool&&) = delete;

  BlockManager& block_manager() noexcept { return block_manager_; }
  const BlockManager& block_manager() const noexcept { return block_manager_; }
  Tensor& storage() noexcept { return storage_; }
  const Tensor& storage() const noexcept { return storage_; }
  std::size_t token_bytes() const noexcept { return token_bytes_; }
  std::size_t kv_plane_bytes() const noexcept { return kind_bytes_; }
  std::size_t block_bytes() const noexcept { return block_bytes_; }
  std::size_t resident_bytes() const noexcept { return storage_.nbytes(); }
  std::size_t total_blocks() const noexcept { return config_.total_blocks; }
  std::size_t free_block_count() const noexcept { return block_manager_.free_block_count(); }
  std::size_t used_block_count() const noexcept { return block_manager_.used_block_count(); }

  void* device_block_ptr(std::size_t layer, BlockId block_id, PagedKvKind kind);
  const void* device_block_ptr(std::size_t layer, BlockId block_id,
                               PagedKvKind kind) const;
  void* device_token_ptr(std::size_t layer, BlockId block_id, PagedKvKind kind,
                         std::size_t offset);
  const void* device_token_ptr(std::size_t layer, BlockId block_id,
                               PagedKvKind kind, std::size_t offset) const;

  const PagedKvCachePoolConfig& config() const noexcept { return config_; }

 private:
  std::size_t kind_offset_bytes(std::size_t layer, BlockId block_id,
                                PagedKvKind kind) const;
  void validate_location(std::size_t layer, BlockId block_id,
                        PagedKvKind kind, std::size_t offset,
                        bool check_offset) const;

  PagedKvCachePoolConfig config_;
  BlockManager block_manager_;
  Tensor storage_;
  std::size_t block_bytes_ = 0;
  std::size_t token_bytes_ = 0;
  std::size_t kind_bytes_ = 0;
};

// Owns one reference to each block in its logical table. The pool must outlive
// every PagedSequenceKvCache that refers to it. No shared ownership or locks
// are used in Phase 6.2.
class PagedSequenceKvCache {
 public:
  PagedSequenceKvCache(PagedKvCachePool& pool, std::size_t max_tokens);
  ~PagedSequenceKvCache() noexcept = default;
  PagedSequenceKvCache(const PagedSequenceKvCache&) = delete;
  PagedSequenceKvCache& operator=(const PagedSequenceKvCache&) = delete;
  PagedSequenceKvCache(PagedSequenceKvCache&&) = delete;
  PagedSequenceKvCache& operator=(PagedSequenceKvCache&&) = delete;

  void append_tokens(std::size_t count) { table_.append_tokens(count); }
  void release_all() noexcept { table_.release_all(); }
  void rollback_to(std::size_t token_count, std::size_t block_count) {
    table_.rollback_to(token_count, block_count);
  }
  std::size_t token_count() const noexcept { return table_.token_count(); }
  std::size_t block_count() const noexcept { return table_.block_count(); }
  const std::vector<BlockId>& block_table() const noexcept { return table_.block_table(); }
  PhysicalLocation physical_location(std::size_t logical_token) const {
    return table_.locate(logical_token);
  }
  void* device_token_ptr(std::size_t layer, std::size_t logical_token,
                         PagedKvKind kind);
  const void* device_token_ptr(std::size_t layer, std::size_t logical_token,
                               PagedKvKind kind) const;

  void copy_token_from_host(std::size_t layer, std::size_t logical_token,
                            PagedKvKind kind, const std::uint16_t* fp16_bits,
                            std::size_t element_count);
  void copy_token_to_host(std::size_t layer, std::size_t logical_token,
                          PagedKvKind kind, std::uint16_t* fp16_bits,
                          std::size_t element_count) const;
  Tensor make_device_block_table_i32() const;
  void copy_block_table_to_cuda(Tensor& output) const;

 private:
  void validate_copy_count(const std::uint16_t* bits,
                           std::size_t element_count) const;

  PagedKvCachePool* pool_ = nullptr;
  SequenceBlockTable table_;
};

}  // namespace llm
