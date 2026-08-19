#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm {

using BlockId = std::uint32_t;
constexpr BlockId kInvalidBlockId = static_cast<BlockId>(~BlockId{0});

class BlockPoolExhausted : public std::runtime_error {
 public:
  explicit BlockPoolExhausted(const std::string& message)
      : std::runtime_error(message) {}
};

// CPU-only Phase 6.1 metadata. This class is intentionally not thread-safe.
class BlockManager {
 public:
  explicit BlockManager(std::size_t total_blocks);
  ~BlockManager() = default;
  BlockManager(const BlockManager&) = delete;
  BlockManager& operator=(const BlockManager&) = delete;
  BlockManager(BlockManager&&) = delete;
  BlockManager& operator=(BlockManager&&) = delete;

  BlockId allocate();
  void retain(BlockId block_id);
  void release(BlockId block_id);

  std::size_t total_block_count() const noexcept { return total_blocks_; }
  std::size_t free_block_count() const noexcept { return free_blocks_.size(); }
  std::size_t used_block_count() const noexcept { return total_blocks_ - free_blocks_.size(); }
  std::size_t ref_count(BlockId block_id) const;
  bool is_allocated(BlockId block_id) const;

 private:
  void validate_id(BlockId block_id) const;

  std::size_t total_blocks_ = 0;
  std::vector<BlockId> free_blocks_;
  std::vector<std::size_t> ref_counts_;
  std::vector<bool> allocated_;
};

struct PhysicalLocation {
  BlockId block_id = kInvalidBlockId;
  std::size_t offset = 0;
};

// One logical sequence's CPU block table. It owns one reference to every
// block in block_table_. Prefix sharing/COW are deliberately out of scope.
class SequenceBlockTable {
 public:
  SequenceBlockTable(BlockManager& manager, std::size_t block_size,
                     std::size_t max_tokens);
  ~SequenceBlockTable() noexcept;
  SequenceBlockTable(const SequenceBlockTable&) = delete;
  SequenceBlockTable& operator=(const SequenceBlockTable&) = delete;
  SequenceBlockTable(SequenceBlockTable&&) = delete;
  SequenceBlockTable& operator=(SequenceBlockTable&&) = delete;

  void append_tokens(std::size_t count);
  std::size_t token_count() const noexcept { return token_count_; }
  std::size_t block_count() const noexcept { return block_table_.size(); }
  std::size_t block_size() const noexcept { return block_size_; }
  std::size_t max_tokens() const noexcept { return max_tokens_; }
  const std::vector<BlockId>& block_table() const noexcept { return block_table_; }
  // Internal transactional rollback for owners that append whole logical
  // tokens. Only removes blocks appended after the supplied snapshot.
  void rollback_to(std::size_t token_count, std::size_t block_count);
  PhysicalLocation locate(std::size_t logical_token) const;
  void release_all() noexcept;

 private:
  BlockManager* manager_ = nullptr;
  std::size_t block_size_ = 0;
  std::size_t max_tokens_ = 0;
  std::size_t token_count_ = 0;
  std::vector<BlockId> block_table_;
};

}  // namespace llm
