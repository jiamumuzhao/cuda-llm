#include "llm/paged_kv_block_manager.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace llm {
namespace {
[[noreturn]] void block_error(const std::string& message) {
  throw std::invalid_argument("BlockManager: " + message);
}
[[noreturn]] void sequence_error(const std::string& message) {
  throw std::invalid_argument("SequenceBlockTable: " + message);
}
}

BlockManager::BlockManager(std::size_t total_blocks)
    : total_blocks_(total_blocks), ref_counts_(total_blocks, 0),
      allocated_(total_blocks, false) {
  if (total_blocks == 0) block_error("total_blocks must be > 0");
  if (total_blocks > static_cast<std::size_t>(kInvalidBlockId))
    block_error("total_blocks exceeds BlockId range");
  free_blocks_.reserve(total_blocks);
  for (std::size_t i = total_blocks; i-- > 0;)
    free_blocks_.push_back(static_cast<BlockId>(i));
}

void BlockManager::validate_id(BlockId block_id) const {
  if (block_id == kInvalidBlockId ||
      static_cast<std::size_t>(block_id) >= total_blocks_)
    block_error("invalid block id=" + std::to_string(block_id));
}

BlockId BlockManager::allocate() {
  if (free_blocks_.empty())
    throw BlockPoolExhausted("BlockManager: block pool exhausted");
  const BlockId block_id = free_blocks_.back();
  free_blocks_.pop_back();
  allocated_[block_id] = true;
  ref_counts_[block_id] = 1;
  return block_id;
}

void BlockManager::retain(BlockId block_id) {
  validate_id(block_id);
  if (!allocated_[block_id]) block_error("retain on free block id=" + std::to_string(block_id));
  if (ref_counts_[block_id] == std::numeric_limits<std::size_t>::max())
    block_error("refcount overflow for block id=" + std::to_string(block_id));
  ++ref_counts_[block_id];
}

void BlockManager::release(BlockId block_id) {
  validate_id(block_id);
  if (!allocated_[block_id]) block_error("release on free block id=" + std::to_string(block_id));
  if (ref_counts_[block_id] == 0)
    block_error("release with zero refcount for block id=" + std::to_string(block_id));
  --ref_counts_[block_id];
  if (ref_counts_[block_id] == 0) {
    allocated_[block_id] = false;
    free_blocks_.push_back(block_id);
  }
}

std::size_t BlockManager::ref_count(BlockId block_id) const {
  validate_id(block_id);
  return ref_counts_[block_id];
}

bool BlockManager::is_allocated(BlockId block_id) const {
  validate_id(block_id);
  return allocated_[block_id];
}

SequenceBlockTable::SequenceBlockTable(BlockManager& manager,
                                       std::size_t block_size,
                                       std::size_t max_tokens)
    : manager_(&manager), block_size_(block_size), max_tokens_(max_tokens) {
  if (block_size == 0) sequence_error("block_size must be > 0");
  if (max_tokens == 0) sequence_error("max_tokens must be > 0");
}

SequenceBlockTable::~SequenceBlockTable() noexcept { release_all(); }

void SequenceBlockTable::append_tokens(std::size_t count) {
  if (count == 0) return;
  if (count > max_tokens_ - token_count_)
    sequence_error("append exceeds max_tokens");
  const std::size_t new_token_count = token_count_ + count;
  const std::size_t required_blocks =
      (new_token_count + block_size_ - 1) / block_size_;
  const std::size_t new_blocks = required_blocks - block_table_.size();
  if (new_blocks == 0) {
    token_count_ = new_token_count;
    return;
  }

  // Reserve before touching the manager. Once allocation starts, all manager
  // changes are kept in newly_allocated and rolled back on every exception.
  block_table_.reserve(required_blocks);
  std::vector<BlockId> newly_allocated;
  newly_allocated.reserve(new_blocks);
  try {
    for (std::size_t i = 0; i < new_blocks; ++i)
      newly_allocated.push_back(manager_->allocate());
    block_table_.insert(block_table_.end(), newly_allocated.begin(), newly_allocated.end());
    token_count_ = new_token_count;
  } catch (...) {
    for (auto it = newly_allocated.rbegin(); it != newly_allocated.rend(); ++it)
      manager_->release(*it);
    throw;
  }
}

PhysicalLocation SequenceBlockTable::locate(std::size_t logical_token) const {
  if (logical_token >= token_count_)
    sequence_error("logical token out of range=" + std::to_string(logical_token));
  const std::size_t logical_block = logical_token / block_size_;
  return PhysicalLocation{block_table_.at(logical_block), logical_token % block_size_};
}

void SequenceBlockTable::rollback_to(std::size_t token_count,
                                     std::size_t block_count) {
  if (token_count > token_count_ || block_count > block_table_.size() ||
      block_count != (token_count == 0 ? 0 :
                      (token_count + block_size_ - 1) / block_size_))
    sequence_error("invalid rollback snapshot");
  for (std::size_t i = block_table_.size(); i-- > block_count;)
    manager_->release(block_table_[i]);
  block_table_.resize(block_count);
  token_count_ = token_count;
}

void SequenceBlockTable::release_all() noexcept {
  if (manager_ == nullptr) return;
  for (const BlockId block_id : block_table_)
    manager_->release(block_id);
  block_table_.clear();
  token_count_ = 0;
}

}  // namespace llm
