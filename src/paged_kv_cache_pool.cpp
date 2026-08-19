#include "llm/paged_kv_cache_pool.h"

#include "llm/cuda_check.h"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace llm {
namespace {
[[noreturn]] void pool_error(const std::string& message) {
  throw std::invalid_argument("PagedKvCachePool: " + message);
}
[[noreturn]] void sequence_error(const std::string& message) {
  throw std::invalid_argument("PagedSequenceKvCache: " + message);
}
std::size_t checked_product(std::size_t a, std::size_t b, const char* name) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
    pool_error(std::string(name) + " overflows size_t");
  return a * b;
}
}

PagedKvCachePool::PagedKvCachePool(const PagedKvCachePoolConfig& config)
    : config_(config), block_manager_(config.total_blocks) {
  if (config.total_blocks == 0 || config.num_layers == 0 ||
      config.num_kv_heads == 0 || config.block_size == 0 || config.head_dim == 0)
    pool_error("all dimensions must be > 0");
  if (config.dtype != DType::F16)
    pool_error("only CUDA F16 is supported");
  const std::size_t token_elements = checked_product(
      config.num_kv_heads, config.head_dim, "token elements");
  token_bytes_ = checked_product(token_elements, sizeof(std::uint16_t), "token_bytes");
  kind_bytes_ = checked_product(config.block_size, token_bytes_, "kv_plane_bytes");
  block_bytes_ = checked_product(
      checked_product(config.num_layers, 2, "block_bytes"), kind_bytes_, "block_bytes");
  const std::size_t total_bytes = checked_product(config.total_blocks, block_bytes_, "storage bytes");
  (void)total_bytes;
  storage_ = Tensor(DType::F16,
                    {static_cast<std::int64_t>(config.num_layers),
                     static_cast<std::int64_t>(config.total_blocks), 2,
                     static_cast<std::int64_t>(config.block_size),
                     static_cast<std::int64_t>(config.num_kv_heads),
                     static_cast<std::int64_t>(config.head_dim)},
                    DeviceType::CUDA);
}

void PagedKvCachePool::validate_location(std::size_t layer, BlockId block_id,
                                         PagedKvKind kind, std::size_t offset,
                                         bool check_offset) const {
  if (layer >= config_.num_layers)
    pool_error("layer out of range");
  if (block_id == kInvalidBlockId ||
      static_cast<std::size_t>(block_id) >= config_.total_blocks ||
      !block_manager_.is_allocated(block_id))
    pool_error("block id is invalid or free");
  if (kind != PagedKvKind::Key && kind != PagedKvKind::Value)
    pool_error("invalid K/V selector");
  if (check_offset && offset >= config_.block_size)
    pool_error("block offset out of range");
}

std::size_t PagedKvCachePool::kind_offset_bytes(std::size_t layer, BlockId block_id,
                                                PagedKvKind kind) const {
  const std::size_t kv_index = kind == PagedKvKind::Key ? 0 : 1;
  const std::size_t layer_block = layer * config_.total_blocks + block_id;
  return (layer_block * 2 + kv_index) * kind_bytes_;
}

void* PagedKvCachePool::device_block_ptr(std::size_t layer, BlockId block_id,
                                         PagedKvKind kind) {
  validate_location(layer, block_id, kind, 0, false);
  return static_cast<char*>(storage_.data()) + kind_offset_bytes(layer, block_id, kind);
}

const void* PagedKvCachePool::device_block_ptr(std::size_t layer, BlockId block_id,
                                               PagedKvKind kind) const {
  validate_location(layer, block_id, kind, 0, false);
  return static_cast<const char*>(storage_.data()) + kind_offset_bytes(layer, block_id, kind);
}

void* PagedKvCachePool::device_token_ptr(std::size_t layer, BlockId block_id,
                                         PagedKvKind kind, std::size_t offset) {
  validate_location(layer, block_id, kind, offset, true);
  return static_cast<char*>(device_block_ptr(layer, block_id, kind)) +
         offset * token_bytes_;
}

const void* PagedKvCachePool::device_token_ptr(std::size_t layer, BlockId block_id,
                                               PagedKvKind kind, std::size_t offset) const {
  validate_location(layer, block_id, kind, offset, true);
  return static_cast<const char*>(device_block_ptr(layer, block_id, kind)) +
         offset * token_bytes_;
}

PagedSequenceKvCache::PagedSequenceKvCache(PagedKvCachePool& pool,
                                           std::size_t max_tokens)
    : pool_(&pool), table_(pool.block_manager(), pool.config().block_size, max_tokens) {}

void* PagedSequenceKvCache::device_token_ptr(std::size_t layer,
                                             std::size_t logical_token,
                                             PagedKvKind kind) {
  const PhysicalLocation location = table_.locate(logical_token);
  return pool_->device_token_ptr(layer, location.block_id, kind, location.offset);
}

const void* PagedSequenceKvCache::device_token_ptr(std::size_t layer,
                                                   std::size_t logical_token,
                                                   PagedKvKind kind) const {
  const PhysicalLocation location = table_.locate(logical_token);
  return pool_->device_token_ptr(layer, location.block_id, kind, location.offset);
}

void PagedSequenceKvCache::validate_copy_count(const std::uint16_t* bits,
                                               std::size_t element_count) const {
  const std::size_t expected = pool_->config().num_kv_heads * pool_->config().head_dim;
  if (bits == nullptr) sequence_error("host FP16 pointer must not be null");
  if (element_count != expected)
    sequence_error("element_count must equal num_kv_heads * head_dim");
}

void PagedSequenceKvCache::copy_token_from_host(
    std::size_t layer, std::size_t logical_token, PagedKvKind kind,
    const std::uint16_t* fp16_bits, std::size_t element_count) {
  validate_copy_count(fp16_bits, element_count);
  CUDA_CHECK(cudaMemcpy(device_token_ptr(layer, logical_token, kind), fp16_bits,
                        element_count * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
}

void PagedSequenceKvCache::copy_token_to_host(
    std::size_t layer, std::size_t logical_token, PagedKvKind kind,
    std::uint16_t* fp16_bits, std::size_t element_count) const {
  validate_copy_count(fp16_bits, element_count);
  CUDA_CHECK(cudaMemcpy(fp16_bits, device_token_ptr(layer, logical_token, kind),
                        element_count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
}

Tensor PagedSequenceKvCache::make_device_block_table_i32() const {
  if (table_.block_table().empty())
    sequence_error("cannot upload an empty block table");
  Tensor device_table(DType::I32,
                      {static_cast<std::int64_t>(table_.block_table().size())},
                      DeviceType::CUDA);
  CUDA_CHECK(cudaMemcpy(device_table.data(), table_.block_table().data(),
                        table_.block_table().size() * sizeof(BlockId),
                        cudaMemcpyHostToDevice));
  return device_table;
}

void PagedSequenceKvCache::copy_block_table_to_cuda(Tensor& output) const {
  if (table_.block_table().empty())
    sequence_error("cannot upload an empty block table");
  if (output.device() != DeviceType::CUDA || output.dtype() != DType::I32 ||
      !output.is_contiguous() || output.shape() != std::vector<int64_t>{
          static_cast<int64_t>(table_.block_table().size())})
    sequence_error("output must be CUDA/I32/contiguous with the current block count");
  CUDA_CHECK(cudaMemcpy(output.data(), table_.block_table().data(),
                        table_.block_table().size() * sizeof(BlockId),
                        cudaMemcpyHostToDevice));
}

}  // namespace llm
