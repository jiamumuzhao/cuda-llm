#pragma once

#include "tensor.h"

#include <cstddef>
#include <stdexcept>

namespace llm {

// Model-owned metadata buffers for paged decode. This workspace is deliberately
// single-threaded: one Qwen3CudaModel must not serve overlapping paged decode
// calls. The block-table capacity is derived from the first compatible pool's
// block size and is reused for all subsequent calls.
class PagedDecodeMetadataWorkspace {
 private:
  std::size_t block_size_;
  std::size_t max_blocks_;

 public:
  static constexpr std::size_t kMaxBatch = 4;
  static constexpr std::size_t kMaxSequence = 32;

  explicit PagedDecodeMetadataWorkspace(std::size_t block_size)
      : block_size_(block_size),
        max_blocks_(checked_max_blocks(block_size)),
        positions(DType::I32, {static_cast<int64_t>(kMaxBatch)}, DeviceType::CUDA),
        block_tables(DType::I32,
                     {static_cast<int64_t>(kMaxBatch),
                      static_cast<int64_t>(max_blocks_)}, DeviceType::CUDA) {}

  std::size_t block_size() const noexcept { return block_size_; }
  std::size_t max_blocks() const noexcept { return max_blocks_; }
  std::size_t resident_bytes() const noexcept {
    return positions.nbytes() + block_tables.nbytes();
  }
  Tensor positions_for_batch(std::size_t batch) const {
    if (batch == 0 || batch > kMaxBatch)
      throw std::invalid_argument("PagedDecodeMetadataWorkspace: invalid batch");
    return positions.prefix_first_dim(batch);
  }
  Tensor block_table_row(std::size_t row, std::size_t block_count) const {
    if (row >= kMaxBatch || block_count == 0 || block_count > max_blocks_)
      throw std::invalid_argument("PagedDecodeMetadataWorkspace: invalid block-table row");
    return block_tables.slice_first_dim(row, 1)
        .reshape({static_cast<int64_t>(max_blocks_)})
        .prefix_first_dim(block_count);
  }

  Tensor positions;
  Tensor block_tables;

 private:
  static std::size_t checked_max_blocks(std::size_t block_size) {
    if (block_size == 0)
      throw std::invalid_argument("PagedDecodeMetadataWorkspace: block_size must be > 0");
    return (kMaxSequence + block_size - 1) / block_size;
  }
};

// One workspace is owned by one Qwen3CudaModel. Decode is intentionally
// single-threaded for that model; callers must not overlap decode calls.
struct DecodeWorkspace {
  static constexpr size_t kMaxBatch = 4;
  Tensor hidden_a{DType::F16, {4, 1024}, DeviceType::CUDA};
  Tensor hidden_b{DType::F16, {4, 1024}, DeviceType::CUDA};
  Tensor input_norm{DType::F16, {4, 1024}, DeviceType::CUDA};
  Tensor q_linear{DType::F16, {4, 2048}, DeviceType::CUDA};
  Tensor k_linear{DType::F16, {4, 1024}, DeviceType::CUDA};
  Tensor v_linear{DType::F16, {4, 1024}, DeviceType::CUDA};
  Tensor attention{DType::F16, {4, 2048}, DeviceType::CUDA};
  Tensor o_proj{DType::F16, {4, 1024}, DeviceType::CUDA};
  Tensor attention_residual{DType::F16, {4, 1024}, DeviceType::CUDA};
  Tensor post_attention_norm{DType::F16, {4, 1024}, DeviceType::CUDA};
  Tensor gate{DType::F16, {4, 3072}, DeviceType::CUDA};
  Tensor up{DType::F16, {4, 3072}, DeviceType::CUDA};
  Tensor down{DType::F16, {4, 1024}, DeviceType::CUDA};
  Tensor final_norm{DType::F16, {4, 1024}, DeviceType::CUDA};
  // F32 storage is used only as a four-byte RAII device buffer; kernels
  // interpret it as int32 position/token metadata.
  Tensor position_ids{DType::F32, {1}, DeviceType::CUDA};
  Tensor variable_position_ids{DType::I32, {4}, DeviceType::CUDA};
  Tensor variable_lengths{DType::I32, {4}, DeviceType::CUDA};
  Tensor token_ids{DType::F32, {4}, DeviceType::CUDA};

  size_t resident_bytes() const {
    return hidden_a.nbytes() + hidden_b.nbytes() + input_norm.nbytes() +
           q_linear.nbytes() + k_linear.nbytes() + v_linear.nbytes() +
           attention.nbytes() + o_proj.nbytes() + attention_residual.nbytes() +
           post_attention_norm.nbytes() + gate.nbytes() + up.nbytes() +
           down.nbytes() + final_norm.nbytes() + position_ids.nbytes() +
           variable_position_ids.nbytes() + variable_lengths.nbytes() +
           token_ids.nbytes();
  }
};

}  // namespace llm
