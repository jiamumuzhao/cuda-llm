#include "llm/qwen3_paged_kv_cache.h"

#include "llm/cuda_check.h"
#include "llm/ops_cuda.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llm {

void Qwen3PagedKvCache::error(const std::string& message) const {
  throw std::invalid_argument("Qwen3PagedKvCache: " + message);
}

void Qwen3PagedKvCache::require_qwen3_pool() const {
  const auto& c = pool_->config();
  if (c.num_layers != kLayers || c.num_kv_heads != kKvHeads ||
      c.head_dim != kHeadDim || c.dtype != DType::F16)
    error("pool must be Qwen3 CUDA/F16 config [layers=28,kv_heads=8,head_dim=128]");
}

Qwen3PagedKvCache::Qwen3PagedKvCache(PagedKvCachePool& pool,
                                     std::size_t max_seq_len)
    : pool_(&pool), sequence_(pool, max_seq_len),
      capacity_(max_seq_len), written_(kLayers, false),
      device_cache_length_i32_(DType::I32, {1}, DeviceType::CUDA),
      device_position_i32_(DType::I32, {1}, DeviceType::CUDA) {
  require_qwen3_pool();
  if (max_seq_len == 0 || max_seq_len > kMaxSequence)
    error("max_seq_len must be in [1,512]");
}

void Qwen3PagedKvCache::require_pending(const char* function) const {
  if (transaction_ != Transaction::Decode)
    error(std::string(function) + " requires an active decode transaction");
}

void Qwen3PagedKvCache::require_kv_tensor(const Tensor& tensor,
                                          const char* name) const {
  if (tensor.device() != DeviceType::CUDA || tensor.dtype() != DType::F16 ||
      !tensor.is_contiguous() || tensor.shape() != std::vector<int64_t>{8, 128})
    error(std::string(name) + " must be CUDA/F16/contiguous [8,128]");
}

void Qwen3PagedKvCache::require_prefill_tensor(const Tensor& tensor,
                                               const char* name,
                                               std::size_t token_count) const {
  if (tensor.device() != DeviceType::CUDA || tensor.dtype() != DType::F16 ||
      !tensor.is_contiguous() ||
      tensor.shape() != std::vector<int64_t>{static_cast<int64_t>(token_count), 8, 128})
    error(std::string(name) + " must be CUDA/F16/contiguous [S,8,128]");
}

void Qwen3PagedKvCache::begin_decode() {
  if (transaction_ != Transaction::None)
    error("begin_decode called while a transaction is active");
  if (length_ >= capacity_) error("begin_decode at full capacity");
  begin_length_ = length_;
  begin_block_count_ = sequence_.block_count();
  pending_position_ = length_;
  sequence_.append_tokens(1);
  const int32_t cache_length = static_cast<int32_t>(pending_position_ + 1);
  CUDA_CHECK(cudaMemcpy(device_cache_length_i32_.data(), &cache_length,
                        sizeof(cache_length), cudaMemcpyHostToDevice));
  const int32_t position = static_cast<int32_t>(pending_position_);
  CUDA_CHECK(cudaMemcpy(device_position_i32_.data(), &position, sizeof(position),
                        cudaMemcpyHostToDevice));
  std::fill(written_.begin(), written_.end(), false);
  transaction_ = Transaction::Decode;
}

void Qwen3PagedKvCache::append_layer_kv_device(
    std::size_t layer, const Tensor& key, const Tensor& value,
    const Tensor& device_block_table_i32, cudaStream_t stream) {
  require_pending("append_layer_kv_device");
  if (layer >= kLayers) error("layer out of range");
  if (written_[layer]) error("layer written twice");
  require_kv_tensor(key, "key");
  require_kv_tensor(value, "value");
  if (device_block_table_i32.device() != DeviceType::CUDA ||
      device_block_table_i32.dtype() != DType::I32 ||
      !device_block_table_i32.is_contiguous() ||
      device_block_table_i32.shape().size() != 1 ||
      device_block_table_i32.shape()[0] !=
          static_cast<int64_t>(sequence_.block_count()))
    error("device_block_table must be CUDA/I32/contiguous with cache block count");
  cuda_paged_kv_write_decode(key, value, *pool_, layer,
                             device_block_table_i32,
                             device_cache_length_i32_, kKvHeads, kHeadDim, stream);
  written_[layer] = true;
}

void Qwen3PagedKvCache::mark_layer_kv_device_written(std::size_t layer) {
  require_pending("mark_layer_kv_device_written");
  if (layer >= kLayers) error("layer out of range");
  if (written_[layer]) error("layer written twice");
  written_[layer] = true;
}

void Qwen3PagedKvCache::append_layer_kv(std::size_t layer, const Tensor& key,
                                        const Tensor& value) {
  require_pending("append_layer_kv");
  if (layer >= kLayers) error("layer out of range");
  if (written_[layer]) error("layer written twice");
  require_kv_tensor(key, "key");
  require_kv_tensor(value, "value");
  const std::size_t bytes = kKvHeads * kHeadDim * sizeof(std::uint16_t);
  void* key_dst = sequence_.device_token_ptr(layer, pending_position_, PagedKvKind::Key);
  void* value_dst = sequence_.device_token_ptr(layer, pending_position_, PagedKvKind::Value);
  // Keep the pair atomic at the state-machine level: written_ changes only
  // after both checked D2D copies succeed. A failed first copy remains abortable.
  CUDA_CHECK(cudaMemcpy(key_dst, key.data(), bytes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(value_dst, value.data(), bytes, cudaMemcpyDeviceToDevice));
  written_[layer] = true;
}

void Qwen3PagedKvCache::commit_decode() {
  require_pending("commit_decode");
  for (bool value : written_)
    if (!value) error("commit_decode requires all 28 layers written");
  length_ = begin_length_ + 1;
  pending_position_ = 0;
  transaction_ = Transaction::None;
  std::fill(written_.begin(), written_.end(), false);
}

void Qwen3PagedKvCache::abort_decode() noexcept {
  if (transaction_ != Transaction::Decode) return;
  try {
    sequence_.rollback_to(begin_length_, begin_block_count_);
  } catch (...) {
    std::terminate();
  }
  length_ = begin_length_;
  const int32_t restored_length = static_cast<int32_t>(length_);
  if (device_cache_length_i32_.data() != nullptr)
    cudaMemcpy(device_cache_length_i32_.data(), &restored_length,
               sizeof(restored_length), cudaMemcpyHostToDevice);
  if (device_position_i32_.data() != nullptr)
    cudaMemcpy(device_position_i32_.data(), &restored_length,
               sizeof(restored_length), cudaMemcpyHostToDevice);
  pending_position_ = 0;
  transaction_ = Transaction::None;
  std::fill(written_.begin(), written_.end(), false);
}

void Qwen3PagedKvCache::release_all() noexcept {
  abort_prefill();
  abort_decode();
  sequence_.release_all();
  length_ = 0;
  const int32_t zero = 0;
  if (device_cache_length_i32_.data() != nullptr)
    cudaMemcpy(device_cache_length_i32_.data(), &zero, sizeof(zero),
               cudaMemcpyHostToDevice);
  if (device_position_i32_.data() != nullptr)
    cudaMemcpy(device_position_i32_.data(), &zero, sizeof(zero),
               cudaMemcpyHostToDevice);
  pending_position_ = 0;
  std::fill(written_.begin(), written_.end(), false);
}

void Qwen3PagedKvCache::begin_prefill(std::size_t token_count) {
  if (transaction_ != Transaction::None)
    error("begin_prefill called while a transaction is active");
  if (length_ != 0 || sequence_.block_count() != 0)
    error("begin_prefill requires an empty cache");
  if (token_count == 0 || token_count > capacity_ || token_count > kMaxSequence)
    error("begin_prefill token_count must be in [1,min(512,capacity)]");
  begin_length_ = length_;
  begin_block_count_ = sequence_.block_count();
  try {
    sequence_.append_tokens(token_count);
  } catch (...) {
    sequence_.rollback_to(begin_length_, begin_block_count_);
    throw;
  }
  prefill_token_count_ = token_count;
  std::fill(written_.begin(), written_.end(), false);
  transaction_ = Transaction::Prefill;
}

void Qwen3PagedKvCache::append_prefill_layer_kv(std::size_t layer,
                                                const Tensor& key,
                                                const Tensor& value) {
  if (transaction_ != Transaction::Prefill)
    error("append_prefill_layer_kv requires an active prefill transaction");
  if (layer >= kLayers) error("layer out of range");
  if (written_[layer]) error("layer written twice");
  require_prefill_tensor(key, "key", prefill_token_count_);
  require_prefill_tensor(value, "value", prefill_token_count_);
  const std::size_t bytes = kKvHeads * kHeadDim * sizeof(std::uint16_t);
  try {
    for (std::size_t token = 0; token < prefill_token_count_; ++token) {
      const char* key_src = static_cast<const char*>(key.data()) + token * bytes;
      const char* value_src = static_cast<const char*>(value.data()) + token * bytes;
      CUDA_CHECK(cudaMemcpy(sequence_.device_token_ptr(layer, token, PagedKvKind::Key),
                            key_src, bytes, cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(sequence_.device_token_ptr(layer, token, PagedKvKind::Value),
                            value_src, bytes, cudaMemcpyDeviceToDevice));
    }
  } catch (...) {
    throw;
  }
  written_[layer] = true;
}

void Qwen3PagedKvCache::append_prefill_layer_kv_batched_slice(
    std::size_t layer, const Tensor& key_batched, const Tensor& value_batched,
    std::size_t batch_index, std::size_t batch_size, std::size_t seq_len) {
  if (transaction_ != Transaction::Prefill)
    error("append_prefill_layer_kv_batched_slice requires an active prefill transaction");
  if (layer >= kLayers) error("layer out of range");
  if (written_[layer]) error("layer written twice");
  if (batch_index >= batch_size || batch_size == 0 || seq_len != prefill_token_count_)
    error("invalid batch index, batch size, or sequence length");
  const std::vector<int64_t> expected = {
      static_cast<int64_t>(batch_size), static_cast<int64_t>(seq_len), 8, 128};
  if (key_batched.device() != DeviceType::CUDA || key_batched.dtype() != DType::F16 ||
      !key_batched.is_contiguous() || key_batched.shape() != expected)
    error("key_batched must be CUDA/F16/contiguous [B,S,8,128]");
  if (value_batched.device() != DeviceType::CUDA || value_batched.dtype() != DType::F16 ||
      !value_batched.is_contiguous() || value_batched.shape() != expected)
    error("value_batched must be CUDA/F16/contiguous [B,S,8,128]");
  const std::size_t bytes = kKvHeads * kHeadDim * sizeof(std::uint16_t);
  const char* key_base = static_cast<const char*>(key_batched.data()) +
                         batch_index * seq_len * bytes;
  const char* value_base = static_cast<const char*>(value_batched.data()) +
                           batch_index * seq_len * bytes;
  for (std::size_t token = 0; token < seq_len; ++token) {
    CUDA_CHECK(cudaMemcpy(sequence_.device_token_ptr(layer, token, PagedKvKind::Key),
                          key_base + token * bytes, bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(sequence_.device_token_ptr(layer, token, PagedKvKind::Value),
                          value_base + token * bytes, bytes, cudaMemcpyDeviceToDevice));
  }
  written_[layer] = true;
}

void Qwen3PagedKvCache::append_prefill_layer_kv_batched_valid_slice(
    std::size_t layer, const Tensor& key_batched, const Tensor& value_batched,
    std::size_t batch_index, std::size_t batch_size, std::size_t max_seq_len,
    std::size_t valid_length) {
  if (transaction_ != Transaction::Prefill)
    error("append_prefill_layer_kv_batched_valid_slice requires an active prefill transaction");
  if (layer >= kLayers) error("layer out of range");
  if (written_[layer]) error("layer written twice");
  if (batch_size == 0 || batch_index >= batch_size || max_seq_len == 0 ||
      valid_length != prefill_token_count_ || valid_length > max_seq_len)
    error("invalid batch index, batch size, max sequence length, or valid length");
  const std::vector<int64_t> expected = {
      static_cast<int64_t>(batch_size), static_cast<int64_t>(max_seq_len), 8, 128};
  if (key_batched.device() != DeviceType::CUDA || key_batched.dtype() != DType::F16 ||
      !key_batched.is_contiguous() || key_batched.shape() != expected)
    error("key_batched must be CUDA/F16/contiguous [B,S,8,128]");
  if (value_batched.device() != DeviceType::CUDA || value_batched.dtype() != DType::F16 ||
      !value_batched.is_contiguous() || value_batched.shape() != expected)
    error("value_batched must be CUDA/F16/contiguous [B,S,8,128]");
  const std::size_t token_bytes = kKvHeads * kHeadDim * sizeof(std::uint16_t);
  const std::size_t batch_stride = max_seq_len * token_bytes;
  const char* key_base = static_cast<const char*>(key_batched.data()) +
                         batch_index * batch_stride;
  const char* value_base = static_cast<const char*>(value_batched.data()) +
                           batch_index * batch_stride;
  for (std::size_t token = 0; token < valid_length; ++token) {
    CUDA_CHECK(cudaMemcpy(sequence_.device_token_ptr(layer, token, PagedKvKind::Key),
                          key_base + token * token_bytes, token_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(sequence_.device_token_ptr(layer, token, PagedKvKind::Value),
                          value_base + token * token_bytes, token_bytes,
                          cudaMemcpyDeviceToDevice));
  }
  written_[layer] = true;
}

void Qwen3PagedKvCache::append_prefill_layer_kv_packed_slice(
    std::size_t layer, const Tensor& key_packed, const Tensor& value_packed,
    std::size_t offset, std::size_t token_count) {
  Tensor block_table_cuda = sequence_.make_device_block_table_i32();
  append_prefill_layer_kv_packed_slice(
      layer, key_packed, value_packed, offset, token_count, block_table_cuda);
}

void Qwen3PagedKvCache::append_prefill_layer_kv_packed_slice(
    std::size_t layer, const Tensor& key_packed, const Tensor& value_packed,
    std::size_t offset, std::size_t token_count,
    const Tensor& block_table_cuda) {
  if (transaction_ != Transaction::Prefill)
    error("append_prefill_layer_kv_packed_slice requires an active prefill transaction");
  if (layer >= kLayers) error("layer out of range");
  if (written_[layer]) error("layer written twice");
  if (token_count != prefill_token_count_ || token_count == 0 ||
      offset > static_cast<std::size_t>(key_packed.shape().at(0)) ||
      token_count > static_cast<std::size_t>(key_packed.shape().at(0)) - offset)
    error("invalid packed offset or token count");
  const std::vector<int64_t> expected = {
      key_packed.shape().at(0), 8, 128};
  if (key_packed.device() != DeviceType::CUDA ||
      value_packed.device() != DeviceType::CUDA ||
      key_packed.dtype() != DType::F16 || value_packed.dtype() != DType::F16 ||
      !key_packed.is_contiguous() || !value_packed.is_contiguous() ||
      key_packed.shape() != expected || value_packed.shape() != expected)
    error("key/value packed must be CUDA/F16/contiguous [T,8,128]");
  if (block_table_cuda.device() != DeviceType::CUDA ||
      block_table_cuda.dtype() != DType::I32 ||
      !block_table_cuda.is_contiguous() ||
      block_table_cuda.numel() != sequence_.block_table().size())
    error("block_table_cuda must be CUDA/I32/contiguous with the sequence block count");
  const auto& config = pool_->config();
  cuda_paged_kv_prefill_copy_packed(
      key_packed.slice_first_dim(offset, token_count),
      value_packed.slice_first_dim(offset, token_count),
      pool_->storage(), block_table_cuda, layer, config.total_blocks,
      config.block_size, config.num_kv_heads, config.head_dim, token_count);
  written_[layer] = true;
}

void Qwen3PagedKvCache::commit_prefill() {
  if (transaction_ != Transaction::Prefill)
    error("commit_prefill requires an active prefill transaction");
  for (bool value : written_) {
    if (!value) {
      abort_prefill();
      error("commit_prefill requires all 28 layers written");
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  length_ = prefill_token_count_;
  prefill_token_count_ = 0;
  transaction_ = Transaction::None;
  std::fill(written_.begin(), written_.end(), false);
}

void Qwen3PagedKvCache::abort_prefill() noexcept {
  if (transaction_ != Transaction::Prefill) return;
  try {
    sequence_.rollback_to(begin_length_, begin_block_count_);
  } catch (...) {
    std::terminate();
  }
  length_ = begin_length_;
  prefill_token_count_ = 0;
  transaction_ = Transaction::None;
  std::fill(written_.begin(), written_.end(), false);
}

void Qwen3PagedKvCache::copy_layer_kv_to_host(
    std::size_t layer, std::size_t token, std::uint16_t* key_bits,
    std::uint16_t* value_bits, std::size_t element_count) const {
  if (layer >= kLayers || token >= length_)
    error("copy_layer_kv_to_host layer/token out of range");
  if (key_bits == nullptr || value_bits == nullptr ||
      element_count != kKvHeads * kHeadDim)
    error("copy_layer_kv_to_host requires non-null buffers and element_count=1024");
  const std::size_t bytes = element_count * sizeof(std::uint16_t);
  CUDA_CHECK(cudaMemcpy(key_bits, sequence_.device_token_ptr(layer, token, PagedKvKind::Key),
                        bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(value_bits, sequence_.device_token_ptr(layer, token, PagedKvKind::Value),
                        bytes, cudaMemcpyDeviceToHost));
}

void Qwen3PagedKvCache::seed_from_contiguous_cache(const Qwen3KvCache& source) {
  if (transaction_ != Transaction::None) error("seed requires no active transaction");
  if (length_ != 0 || sequence_.block_count() != 0)
    error("seed requires an empty paged cache");
  if (source.num_layers() != kLayers || source.num_kv_heads() != kKvHeads ||
      source.head_dim() != kHeadDim || source.capacity() > capacity_ ||
      source.length() > capacity_)
    error("source has incompatible Qwen3 dimensions or length exceeds capacity");
  const std::size_t source_length = source.length();
  if (source_length == 0) return;
  try {
    sequence_.append_tokens(source_length);
    const std::size_t bytes = kKvHeads * kHeadDim * sizeof(std::uint16_t);
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
      for (std::size_t token = 0; token < source_length; ++token) {
        const char* k_src = static_cast<const char*>(source.key_cache(layer).data()) +
                            token * bytes;
        const char* v_src = static_cast<const char*>(source.value_cache(layer).data()) +
                            token * bytes;
        CUDA_CHECK(cudaMemcpy(sequence_.device_token_ptr(layer, token, PagedKvKind::Key),
                              k_src, bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sequence_.device_token_ptr(layer, token, PagedKvKind::Value),
                              v_src, bytes, cudaMemcpyDeviceToDevice));
      }
    }
    length_ = source_length;
  } catch (...) {
    sequence_.release_all();
    length_ = 0;
    throw;
  }
}

}  // namespace llm
