#include "llm/qwen3_kv_cache.h"

#include "llm/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llm {
namespace {
[[noreturn]] void cache_error(const std::string& message) {
  throw std::invalid_argument("Qwen3KvCache: " + message);
}
}

Qwen3KvCache::Qwen3KvCache(size_t max_seq_len, size_t num_layers,
                           size_t num_kv_heads, size_t head_dim)
    : capacity_(max_seq_len), num_kv_heads_(num_kv_heads), head_dim_(head_dim) {
  if (capacity_ == 0 || num_layers == 0 || num_kv_heads_ == 0 || head_dim_ == 0)
    cache_error("capacity, num_layers, num_kv_heads, and head_dim must be > 0");
  keys_.reserve(num_layers);
  values_.reserve(num_layers);
  written_.assign(num_layers, false);
  for (size_t i = 0; i < num_layers; ++i) {
    keys_.emplace_back(DType::F16,
                       std::vector<int64_t>{static_cast<int64_t>(capacity_),
                                            static_cast<int64_t>(num_kv_heads_),
                                            static_cast<int64_t>(head_dim_)},
                       DeviceType::CUDA);
    values_.emplace_back(DType::F16,
                         std::vector<int64_t>{static_cast<int64_t>(capacity_),
                                              static_cast<int64_t>(num_kv_heads_),
                                              static_cast<int64_t>(head_dim_)},
                         DeviceType::CUDA);
    resident_bytes_ += keys_.back().nbytes() + values_.back().nbytes();
  }
}

void Qwen3KvCache::check_layer(size_t layer) const {
  if (layer >= keys_.size())
    cache_error("layer actual=" + std::to_string(layer) + " expected < " +
                std::to_string(keys_.size()));
}

void Qwen3KvCache::check_cache_tensor(const Tensor& tensor, const char* name,
                                      size_t token_count) const {
  const std::vector<int64_t> expected{
      static_cast<int64_t>(token_count), static_cast<int64_t>(num_kv_heads_),
      static_cast<int64_t>(head_dim_)};
  if (tensor.device() != DeviceType::CUDA || tensor.dtype() != DType::F16 ||
      !tensor.is_contiguous() || tensor.shape() != expected) {
    cache_error(std::string(name) + " actual device/dtype/layout/shape mismatch; "
                "expected CUDA/F16/contiguous [" +
                std::to_string(token_count) + "," +
                std::to_string(num_kv_heads_) + "," +
                std::to_string(head_dim_) + "]");
  }
}

void Qwen3KvCache::copy_into(Tensor& destination, const Tensor& source,
                             size_t write_index, size_t token_count) {
  if (write_index > capacity_ || token_count > capacity_ - write_index)
    cache_error("write range exceeds capacity");
  const size_t row_bytes = num_kv_heads_ * head_dim_ * dtype_size(DType::F16);
  auto* dst = static_cast<char*>(destination.data()) + write_index * row_bytes;
  CUDA_CHECK(cudaMemcpy(dst, source.data(), token_count * row_bytes,
                        cudaMemcpyDeviceToDevice));
}

void Qwen3KvCache::reset() {
  length_ = 0;
  pending_count_ = 0;
  pending_prefill_ = false;
  pending_decode_ = false;
  std::fill(written_.begin(), written_.end(), false);
}

Tensor& Qwen3KvCache::key_cache(size_t layer) {
  check_layer(layer);
  return keys_[layer];
}
Tensor& Qwen3KvCache::value_cache(size_t layer) {
  check_layer(layer);
  return values_[layer];
}
const Tensor& Qwen3KvCache::key_cache(size_t layer) const {
  check_layer(layer);
  return keys_[layer];
}
const Tensor& Qwen3KvCache::value_cache(size_t layer) const {
  check_layer(layer);
  return values_[layer];
}

void Qwen3KvCache::write_prefill_layer(size_t layer, const Tensor& k_rope,
                                       const Tensor& v, size_t token_count) {
  check_layer(layer);
  if (length_ != 0 || pending_decode_)
    cache_error("prefill write requires an empty cache and no decode transaction");
  if (token_count == 0 || token_count > capacity_)
    cache_error("prefill token_count actual=" + std::to_string(token_count) +
                " expected in [1,capacity]");
  if (pending_prefill_ && pending_count_ != token_count)
    cache_error("prefill token_count changed within transaction");
  check_cache_tensor(k_rope, "k_rope", token_count);
  check_cache_tensor(v, "v", token_count);
  pending_prefill_ = true;
  pending_count_ = token_count;
  if (written_[layer]) cache_error("prefill layer written twice");
  copy_into(keys_[layer], k_rope, 0, token_count);
  copy_into(values_[layer], v, 0, token_count);
  written_[layer] = true;
}

void Qwen3KvCache::write_prefill_layer_batched_slice(
    size_t layer, const Tensor& k_bshd, const Tensor& v_bshd,
    size_t batch_index, size_t seq_len) {
  check_layer(layer);
  if (length_ != 0 || pending_decode_)
    cache_error("batched prefill write requires an empty cache and no decode transaction");
  if (seq_len == 0 || seq_len > capacity_)
    cache_error("batched prefill seq_len exceeds capacity");
  const std::vector<int64_t> expected{
      static_cast<int64_t>(k_bshd.shape().empty() ? 0 : k_bshd.shape()[0]),
      static_cast<int64_t>(seq_len), 8, 128};
  if (k_bshd.device() != DeviceType::CUDA || k_bshd.dtype() != DType::F16 ||
      !k_bshd.is_contiguous() || k_bshd.shape() != expected ||
      v_bshd.device() != DeviceType::CUDA || v_bshd.dtype() != DType::F16 ||
      !v_bshd.is_contiguous() || v_bshd.shape() != expected)
    cache_error("batched K/V actual device/dtype/layout/shape mismatch; expected CUDA/F16/contiguous [B,S,8,128]");
  if (batch_index >= static_cast<size_t>(k_bshd.shape()[0]))
    cache_error("batched batch_index out of range");
  if (pending_prefill_ && pending_count_ != seq_len)
    cache_error("prefill token_count changed within transaction");
  if (written_[layer]) cache_error("prefill layer written twice");
  pending_prefill_ = true; pending_count_ = seq_len;
  const size_t row_bytes = num_kv_heads_ * head_dim_ * dtype_size(DType::F16);
  const size_t batch_stride = seq_len * row_bytes;
  auto* k_dst = static_cast<char*>(keys_[layer].data());
  auto* v_dst = static_cast<char*>(values_[layer].data());
  const auto* k_src = static_cast<const char*>(k_bshd.data()) + batch_index * batch_stride;
  const auto* v_src = static_cast<const char*>(v_bshd.data()) + batch_index * batch_stride;
  CUDA_CHECK(cudaMemcpy(k_dst, k_src, batch_stride, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(v_dst, v_src, batch_stride, cudaMemcpyDeviceToDevice));
  written_[layer] = true;
}

void Qwen3KvCache::write_prefill_layer_batched_valid_slice(
    size_t layer, const Tensor& k_bshd, const Tensor& v_bshd,
    size_t batch_index, size_t valid_length) {
  check_layer(layer);
  if (length_ != 0 || pending_decode_)
    cache_error("padded prefill write requires an empty cache and no decode transaction");
  if (valid_length == 0 || valid_length > capacity_)
    cache_error("padded prefill valid_length exceeds capacity");
  if (k_bshd.shape().size() != 4 || v_bshd.shape() != k_bshd.shape() ||
      k_bshd.device() != DeviceType::CUDA || k_bshd.dtype() != DType::F16 ||
      !k_bshd.is_contiguous() || v_bshd.device() != DeviceType::CUDA ||
      v_bshd.dtype() != DType::F16 || !v_bshd.is_contiguous() ||
      k_bshd.shape()[0] <= 0 || k_bshd.shape()[1] < static_cast<int64_t>(valid_length) ||
      k_bshd.shape()[2] != static_cast<int64_t>(num_kv_heads_) ||
      k_bshd.shape()[3] != static_cast<int64_t>(head_dim_))
    cache_error("padded K/V actual device/dtype/layout/shape mismatch; expected CUDA/F16/contiguous [B,S,8,128] with S>=valid_length");
  if (batch_index >= static_cast<size_t>(k_bshd.shape()[0]))
    cache_error("padded batch_index out of range");
  if (pending_prefill_ && pending_count_ != valid_length)
    cache_error("padded prefill valid length changed within transaction");
  if (written_[layer]) cache_error("prefill layer written twice");
  pending_prefill_ = true;
  pending_count_ = valid_length;
  const size_t row_bytes = num_kv_heads_ * head_dim_ * dtype_size(DType::F16);
  const size_t batch_stride = static_cast<size_t>(k_bshd.shape()[1]) * row_bytes;
  const auto* k_src = static_cast<const char*>(k_bshd.data()) + batch_index * batch_stride;
  const auto* v_src = static_cast<const char*>(v_bshd.data()) + batch_index * batch_stride;
  CUDA_CHECK(cudaMemcpy(keys_[layer].data(), k_src, valid_length * row_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(values_[layer].data(), v_src, valid_length * row_bytes,
                        cudaMemcpyDeviceToDevice));
  written_[layer] = true;
}

void Qwen3KvCache::append_decode_layer(size_t layer, const Tensor& k_rope_one,
                                       const Tensor& v_one, size_t write_index) {
  check_layer(layer);
  if (length_ == 0 || pending_prefill_)
    cache_error("decode append requires a committed non-empty cache");
  if (write_index != length_ || write_index >= capacity_)
    cache_error("decode write_index actual=" + std::to_string(write_index) +
                " expected current length=" + std::to_string(length_));
  check_cache_tensor(k_rope_one, "k_rope_one", 1);
  check_cache_tensor(v_one, "v_one", 1);
  if (pending_decode_ && pending_count_ != write_index)
    cache_error("decode write index changed within transaction");
  pending_decode_ = true;
  pending_count_ = write_index;
  if (written_[layer]) cache_error("decode layer written twice");
  copy_into(keys_[layer], k_rope_one, write_index, 1);
  copy_into(values_[layer], v_one, write_index, 1);
  written_[layer] = true;
}

void Qwen3KvCache::append_decode_layer_batched_slice(
    size_t layer, const Tensor& k_bhd, const Tensor& v_bhd,
    size_t batch_index, size_t previous_length) {
  check_layer(layer);
  if (length_ == 0 || pending_prefill_)
    cache_error("batched decode append requires a committed non-empty cache");
  if (previous_length != length_ || previous_length >= capacity_)
    cache_error("batched decode previous_length must equal current length and be < capacity");
  const std::vector<int64_t> expected{
      static_cast<int64_t>(k_bhd.shape().empty() ? 0 : k_bhd.shape()[0]), 8, 128};
  if (k_bhd.device() != DeviceType::CUDA || k_bhd.dtype() != DType::F16 ||
      !k_bhd.is_contiguous() || k_bhd.shape() != expected ||
      v_bhd.device() != DeviceType::CUDA || v_bhd.dtype() != DType::F16 ||
      !v_bhd.is_contiguous() || v_bhd.shape() != expected)
    cache_error("batched decode K/V actual device/dtype/layout/shape mismatch; expected CUDA/F16/contiguous [B,8,128]");
  if (batch_index >= static_cast<size_t>(k_bhd.shape()[0]))
    cache_error("batched decode batch_index out of range");
  if (pending_decode_ && pending_count_ != previous_length)
    cache_error("batched decode previous_length changed within transaction");
  if (written_[layer]) cache_error("decode layer written twice");
  pending_decode_ = true;
  pending_count_ = previous_length;
  const size_t row_bytes = num_kv_heads_ * head_dim_ * dtype_size(DType::F16);
  const auto* k_src = static_cast<const char*>(k_bhd.data()) + batch_index * row_bytes;
  const auto* v_src = static_cast<const char*>(v_bhd.data()) + batch_index * row_bytes;
  auto* k_dst = static_cast<char*>(keys_[layer].data()) + previous_length * row_bytes;
  auto* v_dst = static_cast<char*>(values_[layer].data()) + previous_length * row_bytes;
  CUDA_CHECK(cudaMemcpy(k_dst, k_src, row_bytes, cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(v_dst, v_src, row_bytes, cudaMemcpyDeviceToDevice));
  written_[layer] = true;
}

void Qwen3KvCache::record_decode_layer(size_t layer, size_t write_index) {
  check_layer(layer);
  if (length_ == 0 || pending_prefill_ || write_index != length_ ||
      write_index >= capacity_)
    cache_error("invalid decode record order or write_index");
  if (pending_decode_ && pending_count_ != write_index)
    cache_error("decode write index changed within transaction");
  pending_decode_ = true;
  pending_count_ = write_index;
  if (written_[layer]) cache_error("decode layer recorded twice");
  written_[layer] = true;
}

void Qwen3KvCache::commit_prefill(size_t token_count) {
  if (!pending_prefill_ || pending_decode_ || pending_count_ != token_count ||
      token_count == 0 || token_count > capacity_)
    cache_error("invalid prefill commit order or token_count");
  for (bool value : written_)
    if (!value) cache_error("prefill commit before all layers were written");
  length_ = token_count;
  pending_prefill_ = false;
  pending_count_ = 0;
  std::fill(written_.begin(), written_.end(), false);
}

void Qwen3KvCache::abort_prefill() {
  if (pending_decode_)
    cache_error("cannot abort prefill while decode transaction is pending");
  pending_prefill_ = false;
  pending_count_ = 0;
  std::fill(written_.begin(), written_.end(), false);
}

void Qwen3KvCache::commit_decode(size_t previous_length) {
  if (!pending_decode_ || pending_prefill_ || previous_length != length_ ||
      previous_length >= capacity_)
    cache_error("invalid decode commit order or previous_length");
  for (bool value : written_)
    if (!value) cache_error("decode commit before all layers were written");
  ++length_;
  pending_decode_ = false;
  pending_count_ = 0;
  std::fill(written_.begin(), written_.end(), false);
}

void Qwen3KvCache::abort_decode() {
  if (pending_prefill_)
    cache_error("cannot abort decode while prefill transaction is pending");
  pending_decode_ = false;
  pending_count_ = 0;
  std::fill(written_.begin(), written_.end(), false);
}

}  // namespace llm
