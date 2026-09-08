#pragma once

#include "decoder_kv_cache.h"
#include "tensor.h"
#include <cstddef>
#include <vector>

namespace llm {

class Qwen3KvCache final : public DecoderKvCache {
 public:
  Qwen3KvCache(size_t max_seq_len, size_t num_layers = 28,
               size_t num_kv_heads = 8, size_t head_dim = 128);

  void reset();
  size_t length() const { return length_; }
  size_t capacity() const { return capacity_; }
  size_t num_layers() const { return keys_.size(); }
  size_t num_kv_heads() const { return num_kv_heads_; }
  size_t head_dim() const { return head_dim_; }
  size_t resident_bytes() const { return resident_bytes_; }

  Tensor& key_cache(size_t layer);
  Tensor& value_cache(size_t layer);
  const Tensor& key_cache(size_t layer) const;
  const Tensor& value_cache(size_t layer) const;

  void write_prefill_layer(size_t layer, const Tensor& k_rope,
                           const Tensor& v, size_t token_count);
  void write_prefill_layer_batched_slice(size_t layer, const Tensor& k_bshd,
                                         const Tensor& v_bshd,
                                         size_t batch_index, size_t seq_len);
  void write_prefill_layer_batched_valid_slice(size_t layer, const Tensor& k_bshd,
                                               const Tensor& v_bshd,
                                               size_t batch_index, size_t valid_length);
  void append_decode_layer(size_t layer, const Tensor& k_rope_one,
                           const Tensor& v_one, size_t write_index);
  void append_decode_layer_batched_slice(size_t layer, const Tensor& k_bhd,
                                         const Tensor& v_bhd,
                                         size_t batch_index,
                                         size_t previous_length);
  void record_decode_layer(size_t layer, size_t write_index);
  void commit_prefill(size_t token_count);
  void abort_prefill();
  void commit_decode(size_t previous_length);
  void abort_decode();

 private:
  size_t capacity_;
  size_t num_kv_heads_;
  size_t head_dim_;
  size_t length_ = 0;
  size_t pending_count_ = 0;
  size_t resident_bytes_ = 0;
  bool pending_prefill_ = false;
  bool pending_decode_ = false;
  std::vector<Tensor> keys_;
  std::vector<Tensor> values_;
  std::vector<bool> written_;

  void check_layer(size_t layer) const;
  void check_cache_tensor(const Tensor& tensor, const char* name,
                          size_t token_count) const;
  void copy_into(Tensor& destination, const Tensor& source,
                 size_t write_index, size_t token_count);
};

}  // namespace llm
