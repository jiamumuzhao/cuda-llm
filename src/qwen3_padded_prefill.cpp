#include "llm/qwen3_padded_prefill.h"

#include "llm/cuda_check.h"
#include "llm/qwen3_kv_cache.h"
#include "llm/qwen3_paged_kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace llm {
namespace {
constexpr size_t kMaxPrefill = 32;
constexpr int32_t kVocabSize = 151936;

[[noreturn]] void padded_error(const std::string& message) {
  throw std::invalid_argument("PaddedPrefillBatch: " + message);
}
}  // namespace

PagedPaddedPrefillBatch build_paged_padded_prefill_batch(
    const std::vector<std::vector<int32_t>>& prompts,
    const std::vector<Qwen3PagedKvCache*>& caches) {
  const size_t batch = prompts.size();
  if (batch != 1 && batch != 2 && batch != 4)
    padded_error("paged batch size actual=" + std::to_string(batch) +
                 " expected 1, 2, or 4");
  if (caches.size() != batch)
    padded_error("paged cache count actual=" + std::to_string(caches.size()) +
                 " expected=" + std::to_string(batch));

  PagedPaddedPrefillBatch result;
  result.batch_size = batch;
  result.valid_lengths.reserve(batch);
  result.caches = caches;
  std::vector<Qwen3PagedKvCache*> unique;
  unique.reserve(batch);
  PagedKvCachePool* common_pool = nullptr;
  for (size_t b = 0; b < batch; ++b) {
    const auto& prompt = prompts[b];
    if (prompt.empty())
      padded_error("paged prompt[" + std::to_string(b) + "] must be non-empty");
    if (prompt.size() > kMaxPrefill)
      padded_error("paged prompt[" + std::to_string(b) + "] length actual=" +
                   std::to_string(prompt.size()) + " expected <=32");
    for (size_t i = 0; i < prompt.size(); ++i)
      if (prompt[i] < 0 || prompt[i] >= kVocabSize)
        padded_error("paged prompt[" + std::to_string(b) + "][" +
                     std::to_string(i) + "] token out of range [0,151936)");
    if (caches[b] == nullptr) padded_error("paged cache pointer must be non-null");
    if (std::find(unique.begin(), unique.end(), caches[b]) != unique.end())
      padded_error("paged cache pointers must be unique");
    unique.push_back(caches[b]);
    if (common_pool == nullptr) common_pool = &caches[b]->pool();
    if (&caches[b]->pool() != common_pool)
      padded_error("paged caches must share one PagedKvCachePool");
    if (caches[b]->length() != 0 || !caches[b]->block_table().empty() ||
        caches[b]->in_decode_transaction() || caches[b]->in_prefill_transaction() ||
        caches[b]->capacity() < prompt.size())
      padded_error("paged cache[" + std::to_string(b) +
                   "] must be empty and have capacity >= valid length");
    result.valid_lengths.push_back(prompt.size());
    result.max_seq_len = std::max(result.max_seq_len, prompt.size());
  }
  result.padded_token_ids.assign(batch * result.max_seq_len, 0);
  for (size_t b = 0; b < batch; ++b)
    std::copy(prompts[b].begin(), prompts[b].end(),
              result.padded_token_ids.begin() + b * result.max_seq_len);
  return result;
}

PaddedPrefillBatch build_padded_prefill_batch(
    const std::vector<std::vector<int32_t>>& prompts,
    const std::vector<Qwen3KvCache*>& caches) {
  const size_t batch = prompts.size();
  if (batch != 1 && batch != 2 && batch != 4)
    padded_error("batch size actual=" + std::to_string(batch) + " expected 1, 2, or 4");
  if (caches.size() != batch)
    padded_error("cache count actual=" + std::to_string(caches.size()) +
                 " expected=" + std::to_string(batch));

  PaddedPrefillBatch result;
  result.batch_size = batch;
  result.valid_lengths.reserve(batch);
  result.caches = caches;
  std::vector<Qwen3KvCache*> unique;
  unique.reserve(batch);
  for (size_t b = 0; b < batch; ++b) {
    const auto& prompt = prompts[b];
    if (prompt.empty()) padded_error("prompt[" + std::to_string(b) + "] must be non-empty");
    if (prompt.size() > kMaxPrefill)
      padded_error("prompt[" + std::to_string(b) + "] length actual=" +
                   std::to_string(prompt.size()) + " expected <=32; long-context padded prefill is not supported");
    for (size_t i = 0; i < prompt.size(); ++i)
      if (prompt[i] < 0 || prompt[i] >= kVocabSize)
        padded_error("prompt[" + std::to_string(b) + "][" + std::to_string(i) +
                     "] token out of range [0,151936)");
    if (caches[b] == nullptr) padded_error("cache pointer must be non-null");
    if (std::find(unique.begin(), unique.end(), caches[b]) != unique.end())
      padded_error("cache pointers must be unique");
    unique.push_back(caches[b]);
    if (caches[b]->length() != 0 || caches[b]->capacity() < prompt.size() ||
        caches[b]->num_layers() != 28 || caches[b]->num_kv_heads() != 8 ||
        caches[b]->head_dim() != 128)
      padded_error("cache[" + std::to_string(b) + "] must be empty, Qwen3-compatible, and have capacity >= valid length");
    result.valid_lengths.push_back(prompt.size());
    result.max_seq_len = std::max(result.max_seq_len, prompt.size());
  }
  result.padded_token_ids.assign(batch * result.max_seq_len, 0);
  for (size_t b = 0; b < batch; ++b)
    std::copy(prompts[b].begin(), prompts[b].end(),
              result.padded_token_ids.begin() + b * result.max_seq_len);
  return result;
}

Tensor select_last_valid_logits(const Tensor& padded_logits,
                                const std::vector<size_t>& valid_lengths) {
  if (!padded_logits.is_contiguous() || padded_logits.shape().size() != 3)
    padded_error("padded logits must be contiguous rank-3 [B,S,V]");
  const auto& shape = padded_logits.shape();
  const size_t batch = static_cast<size_t>(shape[0]);
  const size_t seq = static_cast<size_t>(shape[1]);
  const size_t vocab = static_cast<size_t>(shape[2]);
  if (batch == 0 || seq == 0 || vocab == 0 || valid_lengths.size() != batch)
    padded_error("invalid padded logits shape or valid_lengths size");
  for (size_t length : valid_lengths)
    if (length == 0 || length > seq) padded_error("valid length is outside padded logits sequence");

  Tensor selected(padded_logits.dtype(),
                  {static_cast<int64_t>(batch), static_cast<int64_t>(vocab)},
                  padded_logits.device());
  const size_t row_bytes = vocab * dtype_size(padded_logits.dtype());
  if (padded_logits.device() == DeviceType::CUDA) {
    const auto* src = static_cast<const char*>(padded_logits.data());
    auto* dst = static_cast<char*>(selected.data());
    const size_t batch_stride = seq * row_bytes;
    for (size_t b = 0; b < batch; ++b)
      CUDA_CHECK(cudaMemcpy(dst + b * row_bytes,
                            src + b * batch_stride + (valid_lengths[b] - 1) * row_bytes,
                            row_bytes, cudaMemcpyDeviceToDevice));
  } else {
    for (size_t b = 0; b < batch; ++b)
      for (size_t v = 0; v < vocab; ++v) {
        const size_t src_index = (b * seq + valid_lengths[b] - 1) * vocab + v;
        selected.set_f32(b * vocab + v, padded_logits.get_f32(src_index));
      }
  }
  return selected;
}
}  // namespace llm
