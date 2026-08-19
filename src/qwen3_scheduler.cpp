#include "llm/qwen3_scheduler.h"

#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_kv_cache.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>

namespace llm {
namespace {
constexpr int32_t kVocabSize = 151936;
constexpr size_t kModelMaxContext = 32;
void invalid(const std::string& message) { throw std::invalid_argument("Qwen3RequestScheduler: " + message); }
void validate_sampling(const SamplingConfig& config) {
  if (!std::isfinite(config.temperature) || config.temperature < 0.0f)
    invalid("temperature must be finite and >= 0");
  if (config.temperature == 0.0f) return;
  if (config.top_k < 0 || config.top_k > kVocabSize) invalid("top_k must be in [0,151936]");
  if (!std::isfinite(config.top_p) || config.top_p <= 0.0f || config.top_p > 1.0f)
    invalid("top_p must be finite in (0,1]");
}
std::string id_text(uint64_t id) { return "request_id=" + std::to_string(id); }
bool supported_batch(size_t n) { return n == 1 || n == 2 || n == 4; }
}

struct Qwen3RequestScheduler::Entry {
  InferenceRequestConfig config;
  RequestState state = RequestState::Waiting;
  std::unique_ptr<Qwen3KvCache> cache;
  Tensor logits;
  std::vector<int32_t> generated;
  uint64_t draw_offset = 0;
  int32_t last_token = 0;
  std::chrono::steady_clock::time_point submitted_at;
  std::optional<uint64_t> first_token_round;
  std::optional<double> ttft_ms;
};

Qwen3RequestScheduler::Qwen3RequestScheduler(
    const Qwen3CudaModel& model, ContinuousSchedulerConfig admission_config,
    ClockNow clock_now)
    : model_(model), sampler_(kVocabSize), admission_config_(admission_config),
      clock_now_(std::move(clock_now)) {
  if (!clock_now_) clock_now_ = [] { return std::chrono::steady_clock::now(); };
  scheduler_started_at_ = now();
}
Qwen3RequestScheduler::~Qwen3RequestScheduler() = default;

void Qwen3RequestScheduler::submit(InferenceRequestConfig request) {
  if (request.request_id == 0) invalid("request_id must be non-zero");
  if (entries_.count(request.request_id) != 0) invalid(id_text(request.request_id) + " is already submitted");
  if (request.prompt_ids.empty()) invalid(id_text(request.request_id) + " prompt must be non-empty");
  if (request.max_seq_len == 0 || request.max_seq_len < request.prompt_ids.size())
    invalid(id_text(request.request_id) + " max_seq_len must be > 0 and >= prompt length");
  for (size_t i = 0; i < request.prompt_ids.size(); ++i)
    if (request.prompt_ids[i] < 0 || request.prompt_ids[i] >= kVocabSize)
      invalid(id_text(request.request_id) + " prompt_ids[" + std::to_string(i) + "] out of range");
  if (request.eos_token_id && (*request.eos_token_id < 0 || *request.eos_token_id >= kVocabSize))
    invalid(id_text(request.request_id) + " eos_token_id out of range");
  validate_sampling(request.sampling);
  auto entry = std::make_unique<Entry>();
  entry->config = std::move(request);
  entry->submitted_at = now();
  const uint64_t id = entry->config.request_id;
  entries_.emplace(id, std::move(entry));
  request_metric_indices_[id] = request_metrics_.size();
  SchedulerRequestMetrics request_metrics;
  request_metrics.request_id = id;
  request_metrics.queue_timeout_ms = entries_.at(id)->config.queue_timeout_ms;
  request_metrics.request_timeout_ms = entries_.at(id)->config.request_timeout_ms;
  request_metrics.submit_time_ms = std::chrono::duration<double, std::milli>(
      entries_.at(id)->submitted_at - scheduler_started_at_).count();
  request_metrics_.push_back(std::move(request_metrics));
  if (admission_config_.max_waiting_requests != 0 &&
      waiting_.size() >= admission_config_.max_waiting_requests) {
    finish(*entries_.at(id), "resource_exhausted");
  } else if (entries_.at(id)->config.max_new_tokens == 0) finish(*entries_.at(id), "max_new_tokens");
  else waiting_.push_back(id);
}

void Qwen3RequestScheduler::validate_budget(const StaticBatchBudget& budget) const {
  if (!supported_batch(budget.max_batch_size)) invalid("budget max_batch_size must be 1, 2, or 4");
  if (budget.max_prefill_tokens == 0 || budget.max_decode_tokens == 0 ||
      budget.max_context_len == 0)
    invalid("budget token and context limits must be > 0");
  if (budget.max_context_len > kModelMaxContext)
    invalid("budget max_context_len must be <= 32 for the current model");
  if (!std::isfinite(budget.max_padding_ratio) || budget.max_padding_ratio < 0.0 ||
      budget.max_padding_ratio > 1.0)
    invalid("budget max_padding_ratio must be finite in [0,1]");
}

void Qwen3RequestScheduler::validate_continuous_config(
    const ContinuousSchedulerConfig& config) const {
  if (config.max_consecutive_prefill == 0 || config.max_consecutive_decode == 0)
    invalid("continuous streak limits must be > 0");
}

size_t Qwen3RequestScheduler::prefill_batch_size_for_budget(
    const StaticBatchBudget& budget, size_t max_total_kv_cache_bytes) const {
  size_t selected_count = 0;
  const size_t max_candidates = std::min(budget.max_batch_size, waiting_.size());
  for (size_t count = 1; count <= max_candidates; ++count) {
    if (!supported_batch(count)) continue;
    size_t max_prompt = 0;
    size_t total = 0;
    bool compatible = true;
    for (size_t i = 0; i < count; ++i) {
      const Entry& entry = *entries_.at(waiting_[i]);
      const size_t prompt_len = entry.config.prompt_ids.size();
      if (prompt_len == 0 || prompt_len > budget.max_context_len ||
          entry.config.max_seq_len > kModelMaxContext ||
          entry.config.max_seq_len < prompt_len) {
        compatible = false;
        break;
      }
      max_prompt = std::max(max_prompt, prompt_len);
      total += prompt_len;
    }
    if (!compatible) continue;
    const size_t padded = count * max_prompt;
    const size_t padding = padded - total;
    const double ratio = padded == 0 ? 0.0 : static_cast<double>(padding) / padded;
    size_t candidate_kv = 0;
    bool kv_overflow = false;
    for (size_t i = 0; i < count; ++i) {
      const size_t bytes = kv_bytes_for_request(entries_.at(waiting_[i])->config);
      if (candidate_kv > std::numeric_limits<size_t>::max() - bytes) {
        kv_overflow = true;
        break;
      }
      candidate_kv += bytes;
    }
    const bool kv_fits = !kv_overflow &&
        (max_total_kv_cache_bytes == 0 ||
         (kv_resident_bytes_ <= max_total_kv_cache_bytes &&
          candidate_kv <= max_total_kv_cache_bytes - kv_resident_bytes_));
    if (padded <= budget.max_prefill_tokens && ratio <= budget.max_padding_ratio && kv_fits)
      selected_count = count;
  }
  return selected_count;
}

size_t Qwen3RequestScheduler::kv_bytes_for_request(
    const InferenceRequestConfig& request) const {
  constexpr size_t kLayers = 28;
  constexpr size_t kKvHeads = 8;
  constexpr size_t kHeadDim = 128;
  constexpr size_t kBytes = 2;  // Qwen3 cache is F16, for both K and V.
  if (request.max_seq_len == 0 ||
      request.max_seq_len > std::numeric_limits<size_t>::max() /
          (kLayers * 2 * kKvHeads * kHeadDim * kBytes))
    invalid("request max_seq_len overflows KV cache byte calculation");
  return request.max_seq_len * kLayers * 2 * kKvHeads * kHeadDim * kBytes;
}

void Qwen3RequestScheduler::allocate_cache(Entry& entry) {
  entry.cache = std::make_unique<Qwen3KvCache>(entry.config.max_seq_len);
  const size_t bytes = entry.cache->resident_bytes();
  if (kv_resident_bytes_ > std::numeric_limits<size_t>::max() - bytes)
    invalid("KV cache resident byte counter overflow");
  kv_resident_bytes_ += bytes;
  kv_peak_resident_bytes_ = std::max(kv_peak_resident_bytes_, kv_resident_bytes_);
}

void Qwen3RequestScheduler::release_cache(Entry& entry) {
  if (!entry.cache) return;
  const size_t bytes = entry.cache->resident_bytes();
  if (bytes > kv_resident_bytes_)
    invalid("KV cache resident byte counter underflow");
  kv_resident_bytes_ -= bytes;
  entry.cache.reset();
}

void Qwen3RequestScheduler::set_batch_observability(
    SchedulerBatchMetrics& metrics) const {
  metrics.finished_count = finished_count_;
  metrics.cancelled_count = cancelled_count_;
  metrics.timed_out_count = timed_out_count_;
  metrics.resource_rejected_count = resource_rejected_count_;
  metrics.kv_resident_bytes = kv_resident_bytes_;
  metrics.kv_peak_resident_bytes = kv_peak_resident_bytes_;
}

size_t Qwen3RequestScheduler::expire_requests() {
  const auto current = now();
  size_t expired = 0;
  std::deque<uint64_t> waiting_remaining;
  while (!waiting_.empty()) {
    const uint64_t id = waiting_.front();
    waiting_.pop_front();
    Entry& entry = *entries_.at(id);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        current - entry.submitted_at).count();
    const bool request_expired = entry.config.request_timeout_ms != 0 &&
        elapsed >= static_cast<long long>(entry.config.request_timeout_ms);
    const bool queue_expired = entry.config.queue_timeout_ms != 0 &&
        elapsed >= static_cast<long long>(entry.config.queue_timeout_ms);
    if (request_expired || queue_expired) {
      finish(entry, request_expired ? "request_timeout" : "queue_timeout");
      ++expired;
    } else {
      waiting_remaining.push_back(id);
    }
  }
  waiting_ = std::move(waiting_remaining);
  std::deque<uint64_t> decode_remaining;
  while (!decode_.empty()) {
    const uint64_t id = decode_.front();
    decode_.pop_front();
    Entry& entry = *entries_.at(id);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        current - entry.submitted_at).count();
    if (entry.config.request_timeout_ms != 0 &&
        elapsed >= static_cast<long long>(entry.config.request_timeout_ms)) {
      finish(entry, "request_timeout");
      ++expired;
    } else {
      decode_remaining.push_back(id);
    }
  }
  decode_ = std::move(decode_remaining);
  return expired;
}

size_t Qwen3RequestScheduler::decode_batch_size_for_budget(
    const StaticBatchBudget& budget) const {
  size_t selected_count = 0;
  const size_t max_candidates =
      std::min({budget.max_batch_size, budget.max_decode_tokens, decode_.size()});
  for (size_t count = 1; count <= max_candidates; ++count) {
    if (!supported_batch(count)) continue;
    bool compatible = true;
    for (size_t i = 0; i < count; ++i) {
      const Entry& entry = *entries_.at(decode_[i]);
      if (entry.state != RequestState::Decode || !entry.cache ||
          entry.cache->length() == 0 ||
          entry.cache->length() >= entry.cache->capacity() ||
          entry.cache->length() + 1 > budget.max_context_len ||
          entry.cache->length() + 1 > kModelMaxContext) {
        compatible = false;
        break;
      }
    }
    if (compatible) selected_count = count;
  }
  return selected_count;
}

void Qwen3RequestScheduler::record_first_token(Entry& entry, uint64_t round_id) {
  if (entry.first_token_round) return;
  entry.first_token_round = round_id;
  const auto now = this->now();
  entry.ttft_ms = std::chrono::duration<double, std::milli>(now - entry.submitted_at).count();
  const auto metric = request_metric_indices_.find(entry.config.request_id);
  if (metric != request_metric_indices_.end()) {
    auto& request_metrics = request_metrics_[metric->second];
    request_metrics.first_token_time_ms = std::chrono::duration<double, std::milli>(
        now - scheduler_started_at_).count();
    request_metrics.ttft_ms = entry.ttft_ms;
  }
}

void Qwen3RequestScheduler::record_admission(Entry& entry) {
  const auto metric = request_metric_indices_.find(entry.config.request_id);
  if (metric == request_metric_indices_.end()) return;
  auto& request_metrics = request_metrics_[metric->second];
  if (!request_metrics.admission_time_ms)
    request_metrics.admission_time_ms = std::chrono::duration<double, std::milli>(
        now() - scheduler_started_at_).count();
}

void Qwen3RequestScheduler::update_request_metrics(const Entry& entry) {
  const auto metric = request_metric_indices_.find(entry.config.request_id);
  if (metric != request_metric_indices_.end())
    request_metrics_[metric->second].generated_token_count = entry.generated.size();
}

void Qwen3RequestScheduler::append_batch_metrics(
    SchedulerBatchMetrics metrics, std::chrono::steady_clock::time_point started) {
  metrics.round_id = next_round_id_++;
  metrics.elapsed_ms = std::chrono::duration<double, std::milli>(
      now() - started).count();
  set_batch_observability(metrics);
  if (metrics.elapsed_ms < 0.0) metrics.elapsed_ms = 0.0;
  batch_metrics_.push_back(std::move(metrics));
}

size_t Qwen3RequestScheduler::prefill_waiting_static_batch(size_t max_batch_size) {
  if (max_batch_size != 1 && max_batch_size != 2 && max_batch_size != 4)
    invalid("static prefill max_batch_size must be 1, 2, or 4");
  if (waiting_.empty()) return 0;
  const uint64_t anchor_id = waiting_.front();
  const size_t anchor_size = entries_.at(anchor_id)->config.prompt_ids.size();
  size_t matching = 0;
  for (const uint64_t id : waiting_)
    if (entries_.at(id)->config.prompt_ids.size() == anchor_size) ++matching;
  const size_t supported_limit = matching >= 4 && max_batch_size >= 4 ? 4 :
                                 matching >= 2 && max_batch_size >= 2 ? 2 : 1;
  std::vector<uint64_t> selected;
  std::deque<uint64_t> remaining;
  for (const uint64_t id : waiting_) {
    if (selected.size() < supported_limit && entries_.at(id)->config.prompt_ids.size() == anchor_size)
      selected.push_back(id);
    else
      remaining.push_back(id);
  }
  waiting_ = std::move(remaining);
  std::vector<std::vector<int32_t>> prompts;
  std::vector<Qwen3KvCache*> caches;
  prompts.reserve(selected.size()); caches.reserve(selected.size());
  for (const uint64_t id : selected) {
    Entry& entry = *entries_.at(id);
    entry.state = RequestState::Prefill;
    allocate_cache(entry);
    prompts.push_back(entry.config.prompt_ids);
    caches.push_back(entry.cache.get());
  }
  Tensor logits = model_.prefill_logits_batch_with_caches(prompts, caches);
  for (const uint64_t id : selected) record_admission(*entries_.at(id));
  for (size_t b = 0; b < selected.size(); ++b) {
    Entry& entry = *entries_.at(selected[b]);
    entry.last_token = sampler_.sample_last_token_of_batch(logits, b, entry.config.sampling, 0);
    entry.draw_offset = 1;
    entry.generated.push_back(entry.last_token);
    update_request_metrics(entry);
    record_first_token(entry, next_round_id_);
    if (entry.config.eos_token_id && entry.last_token == *entry.config.eos_token_id) { finish(entry, "eos"); continue; }
    if (entry.generated.size() >= entry.config.max_new_tokens) { finish(entry, "max_new_tokens"); continue; }
    if (entry.cache->length() >= entry.cache->capacity()) { finish(entry, "cache_capacity"); continue; }
    entry.state = RequestState::Decode;
    decode_.push_back(entry.config.request_id);
  }
  return selected.size();
}

size_t Qwen3RequestScheduler::prefill_waiting_padded_static_batch(size_t max_batch_size) {
  if (max_batch_size != 1 && max_batch_size != 2 && max_batch_size != 4)
    invalid("padded static prefill max_batch_size must be 1, 2, or 4");
  if (waiting_.empty()) return 0;
  // This deliberately consumes only the FIFO prefix. With exactly three
  // waiting requests, leave the third request waiting so a later submission
  // can form a clean batch rather than changing the existing equal-length API.
  size_t count = std::min(max_batch_size, waiting_.size());
  if (waiting_.size() == 3) count = std::min<size_t>(count, 2);
  std::vector<uint64_t> selected;
  selected.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    selected.push_back(waiting_.front());
    waiting_.pop_front();
  }
  std::vector<std::vector<int32_t>> prompts;
  std::vector<Qwen3KvCache*> caches;
  prompts.reserve(count); caches.reserve(count);
  for (uint64_t id : selected) {
    Entry& entry = *entries_.at(id);
    entry.state = RequestState::Prefill;
    allocate_cache(entry);
    prompts.push_back(entry.config.prompt_ids);
    caches.push_back(entry.cache.get());
  }
  try {
    PaddedPrefillBatch metadata = build_padded_prefill_batch(prompts, caches);
    Tensor logits = model_.prefill_logits_padded_batch_with_caches(metadata);
    Tensor last_logits = select_last_valid_logits(logits, metadata.valid_lengths);
    for (const uint64_t id : selected) record_admission(*entries_.at(id));
    for (size_t b = 0; b < selected.size(); ++b) {
      Entry& entry = *entries_.at(selected[b]);
      entry.last_token = sampler_.sample_row(last_logits, b, entry.config.sampling, 0);
      entry.draw_offset = 1;
      entry.generated.push_back(entry.last_token);
      update_request_metrics(entry);
      record_first_token(entry, next_round_id_);
      if (entry.config.eos_token_id && entry.last_token == *entry.config.eos_token_id) { finish(entry, "eos"); continue; }
      if (entry.generated.size() >= entry.config.max_new_tokens) { finish(entry, "max_new_tokens"); continue; }
      if (entry.cache->length() >= entry.cache->capacity()) { finish(entry, "cache_capacity"); continue; }
      entry.state = RequestState::Decode;
      decode_.push_back(entry.config.request_id);
    }
  } catch (...) {
    for (auto it = selected.rbegin(); it != selected.rend(); ++it) {
      Entry& entry = *entries_.at(*it);
      entry.state = RequestState::Waiting;
      release_cache(entry);
      waiting_.push_front(*it);
    }
    throw;
  }
  return selected.size();
}

size_t Qwen3RequestScheduler::decode_active_static_batch(size_t max_batch_size) {
  if (max_batch_size != 1 && max_batch_size != 2 && max_batch_size != 4)
    invalid("static decode max_batch_size must be 1, 2, or 4");
  if (decode_.empty()) return 0;
  const uint64_t anchor_id = decode_.front();
  const size_t anchor_length = entries_.at(anchor_id)->cache->length();
  size_t matching = 0;
  for (const uint64_t id : decode_)
    if (entries_.at(id)->cache && entries_.at(id)->cache->length() == anchor_length) ++matching;
  const size_t supported_limit = matching >= 4 && max_batch_size >= 4 ? 4 :
                                 matching >= 2 && max_batch_size >= 2 ? 2 : 1;
  std::vector<uint64_t> selected;
  std::deque<uint64_t> remaining;
  for (const uint64_t id : decode_) {
    Entry& entry = *entries_.at(id);
    if (selected.size() < supported_limit && entry.cache &&
        entry.cache->length() == anchor_length)
      selected.push_back(id);
    else
      remaining.push_back(id);
  }
  decode_ = std::move(remaining);
  std::vector<int32_t> next_input_ids;
  std::vector<Qwen3KvCache*> caches;
  next_input_ids.reserve(selected.size()); caches.reserve(selected.size());
  for (const uint64_t id : selected) {
    Entry& entry = *entries_.at(id);
    next_input_ids.push_back(entry.last_token);
    caches.push_back(entry.cache.get());
  }
  Tensor logits;
  try {
    logits = model_.decode_logits_batch_with_caches(next_input_ids, caches);
  } catch (...) {
    std::deque<uint64_t> restored;
    for (const uint64_t id : selected) restored.push_back(id);
    for (const uint64_t id : decode_) restored.push_back(id);
    decode_ = std::move(restored);
    throw;
  }
  for (size_t b = 0; b < selected.size(); ++b) {
    Entry& entry = *entries_.at(selected[b]);
    entry.last_token = sampler_.sample_row(logits, b, entry.config.sampling,
                                           entry.draw_offset++);
    entry.generated.push_back(entry.last_token);
    update_request_metrics(entry);
    if (entry.config.eos_token_id && entry.last_token == *entry.config.eos_token_id) {
      finish(entry, "eos");
      continue;
    }
    if (entry.generated.size() >= entry.config.max_new_tokens) {
      finish(entry, "max_new_tokens");
      continue;
    }
    if (entry.cache->length() >= entry.cache->capacity()) {
      finish(entry, "cache_capacity");
      continue;
    }
    decode_.push_back(entry.config.request_id);
  }
  return selected.size();
}

size_t Qwen3RequestScheduler::decode_active_variable_length_batch(size_t max_batch_size) {
  if (max_batch_size != 1 && max_batch_size != 2 && max_batch_size != 4)
    invalid("variable decode max_batch_size must be 1, 2, or 4");
  if (decode_.empty()) return 0;
  const size_t limit = decode_.size() >= 4 && max_batch_size >= 4 ? 4 :
                       decode_.size() >= 2 && max_batch_size >= 2 ? 2 : 1;
  std::vector<uint64_t> selected;
  std::deque<uint64_t> remaining = decode_;
  while (selected.size() < limit) {
    selected.push_back(remaining.front());
    remaining.pop_front();
  }
  decode_ = std::move(remaining);
  std::vector<int32_t> next_input_ids;
  std::vector<Qwen3KvCache*> caches;
  for (const uint64_t id : selected) {
    Entry& entry = *entries_.at(id);
    next_input_ids.push_back(entry.last_token);
    caches.push_back(entry.cache.get());
  }
  Tensor logits;
  try {
    logits = model_.decode_logits_variable_length_batch_with_caches(next_input_ids, caches);
  } catch (...) {
    std::deque<uint64_t> restored;
    for (const uint64_t id : selected) restored.push_back(id);
    for (const uint64_t id : decode_) restored.push_back(id);
    decode_ = std::move(restored);
    throw;
  }
  for (size_t b = 0; b < selected.size(); ++b) {
    Entry& entry = *entries_.at(selected[b]);
    entry.last_token = sampler_.sample_row(logits, b, entry.config.sampling,
                                           entry.draw_offset++);
    entry.generated.push_back(entry.last_token);
    update_request_metrics(entry);
    if (entry.config.eos_token_id && entry.last_token == *entry.config.eos_token_id) {
      finish(entry, "eos");
      continue;
    }
    if (entry.generated.size() >= entry.config.max_new_tokens) {
      finish(entry, "max_new_tokens");
      continue;
    }
    if (entry.cache->length() >= entry.cache->capacity()) {
      finish(entry, "cache_capacity");
      continue;
    }
    decode_.push_back(entry.config.request_id);
  }
  return selected.size();
}

size_t Qwen3RequestScheduler::prefill_waiting_budgeted_static_batch(
    const StaticBatchBudget& budget) {
  return prefill_waiting_budgeted_static_batch_with_kv_budget(budget, 0);
}

size_t Qwen3RequestScheduler::prefill_waiting_budgeted_static_batch_with_kv_budget(
    const StaticBatchBudget& budget, size_t max_total_kv_cache_bytes) {
  validate_budget(budget);
  const auto started = now();
  SchedulerBatchMetrics metrics;
  metrics.kind = SchedulerBatchKind::Prefill;
  metrics.waiting_before = waiting_.size();
  metrics.decode_before = decode_.size();
  const uint64_t round_id = next_round_id_;
  const size_t selected_count = prefill_batch_size_for_budget(
      budget, max_total_kv_cache_bytes);
  size_t valid_tokens = 0;
  size_t padded_tokens = 0;
  double padding_ratio = 0.0;
  if (selected_count != 0) {
    for (size_t i = 0; i < selected_count; ++i) {
      const size_t prompt_len = entries_.at(waiting_[i])->config.prompt_ids.size();
      valid_tokens += prompt_len;
      padded_tokens = std::max(padded_tokens, prompt_len);
    }
    padded_tokens *= selected_count;
    padding_ratio = static_cast<double>(padded_tokens - valid_tokens) / padded_tokens;
  }

  if (selected_count == 0) {
    metrics.skipped_budget_count = waiting_.empty() ? 0 : 1;
    metrics.active_after = waiting_.size() + decode_.size();
    append_batch_metrics(std::move(metrics), started);
    return 0;
  }

  std::vector<uint64_t> selected;
  std::vector<std::vector<int32_t>> prompts;
  std::vector<Qwen3KvCache*> caches;
  try {
    selected.reserve(selected_count);
    prompts.reserve(selected_count);
    caches.reserve(selected_count);
    for (size_t i = 0; i < selected_count; ++i) {
      const uint64_t id = waiting_.front();
      waiting_.pop_front();
      selected.push_back(id);
      Entry& entry = *entries_.at(id);
      entry.state = RequestState::Prefill;
      if (prefill_prepare_failure_after_for_testing_) {
        if (*prefill_prepare_failure_after_for_testing_ == 0) {
          prefill_prepare_failure_after_for_testing_.reset();
          throw std::bad_alloc();
        }
        --*prefill_prepare_failure_after_for_testing_;
      }
      allocate_cache(entry);
      prompts.push_back(entry.config.prompt_ids);
      caches.push_back(entry.cache.get());
    }
    PaddedPrefillBatch metadata = build_padded_prefill_batch(prompts, caches);
    Tensor logits = model_.prefill_logits_padded_batch_with_caches(metadata);
    Tensor last_logits = select_last_valid_logits(logits, metadata.valid_lengths);
    for (const uint64_t id : selected) record_admission(*entries_.at(id));
    for (size_t b = 0; b < selected_count; ++b) {
      Entry& entry = *entries_.at(selected[b]);
      entry.last_token = sampler_.sample_row(last_logits, b, entry.config.sampling, 0);
      entry.draw_offset = 1;
      entry.generated.push_back(entry.last_token);
      update_request_metrics(entry);
      record_first_token(entry, round_id);
      if (entry.config.eos_token_id && entry.last_token == *entry.config.eos_token_id) { finish(entry, "eos"); continue; }
      if (entry.generated.size() >= entry.config.max_new_tokens) { finish(entry, "max_new_tokens"); continue; }
      if (entry.cache->length() >= entry.cache->capacity()) { finish(entry, "cache_capacity"); continue; }
      entry.state = RequestState::Decode;
      decode_.push_back(entry.config.request_id);
    }
  } catch (...) {
    for (auto it = selected.rbegin(); it != selected.rend(); ++it) {
      Entry& entry = *entries_.at(*it);
      entry.state = RequestState::Waiting;
      release_cache(entry);
      waiting_.push_front(*it);
    }
    throw;
  }
  metrics.selected_request_ids = selected;
  metrics.batch_size = selected_count;
  metrics.valid_tokens = valid_tokens;
  metrics.padded_tokens = padded_tokens;
  metrics.padding_tokens = padded_tokens - valid_tokens;
  metrics.padding_ratio = padding_ratio;
  metrics.active_after = waiting_.size() + decode_.size();
  append_batch_metrics(std::move(metrics), started);
  return selected_count;
}

size_t Qwen3RequestScheduler::decode_active_budgeted_static_batch(
    const StaticBatchBudget& budget) {
  validate_budget(budget);
  const auto started = now();
  SchedulerBatchMetrics metrics;
  metrics.kind = SchedulerBatchKind::Decode;
  metrics.waiting_before = waiting_.size();
  metrics.decode_before = decode_.size();
  const size_t selected_count = decode_batch_size_for_budget(budget);
  if (selected_count == 0) {
    metrics.skipped_budget_count = decode_.empty() ? 0 : 1;
    metrics.active_after = waiting_.size() + decode_.size();
    append_batch_metrics(std::move(metrics), started);
    return 0;
  }
  std::vector<uint64_t> selected;
  selected.reserve(selected_count);
  for (size_t i = 0; i < selected_count; ++i) {
    selected.push_back(decode_.front());
    decode_.pop_front();
  }
  std::vector<int32_t> next_input_ids;
  std::vector<Qwen3KvCache*> caches;
  for (uint64_t id : selected) {
    Entry& entry = *entries_.at(id);
    next_input_ids.push_back(entry.last_token);
    caches.push_back(entry.cache.get());
  }
  try {
    Tensor logits = model_.decode_logits_variable_length_batch_with_caches(next_input_ids, caches);
    for (size_t b = 0; b < selected_count; ++b) {
      Entry& entry = *entries_.at(selected[b]);
      entry.last_token = sampler_.sample_row(logits, b, entry.config.sampling, entry.draw_offset++);
      entry.generated.push_back(entry.last_token);
      update_request_metrics(entry);
      if (entry.config.eos_token_id && entry.last_token == *entry.config.eos_token_id) { finish(entry, "eos"); continue; }
      if (entry.generated.size() >= entry.config.max_new_tokens) { finish(entry, "max_new_tokens"); continue; }
      if (entry.cache->length() >= entry.cache->capacity()) { finish(entry, "cache_capacity"); continue; }
      decode_.push_back(entry.config.request_id);
    }
  } catch (...) {
    std::deque<uint64_t> restored;
    for (uint64_t id : selected) restored.push_back(id);
    for (uint64_t id : decode_) restored.push_back(id);
    decode_ = std::move(restored);
    throw;
  }
  metrics.selected_request_ids = selected;
  metrics.batch_size = selected_count;
  metrics.valid_tokens = selected_count;
  metrics.decode_tokens = selected_count;
  metrics.active_after = waiting_.size() + decode_.size();
  append_batch_metrics(std::move(metrics), started);
  return selected_count;
}

std::string Qwen3RequestScheduler::export_batch_metrics_csv() const {
  std::ostringstream out;
  out << "round_id,kind,selected_request_ids,batch_size,waiting_before,decode_before,active_after,"
      << "valid_tokens,padded_tokens,padding_tokens,padding_ratio,decode_tokens,skipped_budget_count,elapsed_ms,"
      << "consecutive_prefill,consecutive_decode,finished_count,cancelled_count,timed_out_count,"
      << "resource_rejected_count,kv_resident_bytes,kv_peak_resident_bytes\n";
  out << std::setprecision(9);
  for (const auto& m : batch_metrics_) {
    out << m.round_id << ',' << (m.kind == SchedulerBatchKind::Prefill ? "prefill" : "decode") << ',';
    for (size_t i = 0; i < m.selected_request_ids.size(); ++i) {
      if (i) out << '|';
      out << m.selected_request_ids[i];
    }
    out << ',' << m.batch_size << ',' << m.waiting_before << ',' << m.decode_before << ','
        << m.active_after << ',' << m.valid_tokens << ',' << m.padded_tokens << ','
        << m.padding_tokens << ',' << m.padding_ratio << ',' << m.decode_tokens << ','
        << m.skipped_budget_count << ',' << m.elapsed_ms << ','
        << m.consecutive_prefill << ',' << m.consecutive_decode << ',' << m.finished_count << ','
        << m.cancelled_count << ',' << m.timed_out_count << ',' << m.resource_rejected_count << ','
        << m.kv_resident_bytes << ',' << m.kv_peak_resident_bytes << '\n';
  }
  return out.str();
}

std::string Qwen3RequestScheduler::export_request_metrics_csv() const {
  std::ostringstream out;
  out << "request_id,submit_time_ms,admission_time_ms,first_token_time_ms,ttft_ms,"
      << "generated_token_count,completion_time_ms,stop_reason,queue_timeout_ms,request_timeout_ms\n";
  out << std::setprecision(9);
  for (const auto& metrics : request_metrics_) {
    out << metrics.request_id << ',' << metrics.submit_time_ms << ',';
    if (metrics.admission_time_ms) out << *metrics.admission_time_ms;
    out << ',';
    if (metrics.first_token_time_ms) out << *metrics.first_token_time_ms;
    out << ',';
    if (metrics.ttft_ms) out << *metrics.ttft_ms;
    out << ',' << metrics.generated_token_count << ',';
    if (metrics.completion_time_ms) out << *metrics.completion_time_ms;
    out << ',' << metrics.stop_reason << ',' << metrics.queue_timeout_ms << ','
        << metrics.request_timeout_ms << '\n';
  }
  return out.str();
}

void Qwen3RequestScheduler::finish(Entry& entry, const std::string& reason) {
  if (entry.state == RequestState::Finished) return;
  entry.state = RequestState::Finished;
  FinishedRequest result{entry.config.request_id, entry.generated, reason,
                         entry.cache ? entry.cache->length() : 0};
  finished_.push_back(std::move(result));
  entry.logits = Tensor();
  update_request_metrics(entry);
  const auto metric = request_metric_indices_.find(entry.config.request_id);
  if (metric != request_metric_indices_.end()) {
    auto& request_metrics = request_metrics_[metric->second];
    request_metrics.completion_time_ms = std::chrono::duration<double, std::milli>(
        now() - scheduler_started_at_).count();
    request_metrics.stop_reason = reason;
  }
  ++finished_count_;
  if (reason == "cancelled") ++cancelled_count_;
  if (reason == "queue_timeout" || reason == "request_timeout") ++timed_out_count_;
  if (reason == "resource_exhausted") ++resource_rejected_count_;
  release_cache(entry);
}

void Qwen3RequestScheduler::run_prefill(Entry& entry) {
  entry.state = RequestState::Prefill;
  allocate_cache(entry);
  entry.logits = model_.prefill_logits_with_cache(entry.config.prompt_ids, *entry.cache);
  record_admission(entry);
  entry.last_token = sampler_.sample_last_row(entry.logits, entry.config.sampling, 0);
  entry.draw_offset = 1;
  entry.generated.push_back(entry.last_token);
  update_request_metrics(entry);
  record_first_token(entry, next_round_id_);
  if (entry.config.eos_token_id && entry.last_token == *entry.config.eos_token_id) return finish(entry, "eos");
  if (entry.generated.size() >= entry.config.max_new_tokens) return finish(entry, "max_new_tokens");
  if (entry.cache->length() >= entry.cache->capacity()) return finish(entry, "cache_capacity");
  entry.state = RequestState::Decode;
  decode_.push_back(entry.config.request_id);
}

void Qwen3RequestScheduler::run_decode(Entry& entry) {
  entry.logits = model_.decode_logits(entry.last_token, *entry.cache);
  entry.last_token = sampler_.sample_last_row(entry.logits, entry.config.sampling, entry.draw_offset++);
  entry.generated.push_back(entry.last_token);
  update_request_metrics(entry);
  if (entry.config.eos_token_id && entry.last_token == *entry.config.eos_token_id) return finish(entry, "eos");
  if (entry.generated.size() >= entry.config.max_new_tokens) return finish(entry, "max_new_tokens");
  if (entry.cache->length() >= entry.cache->capacity()) return finish(entry, "cache_capacity");
}

bool Qwen3RequestScheduler::step() {
  while (!waiting_.empty()) {
    const uint64_t id = waiting_.front(); waiting_.pop_front();
    auto it = entries_.find(id);
    if (it != entries_.end() && it->second->state == RequestState::Waiting) { run_prefill(*it->second); return true; }
  }
  while (!decode_.empty()) {
    const uint64_t id = decode_.front(); decode_.pop_front();
    auto it = entries_.find(id);
    if (it != entries_.end() && it->second->state == RequestState::Decode) {
      run_decode(*it->second);
      if (it->second->state == RequestState::Decode) decode_.push_back(id);
      return true;
    }
  }
  return false;
}

bool Qwen3RequestScheduler::cancel_request(uint64_t request_id) {
  auto it = entries_.find(request_id);
  if (it == entries_.end() || it->second->state == RequestState::Finished)
    return false;
  Entry& entry = *it->second;
  if (entry.state == RequestState::Waiting) {
    waiting_.erase(std::remove(waiting_.begin(), waiting_.end(), request_id), waiting_.end());
  } else if (entry.state == RequestState::Decode) {
    decode_.erase(std::remove(decode_.begin(), decode_.end(), request_id), decode_.end());
  }
  finish(entry, "cancelled");
  return true;
}

bool Qwen3RequestScheduler::step_continuous(const StaticBatchBudget& budget) {
  return step_continuous(budget, admission_config_);
}

bool Qwen3RequestScheduler::step_continuous(
    const StaticBatchBudget& budget, const ContinuousSchedulerConfig& config) {
  validate_budget(budget);
  validate_continuous_config(config);
  admission_config_ = config;
  expire_requests();
  StaticBatchBudget prefill_budget = budget;
  bool prefill_capacity_available = true;
  if (config.max_active_requests != 0) {
    prefill_capacity_available = decode_.size() < config.max_active_requests;
    if (prefill_capacity_available) {
      const size_t remaining_capacity = config.max_active_requests - decode_.size();
      const size_t supported_capacity = remaining_capacity >= 4 ? 4 :
                                        remaining_capacity >= 2 ? 2 : 1;
      prefill_budget.max_batch_size = std::min(prefill_budget.max_batch_size,
                                               supported_capacity);
    }
  }
  const bool prefill_ready = prefill_capacity_available &&
                             prefill_batch_size_for_budget(
                                 prefill_budget, config.max_total_kv_cache_bytes) != 0;
  const bool decode_ready = decode_batch_size_for_budget(budget) != 0;
  if (!prefill_ready && !decode_ready) return false;

  bool choose_prefill = prefill_ready;
  if (prefill_ready && decode_ready) {
    if (continuous_decode_streak_ >= config.max_consecutive_decode) choose_prefill = true;
    else if (continuous_prefill_streak_ >= config.max_consecutive_prefill) choose_prefill = false;
    else choose_prefill = true;
  }
  if (!choose_prefill && !decode_ready) choose_prefill = true;
  if (choose_prefill && !prefill_ready) choose_prefill = false;

  const size_t executed = choose_prefill
      ? prefill_waiting_budgeted_static_batch_with_kv_budget(
            prefill_budget, config.max_total_kv_cache_bytes)
      : decode_active_budgeted_static_batch(budget);
  if (executed == 0) return false;
  if (choose_prefill) {
    ++continuous_prefill_streak_;
    continuous_decode_streak_ = 0;
    batch_metrics_.back().consecutive_prefill = continuous_prefill_streak_;
    batch_metrics_.back().consecutive_decode = continuous_decode_streak_;
  } else {
    ++continuous_decode_streak_;
    continuous_prefill_streak_ = 0;
    batch_metrics_.back().consecutive_prefill = continuous_prefill_streak_;
    batch_metrics_.back().consecutive_decode = continuous_decode_streak_;
  }
  return true;
}

bool Qwen3RequestScheduler::has_unfinished() const { return waiting_count() != 0 || decode_count() != 0; }
RequestState Qwen3RequestScheduler::state(uint64_t request_id) const {
  auto it = entries_.find(request_id);
  if (it == entries_.end()) invalid(id_text(request_id) + " is unknown or already collected");
  return it->second->state;
}
std::vector<FinishedRequest> Qwen3RequestScheduler::take_finished() {
  std::vector<FinishedRequest> result;
  while (!finished_.empty()) { result.push_back(std::move(finished_.front())); entries_.erase(result.back().request_id); finished_.pop_front(); }
  return result;
}
size_t Qwen3RequestScheduler::waiting_count() const { return waiting_.size(); }
size_t Qwen3RequestScheduler::decode_count() const { return decode_.size(); }

std::optional<uint64_t> Qwen3RequestScheduler::first_token_round(uint64_t request_id) const {
  auto it = entries_.find(request_id);
  if (it == entries_.end()) invalid(id_text(request_id) + " is unknown or already collected");
  return it->second->first_token_round;
}

std::optional<double> Qwen3RequestScheduler::ttft_ms(uint64_t request_id) const {
  auto it = entries_.find(request_id);
  if (it == entries_.end()) invalid(id_text(request_id) + " is unknown or already collected");
  return it->second->ttft_ms;
}

}  // namespace llm
