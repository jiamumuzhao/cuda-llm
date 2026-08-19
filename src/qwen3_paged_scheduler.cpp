#include "llm/qwen3_paged_scheduler.h"
#include "llm/qwen3_cuda_model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace llm {
namespace {
constexpr int32_t kVocab = 151936;
constexpr size_t kBlockSize = 16;
[[noreturn]] void error(const std::string& s) {
  throw std::invalid_argument("Qwen3PagedRequestScheduler: " + s);
}
void validate_sampling(const SamplingConfig& s) {
  if (!std::isfinite(s.temperature) || s.temperature < 0) error("invalid temperature");
  if (s.temperature == 0) return;
  if (s.top_k < 0 || s.top_k > kVocab) error("invalid top_k");
  if (!std::isfinite(s.top_p) || s.top_p <= 0 || s.top_p > 1) error("invalid top_p");
}
size_t age_ms(std::chrono::steady_clock::time_point now,
              std::chrono::steady_clock::time_point start) {
  return static_cast<size_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      now - start).count());
}
}

struct Qwen3PagedRequestScheduler::Entry {
  InferenceRequestConfig request;
  RequestState state = RequestState::Waiting;
  std::unique_ptr<Qwen3PagedKvCache> cache;
  std::vector<int32_t> generated;
  int32_t last_token = 0;
  uint64_t draw_offset = 0;
  std::chrono::steady_clock::time_point submitted_at;
};

Qwen3PagedRequestScheduler::Qwen3PagedRequestScheduler(
    const Qwen3CudaModel& model, PagedKvCachePool& pool, PagedSchedulerConfig config)
    : model_(model), pool_(pool), config_(std::move(config)), sampler_(kVocab) {
  if (config_.max_seq_len == 0 || config_.max_seq_len > 32)
    error("max_seq_len must be in [1,32]");
  if (config_.max_decode_batch_size != 1 &&
      config_.max_decode_batch_size != 2 &&
      config_.max_decode_batch_size != 4)
    error("max_decode_batch_size must be 1, 2, or 4");
  const auto& c = pool_.config();
  if (c.dtype != DType::F16 || c.num_layers != 28 || c.num_kv_heads != 8 ||
      c.block_size != kBlockSize || c.head_dim != 128)
    error("pool is not Qwen3-compatible F16");
  if (!config_.clock_now)
    config_.clock_now = [] { return std::chrono::steady_clock::now(); };
  started_at_ = config_.clock_now();
}
Qwen3PagedRequestScheduler::~Qwen3PagedRequestScheduler() = default;

void Qwen3PagedRequestScheduler::validate_request(
    const InferenceRequestConfig& r) const {
  if (!r.request_id || entries_.count(r.request_id)) error("invalid or duplicate request_id");
  if (r.prompt_ids.empty()) error("prompt must be non-empty");
  if (!r.max_seq_len || r.max_seq_len > config_.max_seq_len ||
      r.prompt_ids.size() > r.max_seq_len) error("invalid max_seq_len/prompt length");
  for (int32_t id : r.prompt_ids)
    if (id < 0 || id >= kVocab) error("prompt token out of range");
  if (r.eos_token_id && (*r.eos_token_id < 0 || *r.eos_token_id >= kVocab))
    error("eos token out of range");
  validate_sampling(r.sampling);
}
void Qwen3PagedRequestScheduler::submit(InferenceRequestConfig r) {
  validate_request(r);
  auto e = std::make_unique<Entry>();
  e->request = std::move(r); e->submitted_at = config_.clock_now();
  const uint64_t id = e->request.request_id;
  entries_.emplace(id, std::move(e));
  Entry& stored = *entries_.at(id);
  if (config_.max_waiting_requests && waiting_.size() >= config_.max_waiting_requests)
    finish(stored, "resource_exhausted");
  else if (!stored.request.max_new_tokens) finish(stored, "max_new_tokens");
  else waiting_.push_back(id);
}
void Qwen3PagedRequestScheduler::finish(Entry& e, const std::string& reason) {
  if (e.state == RequestState::Finished) return;
  const size_t length = e.cache ? e.cache->length() : 0;
  if (e.cache) { e.cache->release_all(); e.cache.reset(); }
  e.state = RequestState::Finished;
  finished_.push_back({e.request.request_id, e.generated, reason, length});
}
void Qwen3PagedRequestScheduler::expire() {
  const auto now = config_.clock_now();
  auto check = [&](std::deque<uint64_t>& q, bool queue_timeout) {
    std::deque<uint64_t> keep;
    while (!q.empty()) {
      uint64_t id = q.front(); q.pop_front(); Entry& e = *entries_.at(id);
      size_t age = age_ms(now, e.submitted_at);
      bool request_expired = e.request.request_timeout_ms &&
                             age >= e.request.request_timeout_ms;
      bool queue_expired = queue_timeout && e.request.queue_timeout_ms &&
                          age >= e.request.queue_timeout_ms;
      if (request_expired || queue_expired)
        finish(e, request_expired ? "request_timeout" : "queue_timeout");
      else keep.push_back(id);
    }
    q = std::move(keep);
  };
  check(waiting_, true); check(active_, false);
}
bool Qwen3PagedRequestScheduler::remove_from_queue(
    std::deque<uint64_t>& q, uint64_t id) {
  auto old = q.size(); q.erase(std::remove(q.begin(), q.end(), id), q.end());
  return q.size() != old;
}
bool Qwen3PagedRequestScheduler::cancel_request(uint64_t id) {
  auto it = entries_.find(id);
  if (it == entries_.end() || it->second->state == RequestState::Finished) return false;
  if (!remove_from_queue(waiting_, id) && !remove_from_queue(active_, id)) return false;
  finish(*it->second, "cancelled"); return true;
}
bool Qwen3PagedRequestScheduler::admit_one() {
  if (waiting_.empty() || (config_.max_active_requests &&
      active_.size() >= config_.max_active_requests)) return false;
  const uint64_t id = waiting_.front(); Entry& e = *entries_.at(id);
  const size_t required = (e.request.max_seq_len + kBlockSize - 1) / kBlockSize;
  if (pool_.free_block_count() < required) return false;
  waiting_.pop_front();
  std::unique_ptr<Qwen3PagedKvCache> target;
  try {
    target = std::make_unique<Qwen3PagedKvCache>(pool_, config_.max_seq_len);
    if (config_.before_seed_for_testing) config_.before_seed_for_testing();
    Tensor logits = model_.prefill_logits_paged(e.request.prompt_ids, *target);
    e.cache = std::move(target);
    e.last_token = sampler_.sample_last_row(logits, e.request.sampling, e.draw_offset++);
    e.generated.push_back(e.last_token);
    if (e.request.eos_token_id && e.last_token == *e.request.eos_token_id)
      finish(e, "eos");
    else if (e.generated.size() >= e.request.max_new_tokens)
      finish(e, "max_new_tokens");
    else { e.state = RequestState::Decode; active_.push_back(id); }
    record({0, "paged_prefill", id}); return true;
  } catch (const BlockPoolExhausted&) {
    target.reset();
    e.state = RequestState::Waiting;
    waiting_.push_front(id);
    record({0, "deferred", id, 0, 0, 0, 0, 0, 0, peak_resident_bytes_, true});
    return false;
  } catch (const std::exception&) {
    target.reset();
    finish(e, "error"); record({0, "error", id}); return true;
  }
}
bool Qwen3PagedRequestScheduler::decode_batch() {
  decode_batch_error_pending_ = false;
  if (active_.empty()) return false;

  std::size_t eligible = 0;
  for (uint64_t id : active_) {
    Entry& e = *entries_.at(id);
    if (e.state == RequestState::Decode && e.cache &&
        e.cache->length() < e.request.max_seq_len)
      ++eligible;
  }
  std::size_t desired = std::min(config_.max_decode_batch_size, eligible);
  // The model API supports only B=1/2/4. Keep a lone third request in the
  // FIFO for the next scheduler step instead of issuing an unsupported B=3.
  if (desired == 3) desired = 2;

  std::vector<uint64_t> selected;
  selected.reserve(desired);
  std::deque<uint64_t> keep;
  bool finished_at_limit = false;
  while (!active_.empty()) {
    const uint64_t id = active_.front();
    active_.pop_front();
    Entry& e = *entries_.at(id);
    if (!e.cache || e.cache->length() >= e.request.max_seq_len) {
      finish(e, "max_seq_len");
      finished_at_limit = true;
    } else if (selected.size() < desired) {
      selected.push_back(id);
    } else {
      keep.push_back(id);
    }
  }
  active_ = std::move(keep);
  if (selected.empty()) return finished_at_limit;

  std::vector<int32_t> next_input_ids;
  std::vector<Qwen3PagedKvCache*> caches;
  next_input_ids.reserve(selected.size());
  caches.reserve(selected.size());
  for (uint64_t id : selected) {
    Entry& e = *entries_.at(id);
    next_input_ids.push_back(e.last_token);
    caches.push_back(e.cache.get());
  }

  try {
    Tensor logits = model_.decode_logits_paged_batch(next_input_ids, caches);
    for (std::size_t row = 0; row < selected.size(); ++row) {
      Entry& e = *entries_.at(selected[row]);
      e.last_token = sampler_.sample_row(logits, row, e.request.sampling,
                                         e.draw_offset++);
      e.generated.push_back(e.last_token);
      if (e.request.eos_token_id && e.last_token == *e.request.eos_token_id)
        finish(e, "eos");
      else if (e.generated.size() >= e.request.max_new_tokens)
        finish(e, "max_new_tokens");
      else if (e.cache->length() >= e.request.max_seq_len)
        finish(e, "max_seq_len");
      else {
        e.state = RequestState::Decode;
        active_.push_back(selected[row]);
      }
    }
    PagedSchedulerMetrics m;
    m.action = "decode";
    m.selected_request_id = selected.front();
    m.selected_request_ids = selected;
    m.decode_batch_size = selected.size();
    record(std::move(m));
    return true;
  } catch (...) {
    for (auto it = selected.rbegin(); it != selected.rend(); ++it)
      active_.push_front(*it);
    decode_batch_error_pending_ = true;
    PagedSchedulerMetrics m;
    m.action = "decode_error";
    m.selected_request_id = selected.front();
    m.selected_request_ids = selected;
    m.decode_batch_size = selected.size();
    m.decode_batch_error = true;
    record(std::move(m));
    return false;
  }
}
void Qwen3PagedRequestScheduler::record(PagedSchedulerMetrics m) {
  m.step_id = next_step_id_++; m.waiting = waiting_.size(); m.active = active_.size();
  m.finished = finished_.size(); m.pool_free_blocks = pool_.free_block_count();
  m.pool_used_blocks = pool_.used_block_count();
  m.paged_resident_bytes = m.pool_used_blocks * pool_.block_bytes();
  peak_resident_bytes_ = std::max(peak_resident_bytes_, m.paged_resident_bytes);
  m.paged_peak_resident_bytes = peak_resident_bytes_; metrics_.push_back(std::move(m));
}
bool Qwen3PagedRequestScheduler::step() {
  decode_batch_error_pending_ = false;
  expire();
  // Fill an under-sized decode batch before issuing a decode.  This keeps
  // FIFO admission deterministic and is necessary for max_decode_batch_size
  // 2/4; the default size=1 retains the original admission/decode behavior.
  if (!waiting_.empty() &&
      active_.size() < config_.max_decode_batch_size &&
      admit_one()) return true;
  if (decode_batch()) return true;
  if (decode_batch_error_pending_) return false;
  if (admit_one()) return true;
  record({0, waiting_.empty() ? "idle" : "deferred", 0, 0, 0, 0, 0, 0, 0,
          peak_resident_bytes_, !waiting_.empty()});
  return false;
}
std::vector<FinishedRequest> Qwen3PagedRequestScheduler::take_finished() {
  std::vector<FinishedRequest> out;
  while (!finished_.empty()) { out.push_back(std::move(finished_.front())); finished_.pop_front(); }
  return out;
}
bool Qwen3PagedRequestScheduler::has_unfinished() const {
  return !waiting_.empty() || !active_.empty();
}

PagedSchedulerRequestSnapshot Qwen3PagedSchedulerTestAccess::request_snapshot(
    const Qwen3PagedRequestScheduler& scheduler, std::uint64_t request_id) {
  PagedSchedulerRequestSnapshot snapshot;
  const auto it = scheduler.entries_.find(request_id);
  if (it == scheduler.entries_.end()) return snapshot;
  const auto& entry = *it->second;
  snapshot.present = true;
  snapshot.state = entry.state;
  snapshot.generated_ids = entry.generated;
  snapshot.last_token = entry.last_token;
  snapshot.draw_offset = entry.draw_offset;
  snapshot.cache_length = entry.cache ? entry.cache->length() : 0;
  return snapshot;
}

std::vector<std::uint64_t> Qwen3PagedSchedulerTestAccess::active_queue(
    const Qwen3PagedRequestScheduler& scheduler) {
  return {scheduler.active_.begin(), scheduler.active_.end()};
}
}  // namespace llm
