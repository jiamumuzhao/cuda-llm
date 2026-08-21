#pragma once

#include "llm/qwen3_scheduler.h"
#include "llm/qwen3_paged_kv_cache.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

// Independent, non-thread-safe paged scheduler. Each step executes at most
// one request's prefill or one paged decode batch; cancellation applies at
// scheduler batch boundaries and callers must not overlap scheduler calls.
// Supports synchronous paged continuous batching. Prefill and decode batches
// are bounded to CUDA-supported sizes 1/2/4, with optional token budgets and
// fairness quotas. Calls are not thread-safe and cancellation occurs only at
// scheduler batch boundaries.
struct PagedSchedulerConfig {
  std::size_t max_seq_len = 512;
  std::size_t max_active_requests = 0;
  std::size_t max_waiting_requests = 0;
  // Phase 7 admission/decode budgets. Zero keeps the legacy unlimited
  // behavior. Supported CUDA batch sizes remain 1, 2, and 4.
  std::size_t max_prefill_batch_size = 1;
  std::size_t max_prefill_tokens = 0;
  // Prefer same-length packed batches; mixed lengths use the padded fallback.
  bool prefer_packed_prefill = true;
  std::size_t max_decode_tokens = 0;
  std::size_t max_consecutive_prefill = 0;
  std::size_t max_consecutive_decode = 0;
  std::size_t max_decode_batch_size = 1;
  // Test-only deterministic hook invoked immediately before direct paged
  // prefill. Production callers should leave unset.
  std::function<void()> before_seed_for_testing;
  using ClockNow = std::function<std::chrono::steady_clock::time_point()>;
  ClockNow clock_now;
};

struct PagedSchedulerMetrics {
  std::uint64_t step_id = 0;
  std::string action;
  std::uint64_t selected_request_id = 0;
  std::size_t waiting = 0;
  std::size_t active = 0;
  std::size_t finished = 0;
  std::size_t pool_free_blocks = 0;
  std::size_t pool_used_blocks = 0;
  std::size_t paged_resident_bytes = 0;
  std::size_t paged_peak_resident_bytes = 0;
  bool admission_deferred = false;
  std::size_t decode_batch_size = 0;
  std::size_t prefill_batch_size = 0;
  std::size_t valid_prefill_tokens = 0;
  std::size_t padded_prefill_tokens = 0;
  double prefill_padding_ratio = 0.0;
  std::size_t decode_tokens = 0;
  std::size_t consecutive_prefill = 0;
  std::size_t consecutive_decode = 0;
  std::vector<std::uint64_t> selected_request_ids;
  bool decode_batch_error = false;
};

// Read-only test observability. This intentionally exposes no scheduler
// mutation and is not a production control surface.
struct PagedSchedulerRequestSnapshot {
  bool present = false;
  RequestState state = RequestState::Finished;
  std::vector<int32_t> generated_ids;
  int32_t last_token = 0;
  std::uint64_t draw_offset = 0;
  std::size_t cache_length = 0;
};

class Qwen3PagedRequestScheduler {
 public:
  using ClockNow = PagedSchedulerConfig::ClockNow;
  Qwen3PagedRequestScheduler(const Qwen3CudaModel&, PagedKvCachePool&,
                             PagedSchedulerConfig = {});
  ~Qwen3PagedRequestScheduler();
  Qwen3PagedRequestScheduler(const Qwen3PagedRequestScheduler&) = delete;
  Qwen3PagedRequestScheduler& operator=(const Qwen3PagedRequestScheduler&) = delete;

  void submit(InferenceRequestConfig request);
  bool step();
  bool cancel_request(std::uint64_t request_id);
  bool has_unfinished() const;
  std::size_t waiting_count() const noexcept { return waiting_.size(); }
  std::size_t active_count() const noexcept { return active_.size(); }
  std::vector<FinishedRequest> take_finished();
  const std::vector<PagedSchedulerMetrics>& metrics() const noexcept { return metrics_; }
  std::string export_metrics_csv() const;

 private:
  friend class Qwen3PagedSchedulerTestAccess;
  struct Entry;
  void validate_request(const InferenceRequestConfig&) const;
  void finish(Entry&, const std::string&);
  void expire();
  bool admit_one();
  bool admit_batch();
  bool decode_batch();
  bool remove_from_queue(std::deque<std::uint64_t>&, std::uint64_t);
  void record(PagedSchedulerMetrics);

  const Qwen3CudaModel& model_;
  PagedKvCachePool& pool_;
  PagedSchedulerConfig config_;
  CudaSampler sampler_;
  std::unordered_map<std::uint64_t, std::unique_ptr<Entry>> entries_;
  std::deque<std::uint64_t> waiting_, active_;
  std::deque<FinishedRequest> finished_;
  std::vector<PagedSchedulerMetrics> metrics_;
  std::chrono::steady_clock::time_point started_at_;
  std::uint64_t next_step_id_ = 1;
  std::size_t peak_resident_bytes_ = 0;
  bool decode_batch_error_pending_ = false;
  std::size_t consecutive_prefill_ = 0;
  std::size_t consecutive_decode_ = 0;
};

class Qwen3PagedSchedulerTestAccess {
 public:
  static PagedSchedulerRequestSnapshot request_snapshot(
      const Qwen3PagedRequestScheduler&, std::uint64_t request_id);
  static std::vector<std::uint64_t> active_queue(
      const Qwen3PagedRequestScheduler&);

 private:
  Qwen3PagedSchedulerTestAccess() = delete;
};

}  // namespace llm
