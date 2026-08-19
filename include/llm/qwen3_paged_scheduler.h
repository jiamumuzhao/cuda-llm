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
// For max_decode_batch_size 2/4, this uses a correctness-first bounded
// batch-fill policy: while waiting is non-empty and active is below the target
// batch size, one single-request admission prefill is executed per step. Once
// the target is full, the next step submits one decode batch. This is not the
// final token-budget fairness policy.
struct PagedSchedulerConfig {
  std::size_t max_seq_len = 32;
  std::size_t max_active_requests = 0;
  std::size_t max_waiting_requests = 0;
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

 private:
  friend class Qwen3PagedSchedulerTestAccess;
  struct Entry;
  void validate_request(const InferenceRequestConfig&) const;
  void finish(Entry&, const std::string&);
  void expire();
  bool admit_one();
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
