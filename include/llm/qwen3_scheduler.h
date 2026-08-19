#pragma once

#include "ops_cuda.h"
#include "qwen3_padded_prefill.h"
#include <cstdint>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

class Qwen3CudaModel;
class Qwen3KvCache;

enum class RequestState { Waiting, Prefill, Decode, Finished };

struct InferenceRequestConfig {
  uint64_t request_id = 0;
  std::vector<int32_t> prompt_ids;
  size_t max_new_tokens = 0;
  size_t max_seq_len = 0;
  std::optional<int32_t> eos_token_id;
  SamplingConfig sampling;
  size_t queue_timeout_ms = 0;
  size_t request_timeout_ms = 0;
};

struct FinishedRequest {
  uint64_t request_id = 0;
  std::vector<int32_t> generated_ids;
  std::string stop_reason;
  size_t final_cache_length = 0;
};

struct StaticBatchBudget {
  size_t max_batch_size = 1;
  size_t max_prefill_tokens = 1;
  size_t max_decode_tokens = 1;
  size_t max_context_len = 1;
  double max_padding_ratio = 0.0;
};

struct ContinuousSchedulerConfig {
  size_t max_consecutive_prefill = 2;
  size_t max_consecutive_decode = 2;
  // Zero means unlimited. The limit applies to requests admitted into Decode.
  size_t max_active_requests = 0;
  size_t max_waiting_requests = 0;
  size_t max_total_kv_cache_bytes = 0;
};

enum class SchedulerBatchKind { Prefill, Decode };

struct SchedulerBatchMetrics {
  uint64_t round_id = 0;
  SchedulerBatchKind kind = SchedulerBatchKind::Prefill;
  std::vector<uint64_t> selected_request_ids;
  size_t batch_size = 0;
  size_t waiting_before = 0;
  size_t decode_before = 0;
  size_t active_after = 0;
  size_t valid_tokens = 0;
  size_t padded_tokens = 0;
  size_t padding_tokens = 0;
  double padding_ratio = 0.0;
  size_t decode_tokens = 0;
  size_t skipped_budget_count = 0;
  double elapsed_ms = 0.0;
  size_t consecutive_prefill = 0;
  size_t consecutive_decode = 0;
  size_t finished_count = 0;
  size_t cancelled_count = 0;
  size_t timed_out_count = 0;
  size_t resource_rejected_count = 0;
  size_t kv_resident_bytes = 0;
  size_t kv_peak_resident_bytes = 0;
};

struct SchedulerRequestMetrics {
  uint64_t request_id = 0;
  double submit_time_ms = 0.0;
  std::optional<double> admission_time_ms;
  std::optional<double> first_token_time_ms;
  std::optional<double> ttft_ms;
  size_t generated_token_count = 0;
  std::optional<double> completion_time_ms;
  std::string stop_reason;
  size_t queue_timeout_ms = 0;
  size_t request_timeout_ms = 0;
};

class Qwen3RequestScheduler {
 public:
  using ClockNow = std::function<std::chrono::steady_clock::time_point()>;
  explicit Qwen3RequestScheduler(const Qwen3CudaModel& model,
                                 ContinuousSchedulerConfig admission_config = {},
                                 ClockNow clock_now = {});
  ~Qwen3RequestScheduler();
  Qwen3RequestScheduler(const Qwen3RequestScheduler&) = delete;
  Qwen3RequestScheduler& operator=(const Qwen3RequestScheduler&) = delete;

  void submit(InferenceRequestConfig request);
  bool step();
  // Opt-in continuous scheduler: executes at most one budgeted CUDA batch.
  bool step_continuous(const StaticBatchBudget& budget);
  bool step_continuous(const StaticBatchBudget& budget,
                       const ContinuousSchedulerConfig& config);
  // Synchronous, batch-boundary cancellation. The scheduler is not
  // concurrently callable and does not interrupt an already submitted CUDA batch.
  bool cancel_request(uint64_t request_id);
  size_t prefill_waiting_static_batch(size_t max_batch_size);
  size_t prefill_waiting_padded_static_batch(size_t max_batch_size);
  size_t decode_active_static_batch(size_t max_batch_size);
  size_t decode_active_variable_length_batch(size_t max_batch_size);
  size_t prefill_waiting_budgeted_static_batch(const StaticBatchBudget& budget);
  size_t decode_active_budgeted_static_batch(const StaticBatchBudget& budget);
  const std::vector<SchedulerBatchMetrics>& batch_metrics() const { return batch_metrics_; }
  std::string export_batch_metrics_csv() const;
  const std::vector<SchedulerRequestMetrics>& request_metrics() const {
    return request_metrics_;
  }
  std::string export_request_metrics_csv() const;
  void clear_batch_metrics() { batch_metrics_.clear(); }
  std::optional<uint64_t> first_token_round(uint64_t request_id) const;
  std::optional<double> ttft_ms(uint64_t request_id) const;
  bool has_unfinished() const;
  RequestState state(uint64_t request_id) const;
  std::vector<FinishedRequest> take_finished();
  size_t waiting_count() const;
  size_t decode_count() const;
  size_t kv_resident_bytes() const { return kv_resident_bytes_; }
  size_t kv_peak_resident_bytes() const { return kv_peak_resident_bytes_; }

 private:
  friend class Qwen3RequestSchedulerTestAccess;
  struct Entry;
  void finish(Entry& entry, const std::string& reason);
  void run_prefill(Entry& entry);
  void run_decode(Entry& entry);
  void validate_budget(const StaticBatchBudget& budget) const;
  void validate_continuous_config(const ContinuousSchedulerConfig& config) const;
  size_t prefill_batch_size_for_budget(const StaticBatchBudget& budget,
                                       size_t max_total_kv_cache_bytes = 0) const;
  size_t decode_batch_size_for_budget(const StaticBatchBudget& budget) const;
  size_t prefill_waiting_budgeted_static_batch_with_kv_budget(
      const StaticBatchBudget& budget, size_t max_total_kv_cache_bytes);
  size_t kv_bytes_for_request(const InferenceRequestConfig& request) const;
  void allocate_cache(Entry& entry);
  void release_cache(Entry& entry);
  size_t expire_requests();
  void set_batch_observability(SchedulerBatchMetrics& metrics) const;
  std::chrono::steady_clock::time_point now() const { return clock_now_(); }
  void record_first_token(Entry& entry, uint64_t round_id);
  void record_admission(Entry& entry);
  void update_request_metrics(const Entry& entry);
  void append_batch_metrics(SchedulerBatchMetrics metrics,
                            std::chrono::steady_clock::time_point started);

  const Qwen3CudaModel& model_;
  CudaSampler sampler_;
  std::unordered_map<uint64_t, std::unique_ptr<Entry>> entries_;
  std::deque<uint64_t> waiting_;
  std::deque<uint64_t> decode_;
  std::deque<FinishedRequest> finished_;
  std::vector<SchedulerBatchMetrics> batch_metrics_;
  std::vector<SchedulerRequestMetrics> request_metrics_;
  std::unordered_map<uint64_t, size_t> request_metric_indices_;
  std::chrono::steady_clock::time_point scheduler_started_at_;
  uint64_t next_round_id_ = 1;
  size_t continuous_prefill_streak_ = 0;
  size_t continuous_decode_streak_ = 0;
  std::optional<size_t> prefill_prepare_failure_after_for_testing_;
  ContinuousSchedulerConfig admission_config_;
  ClockNow clock_now_;
  size_t kv_resident_bytes_ = 0;
  size_t kv_peak_resident_bytes_ = 0;
  size_t finished_count_ = 0;
  size_t cancelled_count_ = 0;
  size_t timed_out_count_ = 0;
  size_t resource_rejected_count_ = 0;
};

}  // namespace llm
