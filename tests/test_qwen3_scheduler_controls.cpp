#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_scheduler.h"

#include <cuda_runtime.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
using Clock = std::chrono::steady_clock;

static void fail(const std::string& message) {
  throw std::runtime_error("test_qwen3_scheduler_controls: " + message);
}

static InferenceRequestConfig request(uint64_t id, size_t prompt_length,
                                      size_t max_new_tokens = 3,
                                      size_t max_seq_len = 16) {
  InferenceRequestConfig r;
  r.request_id = id;
  r.max_new_tokens = max_new_tokens;
  r.max_seq_len = max_seq_len;
  for (size_t i = 0; i < prompt_length; ++i)
    r.prompt_ids.push_back(static_cast<int32_t>(1 + ((id * 101 + i * 17) % 150000)));
  return r;
}

static std::map<uint64_t, FinishedRequest> collect(Qwen3RequestScheduler& scheduler) {
  std::map<uint64_t, FinishedRequest> result;
  for (auto& item : scheduler.take_finished()) result.emplace(item.request_id, std::move(item));
  return result;
}

static StaticBatchBudget budget() {
  StaticBatchBudget b;
  b.max_batch_size = 4;
  b.max_prefill_tokens = 32;
  b.max_decode_tokens = 4;
  b.max_context_len = 16;
  b.max_padding_ratio = 1.0;
  return b;
}

int main() {
  try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability="
              << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    const StaticBatchBudget b = budget();

    // Waiting cancellation is immediate and does not allocate a KV cache.
    Qwen3RequestScheduler waiting_cancel(model);
    waiting_cancel.submit(request(1, 4));
    if (!waiting_cancel.cancel_request(1) || waiting_cancel.waiting_count() != 0 ||
        waiting_cancel.kv_resident_bytes() != 0 || waiting_cancel.step_continuous(b))
      fail("waiting cancellation entered CUDA or did not remove the request");
    auto waiting_result = collect(waiting_cancel);
    if (waiting_result.at(1).stop_reason != "cancelled" || waiting_cancel.cancel_request(1) ||
        waiting_cancel.cancel_request(999) || !waiting_cancel.take_finished().empty())
      fail("unknown/repeated cancellation semantics mismatch");
    std::cout << "waiting cancel: no KV allocation/CUDA, reason=cancelled; unknown/repeat=false passed\n";

    // Active cancellation takes effect before the next batch and does not affect its peer.
    Qwen3RequestScheduler active_cancel(model);
    active_cancel.submit(request(2, 4));
    active_cancel.submit(request(3, 4));
    if (!active_cancel.step_continuous(b) || active_cancel.decode_count() != 2)
      fail("active cancellation setup prefill failed");
    const size_t resident_before_cancel = active_cancel.kv_resident_bytes();
    if (!active_cancel.cancel_request(2) || active_cancel.decode_count() != 1 ||
        active_cancel.kv_resident_bytes() >= resident_before_cancel)
      fail("active cancellation did not release its cache at batch boundary");
    auto baseline = model.generate_greedy(request(3, 4).prompt_ids, 3, std::nullopt, 16);
    while (active_cancel.step_continuous(b)) {}
    auto active_results = collect(active_cancel);
    if (active_results.at(2).stop_reason != "cancelled" ||
        active_results.at(3).generated_ids != baseline.generated_ids ||
        active_results.at(3).stop_reason != baseline.stop_reason ||
        active_results.at(3).final_cache_length != baseline.final_cache_length ||
        active_cancel.kv_resident_bytes() != 0)
      fail("active cancellation changed peer output or leaked KV cache");
    std::cout << "active cancel: batch-boundary release and peer baseline parity passed\n";

    // Injected clock makes queue/request timeout tests deterministic without sleep.
    auto current = std::make_shared<Clock::time_point>(Clock::time_point{});
    auto clock_now = [current] { return *current; };
    Qwen3RequestScheduler queue_timeout(model, {}, clock_now);
    auto queue_request = request(10, 4);
    queue_request.queue_timeout_ms = 100;
    queue_timeout.submit(queue_request);
    *current += std::chrono::milliseconds(101);
    if (queue_timeout.step_continuous(b) || queue_timeout.waiting_count() != 0 ||
        queue_timeout.kv_resident_bytes() != 0)
      fail("queue timeout admitted work or allocated KV");
    if (collect(queue_timeout).at(10).stop_reason != "queue_timeout")
      fail("queue timeout reason mismatch");

    Qwen3RequestScheduler request_timeout(model, {}, clock_now);
    auto lifecycle_request = request(11, 4);
    lifecycle_request.request_timeout_ms = 100;
    request_timeout.submit(lifecycle_request);
    if (!request_timeout.step_continuous(b) || request_timeout.decode_count() != 1)
      fail("request timeout setup prefill failed");
    *current += std::chrono::milliseconds(101);
    if (request_timeout.step_continuous(b) || request_timeout.kv_resident_bytes() != 0 ||
        collect(request_timeout).at(11).stop_reason != "request_timeout")
      fail("request timeout did not release active KV");

    Qwen3RequestScheduler priority(model, {}, clock_now);
    auto priority_request = request(12, 4);
    priority_request.queue_timeout_ms = 100;
    priority_request.request_timeout_ms = 100;
    priority.submit(priority_request);
    *current += std::chrono::milliseconds(101);
    if (!priority.cancel_request(12) || priority.step_continuous(b) ||
        collect(priority).at(12).stop_reason != "cancelled")
      fail("cancel did not take precedence over timeout");
    std::cout << "queue/request timeout, deterministic clock, cancel priority and KV release passed\n";

    // Waiting admission cap rejects without constructing a cache.
    ContinuousSchedulerConfig waiting_cap;
    waiting_cap.max_waiting_requests = 1;
    Qwen3RequestScheduler capped(model, waiting_cap);
    capped.submit(request(20, 4));
    capped.submit(request(21, 4));
    if (capped.waiting_count() != 1 || capped.kv_resident_bytes() != 0)
      fail("max_waiting_requests did not reject before KV allocation");
    auto capped_results = collect(capped);
    if (capped_results.at(21).stop_reason != "resource_exhausted")
      fail("max_waiting_requests reason mismatch");
    std::cout << "max_waiting_requests rejection reason=resource_exhausted no-KV passed\n";

    // One-cache KV budget defers the second FIFO request, then recovers after completion.
    Qwen3RequestScheduler kv_budget_scheduler(model);
    auto first = request(30, 4, 1);
    auto second = request(31, 4, 1);
    kv_budget_scheduler.submit(first);
    kv_budget_scheduler.submit(second);
    const size_t one_cache_bytes = 28 * 2 * 8 * 128 * 2 * first.max_seq_len;
    StaticBatchBudget kv_budget = b;
    ContinuousSchedulerConfig kv_config;
    kv_config.max_total_kv_cache_bytes = one_cache_bytes;
    if (!kv_budget_scheduler.step_continuous(kv_budget, kv_config) ||
        kv_budget_scheduler.waiting_count() != 1)
      fail("KV budget did not admit only the first FIFO request");
    if (!kv_budget_scheduler.step_continuous(kv_budget, kv_config) ||
        kv_budget_scheduler.waiting_count() != 0)
      fail("KV budget was not returned after request completion");
    if (kv_budget_scheduler.kv_peak_resident_bytes() != one_cache_bytes)
      fail("KV resident/peak byte calculation mismatch");
    auto kv_results = collect(kv_budget_scheduler);
    if (kv_results.size() != 2 || kv_results.at(30).stop_reason != "max_new_tokens" ||
        kv_results.at(31).stop_reason != "max_new_tokens" ||
        kv_budget_scheduler.kv_resident_bytes() != 0)
      fail("KV budget completion/reclamation mismatch");
    std::cout << "max_total_kv_cache_bytes exact resident calculation, defer and reclaim passed\n";

    const auto& metrics = kv_budget_scheduler.batch_metrics();
    if (metrics.empty() || metrics.back().kv_resident_bytes != 0 ||
        metrics.back().kv_peak_resident_bytes != one_cache_bytes)
      fail("batch KV observability mismatch");
    const std::string request_csv = kv_budget_scheduler.export_request_metrics_csv();
    if (request_csv.find("stop_reason,queue_timeout_ms,request_timeout_ms") == std::string::npos ||
        request_csv.find("max_new_tokens") == std::string::npos)
      fail("request CSV missing termination/timeout fields");
    std::cout << "batch/request metrics CSV termination, timeout and KV fields passed\n";

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_scheduler_controls passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
