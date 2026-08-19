#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_scheduler.h"

#include <cuda_runtime.h>

#include <functional>
#include <iostream>
#include <map>
#include <new>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

using namespace llm;

namespace llm {
class Qwen3RequestSchedulerTestAccess {
 public:
  static void fail_prefill_prepare_after(Qwen3RequestScheduler& scheduler, size_t prepared) {
    scheduler.prefill_prepare_failure_after_for_testing_ = prepared;
  }
};
}  // namespace llm

static void fail(const std::string& message) {
  throw std::runtime_error("test_qwen3_scheduler_continuous: " + message);
}

static void expect_throw(const std::function<void()>& fn) {
  try { fn(); } catch (const std::invalid_argument&) { return; }
  fail("invalid budget was accepted");
}

static InferenceRequestConfig request(uint64_t id, size_t prompt_length,
                                      size_t max_new_tokens, size_t max_seq_len = 16) {
  InferenceRequestConfig r;
  r.request_id = id;
  r.max_new_tokens = max_new_tokens;
  r.max_seq_len = max_seq_len;
  for (size_t i = 0; i < prompt_length; ++i)
    r.prompt_ids.push_back(static_cast<int32_t>(1 + ((id * 101 + i * 17) % 150000)));
  r.sampling.temperature = .8f;
  r.sampling.top_k = 50;
  r.sampling.top_p = .9f;
  r.sampling.seed = 20260806 + id;
  return r;
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

static std::map<uint64_t, FinishedRequest> collect(Qwen3RequestScheduler& scheduler) {
  std::map<uint64_t, FinishedRequest> result;
  for (auto& item : scheduler.take_finished()) result.emplace(item.request_id, std::move(item));
  return result;
}

static size_t csv_field_count(const std::string& line) {
  size_t fields = 1;
  for (const char c : line) if (c == ',') ++fields;
  return fields;
}

static std::map<uint64_t, FinishedRequest> run_continuous(
    Qwen3CudaModel& model, const std::vector<InferenceRequestConfig>& requests,
    StaticBatchBudget b) {
  Qwen3RequestScheduler scheduler(model);
  for (const auto& r : requests) scheduler.submit(r);
  while (scheduler.step_continuous(b)) {}
  if (scheduler.has_unfinished()) fail("continuous scheduler stopped with executable work");
  return collect(scheduler);
}

int main() {
  try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability="
              << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    const StaticBatchBudget b = budget();

    Qwen3RequestScheduler prefill_rollback(model);
    prefill_rollback.submit(request(90, 4, 3));
    prefill_rollback.submit(request(91, 6, 3));
    Qwen3RequestSchedulerTestAccess::fail_prefill_prepare_after(prefill_rollback, 1);
    try {
      prefill_rollback.step_continuous(b);
      fail("injected prefill prepare failure was not propagated");
    } catch (const std::bad_alloc&) {
    }
    if (prefill_rollback.waiting_count() != 2 || prefill_rollback.decode_count() != 0 ||
        prefill_rollback.state(90) != RequestState::Waiting ||
        prefill_rollback.state(91) != RequestState::Waiting ||
        !prefill_rollback.take_finished().empty())
      fail("prefill prepare failure did not restore waiting FIFO state");
    if (!prefill_rollback.step_continuous(b) || prefill_rollback.waiting_count() != 0 ||
        prefill_rollback.decode_count() != 2)
      fail("prefill did not retry cleanly after prepare rollback");

    Qwen3RequestScheduler interleaved(model);
    interleaved.submit(request(1, 4, 3));
    if (!interleaved.step_continuous(b) || !interleaved.step_continuous(b))
      fail("A prefill/decode did not execute");
    interleaved.submit(request(2, 7, 3));
    if (!interleaved.step_continuous(b) || !interleaved.step_continuous(b))
      fail("B prefill or joint decode did not execute");
    const auto& sequence = interleaved.batch_metrics();
    if (sequence.size() != 4 || sequence[0].kind != SchedulerBatchKind::Prefill ||
        sequence[1].kind != SchedulerBatchKind::Decode ||
        sequence[2].kind != SchedulerBatchKind::Prefill ||
        sequence[3].kind != SchedulerBatchKind::Decode ||
        sequence[0].selected_request_ids != std::vector<uint64_t>{1} ||
        sequence[2].selected_request_ids != std::vector<uint64_t>{2} ||
        sequence[3].selected_request_ids != std::vector<uint64_t>({1, 2}) ||
        sequence[2].padded_tokens != 7 || sequence[3].decode_tokens != 2)
      fail("A prefill -> A decode -> B prefill -> A/B decode metrics mismatch");
    while (interleaved.step_continuous(b)) {}
    if (collect(interleaved).size() != 2) fail("interleaved result count mismatch");
    for (const auto& metrics : interleaved.batch_metrics()) {
      if (metrics.batch_size != metrics.selected_request_ids.size() ||
          metrics.active_after > metrics.waiting_before + metrics.decode_before ||
          (metrics.kind == SchedulerBatchKind::Prefill &&
           (metrics.valid_tokens > metrics.padded_tokens ||
            metrics.padding_tokens != metrics.padded_tokens - metrics.valid_tokens)))
        fail("batch metrics invariant mismatch");
    }
    for (const auto& metrics : interleaved.request_metrics()) {
      if (!metrics.first_token_time_ms || !metrics.ttft_ms ||
          *metrics.first_token_time_ms < metrics.submit_time_ms ||
          !metrics.completion_time_ms || metrics.generated_token_count == 0)
        fail("request timing/completion metrics mismatch");
    }
    const std::string batch_csv = interleaved.export_batch_metrics_csv();
    std::istringstream batch_lines(batch_csv);
    std::string line;
    size_t line_number = 0;
    size_t batch_csv_fields = 0;
    while (std::getline(batch_lines, line)) {
      if (line.empty()) continue;
      if (line_number++ == 0) batch_csv_fields = csv_field_count(line);
      else if (csv_field_count(line) != batch_csv_fields) fail("batch CSV field count mismatch");
    }
    const std::string request_csv = interleaved.export_request_metrics_csv();
    std::istringstream request_lines(request_csv);
    line_number = 0;
    size_t request_csv_fields = 0;
    while (std::getline(request_lines, line)) {
      if (line.empty()) continue;
      if (line_number++ == 0) request_csv_fields = csv_field_count(line);
      else if (csv_field_count(line) != request_csv_fields) fail("request CSV field count mismatch");
    }

    Qwen3RequestScheduler decode_limit(model);
    decode_limit.submit(request(3, 4, 5));
    if (!decode_limit.step_continuous(b) || !decode_limit.step_continuous(b))
      fail("decode limit setup prefill/decode failed");
    decode_limit.submit(request(4, 6, 5));
    ContinuousSchedulerConfig decode_one;
    decode_one.max_consecutive_decode = 1;
    if (!decode_limit.step_continuous(b, decode_one) ||
        decode_limit.batch_metrics().back().kind != SchedulerBatchKind::Prefill ||
        decode_limit.batch_metrics().back().consecutive_prefill != 1)
      fail("max_consecutive_decode=1 did not force prefill");

    Qwen3RequestScheduler prefill_limit(model);
    prefill_limit.submit(request(5, 4, 5));
    ContinuousSchedulerConfig prefill_one;
    prefill_one.max_consecutive_prefill = 1;
    if (!prefill_limit.step_continuous(b, prefill_one))
      fail("prefill limit setup prefill failed");
    prefill_limit.submit(request(6, 6, 5));
    if (!prefill_limit.step_continuous(b, prefill_one) ||
        prefill_limit.batch_metrics().back().kind != SchedulerBatchKind::Decode)
      fail("max_consecutive_prefill=1 did not force decode");

    Qwen3RequestScheduler active_limit(model);
    active_limit.submit(request(12, 4, 2));
    active_limit.submit(request(13, 6, 2));
    ContinuousSchedulerConfig one_active;
    one_active.max_active_requests = 1;
    if (!active_limit.step_continuous(b, one_active) || active_limit.decode_count() != 1 ||
        active_limit.waiting_count() != 1 || !active_limit.step_continuous(b, one_active) ||
        !active_limit.step_continuous(b, one_active) || active_limit.waiting_count() != 0)
      fail("max_active_requests did not defer and later admit FIFO waiting work");

    Qwen3RequestScheduler fairness(model);
    fairness.submit(request(5, 4, 5));
    if (!fairness.step_continuous(b)) fail("fairness setup prefill failed");
    fairness.submit(request(6, 6, 5));
    if (!fairness.step_continuous(b)) fail("second consecutive prefill failed");
    fairness.submit(request(7, 5, 5));
    if (!fairness.step_continuous(b) ||
        fairness.batch_metrics().back().kind != SchedulerBatchKind::Decode)
      fail("two consecutive prefills did not force decode when waiting work exists");
    while (fairness.step_continuous(b)) {}

    Qwen3RequestScheduler decode_fairness(model);
    decode_fairness.submit(request(8, 4, 5));
    if (!decode_fairness.step_continuous(b) || !decode_fairness.step_continuous(b) ||
        !decode_fairness.step_continuous(b))
      fail("decode fairness setup did not execute prefill plus two decodes");
    decode_fairness.submit(request(9, 6, 5));
    if (!decode_fairness.step_continuous(b) ||
        decode_fairness.batch_metrics().back().kind != SchedulerBatchKind::Prefill)
      fail("two consecutive decodes did not force prefill when waiting work exists");
    while (decode_fairness.step_continuous(b)) {}

    Qwen3RequestScheduler independent_finish(model);
    independent_finish.submit(request(10, 4, 2));
    independent_finish.submit(request(11, 6, 3));
    while (independent_finish.step_continuous(b)) {}
    const auto independently_finished = collect(independent_finish);
    if (independently_finished.at(10).generated_ids.size() != 2 ||
        independently_finished.at(11).generated_ids.size() != 3 ||
        independently_finished.at(10).stop_reason != "max_new_tokens" ||
        independently_finished.at(11).stop_reason != "max_new_tokens")
      fail("finished request did not leave decode while later request continued");

    const std::vector<InferenceRequestConfig> requests{
        request(20, 4, 3), request(21, 7, 3), request(22, 5, 3)};
    StaticBatchBudget one = b;
    one.max_batch_size = 1;
    one.max_prefill_tokens = 16;
    one.max_decode_tokens = 1;
    const auto serial = run_continuous(model, requests, one);
    const auto batched = run_continuous(model, requests, b);
    for (const auto& r : requests) {
      const auto& serial_result = serial.at(r.request_id);
      const auto& batched_result = batched.at(r.request_id);
      if (serial_result.generated_ids != batched_result.generated_ids ||
          serial_result.stop_reason != batched_result.stop_reason ||
          serial_result.final_cache_length != batched_result.final_cache_length)
        fail("continuous batching changed request-level deterministic result");
    }
    const auto reordered = run_continuous(model, {requests[2], requests[0], requests[1]}, b);
    for (const auto& r : requests)
      if (batched.at(r.request_id).generated_ids != reordered.at(r.request_id).generated_ids)
        fail("submission order changed sampled request result");

    const int32_t eos = model.generate_greedy(request(30, 4, 1).prompt_ids, 1,
                                               std::nullopt, 16).generated_ids.front();
    Qwen3RequestScheduler stops(model);
    auto eos_request = request(30, 4, 8);
    eos_request.sampling.temperature = 0.0f;
    eos_request.eos_token_id = eos;
    stops.submit(eos_request);
    stops.submit(request(31, 5, 1));
    stops.submit(request(32, 4, 3, 4));
    while (stops.step_continuous(b)) {}
    const auto stopped = collect(stops);
    if (stopped.at(30).stop_reason != "eos" || stopped.at(30).generated_ids.size() != 1 ||
        stopped.at(31).stop_reason != "max_new_tokens" ||
        stopped.at(32).stop_reason != "cache_capacity")
      fail("EOS/max_new_tokens/cache capacity stop semantics mismatch");

    Qwen3RequestScheduler blocked(model);
    blocked.submit(request(40, 4, 3));
    StaticBatchBudget context_four = b;
    context_four.max_context_len = 4;
    if (!blocked.step_continuous(context_four)) fail("blocked decode setup prefill failed");
    blocked.submit(request(41, 5, 3));
    if (blocked.step_continuous(context_four) || blocked.waiting_count() != 1 ||
        blocked.decode_count() != 1 || blocked.batch_metrics().back().batch_size == 0)
      fail("both budget-blocked queues did not return false without an empty batch metric");
    const size_t waiting_before = blocked.waiting_count();
    StaticBatchBudget invalid_budget = context_four;
    invalid_budget.max_batch_size = 3;
    expect_throw([&] { blocked.step_continuous(invalid_budget); });
    if (blocked.waiting_count() != waiting_before || blocked.decode_count() != 1 ||
        blocked.state(40) != RequestState::Decode || blocked.state(41) != RequestState::Waiting)
      fail("invalid budget changed continuous queue state");

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_scheduler_continuous passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
