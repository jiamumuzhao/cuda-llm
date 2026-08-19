#include "llm/cuda_check.h"
#include "llm/qwen3_layer_cuda.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_paged_scheduler.h"

#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace llm;
static void fail(const std::string& s) { throw std::runtime_error("test_qwen3_paged_scheduler: " + s); }
static void expect(bool v, const std::string& s) { if (!v) fail(s); }
static InferenceRequestConfig request(uint64_t id, size_t max_new = 2) {
  InferenceRequestConfig r;
  r.request_id = id; r.prompt_ids = {11, 22, 33, 44};
  r.max_new_tokens = max_new; r.max_seq_len = 16;
  r.sampling.temperature = 0.0f;
  return r;
}
static InferenceRequestConfig request_with_prompt(
    uint64_t id, std::vector<int32_t> prompt, size_t max_new,
    size_t max_seq_len = 32) {
  InferenceRequestConfig r;
  r.request_id = id;
  r.prompt_ids = std::move(prompt);
  r.max_new_tokens = max_new;
  r.max_seq_len = max_seq_len;
  r.sampling.temperature = 0.0f;
  return r;
}
static void check_result(const FinishedRequest& actual,
                         const GreedyGenerationResult& expected, uint64_t id) {
  expect(actual.request_id == id, "request id mismatch");
  expect(actual.generated_ids == expected.generated_ids, "generated ids mismatch");
  expect(actual.stop_reason == expected.stop_reason, "stop reason mismatch");
  expect(actual.final_cache_length == expected.final_cache_length, "cache length mismatch");
}
static const PagedSchedulerMetrics* find_decode_metric(
    const Qwen3PagedRequestScheduler& scheduler, std::size_t batch_size) {
  for (auto it = scheduler.metrics().rbegin(); it != scheduler.metrics().rend(); ++it)
    if (it->action == "decode" && it->decode_batch_size == batch_size) return &*it;
  return nullptr;
}
static bool has_decode_batch_metric(
    const Qwen3PagedRequestScheduler& scheduler, std::size_t batch_size) {
  return find_decode_metric(scheduler, batch_size) != nullptr;
}
static void expect_snapshot_equal(const PagedSchedulerRequestSnapshot& a,
                                  const PagedSchedulerRequestSnapshot& b,
                                  const std::string& label) {
  expect(a.present == b.present, label + " presence changed");
  expect(a.state == b.state, label + " state changed");
  expect(a.generated_ids == b.generated_ids, label + " generated IDs changed");
  expect(a.last_token == b.last_token, label + " last token changed");
  expect(a.draw_offset == b.draw_offset, label + " draw offset changed");
  expect(a.cache_length == b.cache_length, label + " cache length changed");
}
int main() {
  int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
  if (!devices) { std::cout << "SKIP test_qwen3_paged_scheduler: no CUDA device\n"; return 0; }
  try {
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    PagedKvCachePool pool({2, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig cfg; cfg.max_seq_len = 16; cfg.max_active_requests = 1;
    Qwen3PagedRequestScheduler scheduler(model, pool, cfg);
    const auto expected = model.generate_greedy({11,22,33,44}, 2, std::nullopt, 16);
    scheduler.submit(request(1, 2)); scheduler.submit(request(2, 2));
    expect(scheduler.waiting_count() == 2, "requests not FIFO waiting");
    expect(scheduler.step(), "first admission did not execute");
    expect(scheduler.active_count() == 1 && scheduler.waiting_count() == 1,
           "max_active admission mismatch");
    expect(pool.used_block_count() == 1, "first request block allocation mismatch");
    while (scheduler.has_unfinished()) scheduler.step();
    auto finished = scheduler.take_finished();
    expect(finished.size() == 2, "two requests did not finish");
    check_result(finished[0], expected, 1); check_result(finished[1], expected, 2);
    expect(pool.used_block_count() == 0 && pool.free_block_count() == 2,
           "completed requests did not reclaim blocks");
    bool saw_defer = false;
    for (const auto& m : scheduler.metrics()) saw_defer |= m.admission_deferred;
    std::cout << "FIFO/max_active and paged result alignment passed; deferred="
              << (saw_defer ? "true" : "false") << "\n";

    PagedKvCachePool budget_pool({1, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig budget_cfg; budget_cfg.max_seq_len = 16;
    Qwen3PagedRequestScheduler budget_scheduler(model, budget_pool, budget_cfg);
    budget_scheduler.submit(request(3, 2)); budget_scheduler.submit(request(4, 1));
    expect(budget_scheduler.step(), "budget first admission failed");
    expect(budget_scheduler.waiting_count() == 1 && budget_pool.used_block_count() == 1,
           "budget defer did not preserve FIFO");
    expect(budget_scheduler.step(), "budget second step failed");
    expect(budget_scheduler.waiting_count() == 1, "budget request did not remain waiting");
    while (budget_scheduler.has_unfinished()) budget_scheduler.step();
    auto budget_finished = budget_scheduler.take_finished();
    expect(budget_finished.size() == 2 && budget_pool.used_block_count() == 0,
           "budget block reclaim/admission failed");
    std::cout << "conservative block budget defer and reclaim passed\n";

    PagedKvCachePool reject_pool({2, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig reject_cfg; reject_cfg.max_seq_len = 16;
    reject_cfg.max_waiting_requests = 1;
    Qwen3PagedRequestScheduler reject_scheduler(model, reject_pool, reject_cfg);
    reject_scheduler.submit(request(5, 1)); reject_scheduler.submit(request(6, 1));
    expect(reject_pool.used_block_count() == 0, "waiting rejection allocated blocks");
    auto rejected = reject_scheduler.take_finished();
    expect(rejected.size() == 1 && rejected[0].request_id == 6 &&
           rejected[0].stop_reason == "resource_exhausted", "waiting rejection mismatch");
    expect(reject_scheduler.cancel_request(5), "waiting cancel failed");
    expect(!reject_scheduler.cancel_request(5) && !reject_scheduler.cancel_request(99),
           "repeated/unknown cancel mismatch");
    expect(reject_pool.used_block_count() == 0, "waiting cancellation allocated blocks");
    std::cout << "max_waiting resource rejection and waiting cancellation passed\n";

    auto now = std::chrono::steady_clock::time_point{};
    PagedKvCachePool timeout_pool({2, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig timeout_cfg; timeout_cfg.max_seq_len = 16;
    timeout_cfg.clock_now = [&now] { return now; };
    Qwen3PagedRequestScheduler timeout_scheduler(model, timeout_pool, timeout_cfg);
    auto timed = request(7, 1); timed.queue_timeout_ms = 10;
    timeout_scheduler.submit(timed); now += std::chrono::milliseconds(11);
    expect(!timeout_scheduler.step(), "queue timeout step unexpectedly ran");
    auto timeout_result = timeout_scheduler.take_finished();
    expect(timeout_result.size() == 1 && timeout_result[0].stop_reason == "queue_timeout",
           "queue timeout mismatch");
    expect(timeout_pool.used_block_count() == 0, "queue timeout allocated blocks");
    std::cout << "fake-clock queue timeout passed\n";

    PagedKvCachePool active_cancel_pool({2, 28, 8, 16, 128, DType::F16});
    Qwen3PagedRequestScheduler active_cancel(model, active_cancel_pool, timeout_cfg);
    active_cancel.submit(request(8, 3)); expect(active_cancel.step(), "active cancel admission failed");
    expect(active_cancel.active_count() == 1, "active cancel request not active");
    expect(active_cancel.cancel_request(8), "active cancel failed");
    expect(active_cancel.active_count() == 0 && active_cancel_pool.used_block_count() == 0,
           "active cancel did not reclaim blocks");
    auto cancelled = active_cancel.take_finished();
    expect(cancelled.size() == 1 && cancelled[0].stop_reason == "cancelled",
           "active cancel mismatch");
    std::cout << "active cancellation and block reclaim passed\n";

    // Deterministically consume the only free block after admission accounting
    // but before seed, proving that seed exhaustion restores the FIFO head.
    PagedKvCachePool injected_pool({1, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig injected_cfg; injected_cfg.max_seq_len = 16;
    BlockId injected_block = kInvalidBlockId;
    bool inject_once = true;
    injected_cfg.before_seed_for_testing = [&] {
      if (inject_once) {
        injected_block = injected_pool.block_manager().allocate();
        inject_once = false;
      }
    };
    Qwen3PagedRequestScheduler injected_scheduler(model, injected_pool, injected_cfg);
    injected_scheduler.submit(request(9, 1));
    const bool injected_step = injected_scheduler.step();
    expect(injected_step == false, "seed exhaustion was not deferred");
    expect(injected_scheduler.waiting_count() == 1 && injected_scheduler.active_count() == 0 &&
           injected_scheduler.has_unfinished(), "seed exhaustion did not restore FIFO waiting");
    expect(injected_pool.used_block_count() == 1 && injected_pool.free_block_count() == 0,
           "seed exhaustion changed pool unexpectedly");
    bool saw_seed_defer = false;
    for (const auto& m : injected_scheduler.metrics())
      saw_seed_defer |= m.action == "deferred" && m.selected_request_id == 9 &&
                        m.admission_deferred;
    expect(saw_seed_defer, "seed exhaustion deferred metric missing");
    injected_pool.block_manager().release(injected_block);
    expect(injected_scheduler.step(), "admission did not recover after injected failure");
    while (injected_scheduler.has_unfinished()) injected_scheduler.step();
    expect(injected_scheduler.take_finished().size() == 1 &&
           injected_pool.used_block_count() == 0, "seed recovery leaked block");
    std::cout << "post-accounting seed exhaustion FIFO rollback/recovery passed\n";

    // Dynamic admission is serialized at the B=1 scheduler boundary; B cannot
    // affect A while A is active, and both results match independent runs.
    PagedKvCachePool dynamic_pool({2, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig dynamic_cfg; dynamic_cfg.max_seq_len = 16;
    Qwen3PagedRequestScheduler dynamic_scheduler(model, dynamic_pool, dynamic_cfg);
    const auto expected_a = model.generate_greedy({11,22,33,44}, 3, std::nullopt, 16);
    const auto expected_b = model.generate_greedy({11,22,33,44}, 2, std::nullopt, 16);
    auto request_a = request(10, 3); auto request_b = request(11, 2);
    dynamic_scheduler.submit(request_a);
    expect(dynamic_scheduler.step() && dynamic_scheduler.active_count() == 1,
           "dynamic A did not enter active");
    dynamic_scheduler.submit(request_b);
    expect(dynamic_scheduler.waiting_count() == 1 && dynamic_pool.used_block_count() == 1,
           "dynamic B changed A ownership");
    while (dynamic_scheduler.has_unfinished()) dynamic_scheduler.step();
    auto dynamic_finished = dynamic_scheduler.take_finished();
    expect(dynamic_finished.size() == 2, "dynamic requests did not finish");
    check_result(dynamic_finished[0], expected_a, 10);
    check_result(dynamic_finished[1], expected_b, 11);
    expect(dynamic_pool.used_block_count() == 0, "dynamic admission leaked blocks");
    std::cout << "dynamic join FIFO/result alignment and ownership isolation passed\n";

    // Active request timeout is checked before decode at the next scheduler
    // boundary and immediately releases its paged blocks; B can then admit.
    auto active_now = std::chrono::steady_clock::time_point{};
    PagedKvCachePool active_timeout_pool({2, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig active_timeout_cfg; active_timeout_cfg.max_seq_len = 16;
    active_timeout_cfg.clock_now = [&active_now] { return active_now; };
    Qwen3PagedRequestScheduler active_timeout(model, active_timeout_pool,
                                               active_timeout_cfg);
    auto timeout_active_request = request(12, 3);
    timeout_active_request.request_timeout_ms = 10;
    active_timeout.submit(timeout_active_request);
    expect(active_timeout.step() && active_timeout.active_count() == 1,
           "active timeout request did not admit");
    active_timeout.submit(request(13, 1));
    active_now += std::chrono::milliseconds(11);
    expect(active_timeout.step(), "active timeout boundary did not progress");
    auto active_timeout_finished = active_timeout.take_finished();
    expect(active_timeout_finished.size() == 2,
           "active timeout did not finish timeout and waiting successor");
    expect(active_timeout_finished[0].request_id == 12 &&
           active_timeout_finished[0].stop_reason == "request_timeout" &&
           active_timeout_finished[0].final_cache_length == 4,
           "active timeout result mismatch");
    expect(active_timeout_finished[1].request_id == 13,
           "waiting successor did not admit after active timeout");
    expect(active_timeout_pool.used_block_count() == 0,
           "active timeout did not reclaim all blocks");
    std::cout << "active request timeout boundary/reclaim and successor admission passed\n";

    // Phase 6.4c: two active requests must use one real B=2 paged decode call,
    // while preserving FIFO selection and independent greedy results.
    PagedKvCachePool batch2_pool({8, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig batch2_cfg;
    batch2_cfg.max_seq_len = 32;
    batch2_cfg.max_decode_batch_size = 2;
    Qwen3PagedRequestScheduler batch2_scheduler(model, batch2_pool, batch2_cfg);
    const std::vector<int32_t> batch2_prompt_a = {11, 22, 33, 44, 55, 66, 77, 88};
    const std::vector<int32_t> batch2_prompt_b = {101, 202, 303, 404, 505, 606};
    const auto batch2_expected_a = model.generate_greedy(batch2_prompt_a, 3, std::nullopt, 32);
    const auto batch2_expected_b = model.generate_greedy(batch2_prompt_b, 4, std::nullopt, 32);
    batch2_scheduler.submit(request_with_prompt(201, batch2_prompt_a, 3));
    batch2_scheduler.submit(request_with_prompt(202, batch2_prompt_b, 4));
    expect(batch2_scheduler.step() && batch2_scheduler.step(),
           "B=2 admissions did not execute");
    expect(batch2_scheduler.active_count() == 2 && batch2_scheduler.waiting_count() == 0,
           "B=2 requests did not become active");
    expect(batch2_scheduler.step(), "B=2 decode did not execute");
    const auto* b2_metric = find_decode_metric(batch2_scheduler, 2);
    expect(b2_metric && b2_metric->selected_request_ids == std::vector<uint64_t>{201, 202},
           "B=2 decode metric/FIFO selection mismatch");
    while (batch2_scheduler.has_unfinished()) batch2_scheduler.step();
    const auto batch2_finished = batch2_scheduler.take_finished();
    expect(batch2_finished.size() == 2, "B=2 requests did not finish");
    check_result(batch2_finished[0], batch2_expected_a, 201);
    check_result(batch2_finished[1], batch2_expected_b, 202);
    expect(batch2_pool.used_block_count() == 0, "B=2 decode leaked paged blocks");
    std::cout << "paged decode B=2 FIFO batch/result alignment passed\n";

    // B=4 exercises the supported maximum and the smaller follow-up batches
    // after rows finish at different generated-token limits.  The scheduler
    // deliberately emits B=2 instead of unsupported B=3, then B=1.
    PagedKvCachePool batch4_pool({12, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig batch4_cfg;
    batch4_cfg.max_seq_len = 32;
    batch4_cfg.max_decode_batch_size = 4;
    Qwen3PagedRequestScheduler batch4_scheduler(model, batch4_pool, batch4_cfg);
    const std::vector<std::vector<int32_t>> batch4_prompts = {
        std::vector<int32_t>(15, 17), std::vector<int32_t>(16, 29),
        std::vector<int32_t>(17, 41), {71, 72, 73, 74}};
    const std::vector<std::size_t> batch4_max_new = {2, 3, 4, 5};
    std::vector<GreedyGenerationResult> batch4_expected;
    for (std::size_t i = 0; i < batch4_prompts.size(); ++i) {
      batch4_expected.push_back(model.generate_greedy(
          batch4_prompts[i], batch4_max_new[i], std::nullopt, 32));
      batch4_scheduler.submit(request_with_prompt(
          301 + i, batch4_prompts[i], batch4_max_new[i]));
    }
    for (std::size_t i = 0; i < 4; ++i) {
      expect(batch4_scheduler.step(), "B=4 admission did not execute");
      expect(!batch4_scheduler.metrics().empty() &&
                 batch4_scheduler.metrics().back().action == "paged_prefill" &&
                 batch4_scheduler.metrics().back().decode_batch_size == 0,
             "B=4 batch-fill was not one prefill per step");
    }
    expect(batch4_scheduler.active_count() == 4, "B=4 active count mismatch");
    expect(batch4_scheduler.step(), "B=4 decode did not execute");
    const auto* b4_metric = find_decode_metric(batch4_scheduler, 4);
    expect(b4_metric && b4_metric->selected_request_ids ==
               std::vector<uint64_t>{301, 302, 303, 304},
           "B=4 decode metric/FIFO selection mismatch");
    while (batch4_scheduler.has_unfinished()) batch4_scheduler.step();
    const auto batch4_finished = batch4_scheduler.take_finished();
    expect(batch4_finished.size() == 4, "B=4 requests did not finish");
    for (std::size_t i = 0; i < batch4_finished.size(); ++i)
      check_result(batch4_finished[i], batch4_expected[i], 301 + i);
    expect(has_decode_batch_metric(batch4_scheduler, 2),
           "B=4 follow-up did not exercise B=2");
    expect(has_decode_batch_metric(batch4_scheduler, 1),
           "B=4 follow-up did not exercise B=1");
    expect(batch4_pool.used_block_count() == 0, "B=4 decode leaked paged blocks");
    std::cout << "paged decode B=4/B=2/B=1 FIFO batch/result alignment passed\n";

    // Dynamic B=2 admission: once A/B fill the active batch, C remains
    // waiting through their first decode; after A finishes, one bounded fill
    // admission adds C and the next decode is the legal B=2 pair B/C.
    PagedKvCachePool dynamic2_pool({12, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig dynamic2_cfg;
    dynamic2_cfg.max_seq_len = 32;
    dynamic2_cfg.max_decode_batch_size = 2;
    Qwen3PagedRequestScheduler dynamic2_scheduler(model, dynamic2_pool, dynamic2_cfg);
    const std::vector<int32_t> dynamic2_prompt_c = {801, 802, 803, 804, 805};
    const auto dynamic2_expected_a = model.generate_greedy(batch2_prompt_a, 2, std::nullopt, 32);
    const auto dynamic2_expected_b = model.generate_greedy(batch2_prompt_b, 4, std::nullopt, 32);
    const auto dynamic2_expected_c = model.generate_greedy(dynamic2_prompt_c, 3, std::nullopt, 32);
    dynamic2_scheduler.submit(request_with_prompt(501, batch2_prompt_a, 2));
    dynamic2_scheduler.submit(request_with_prompt(502, batch2_prompt_b, 4));
    expect(dynamic2_scheduler.step() && dynamic2_scheduler.step(),
           "dynamic B=2 initial admissions failed");
    dynamic2_scheduler.submit(request_with_prompt(503, dynamic2_prompt_c, 3));
    expect(dynamic2_scheduler.step(), "dynamic B=2 first decode failed");
    expect(dynamic2_scheduler.waiting_count() == 1 &&
               dynamic2_scheduler.metrics().back().selected_request_ids ==
                   std::vector<uint64_t>{501, 502},
           "dynamic C interfered with A/B FIFO batch");
    expect(dynamic2_scheduler.step(), "dynamic B=2 refill admission failed");
    expect(dynamic2_scheduler.waiting_count() == 0 && dynamic2_scheduler.active_count() == 2,
           "dynamic C was not admitted after A completed");
    expect(dynamic2_scheduler.metrics().back().action == "paged_prefill",
           "dynamic refill was not a single admission prefill");
    expect(dynamic2_scheduler.step(), "dynamic B/C decode failed");
    expect(dynamic2_scheduler.metrics().back().selected_request_ids ==
               std::vector<uint64_t>{502, 503},
           "dynamic B/C FIFO batch selection mismatch");
    while (dynamic2_scheduler.has_unfinished()) dynamic2_scheduler.step();
    const auto dynamic2_finished = dynamic2_scheduler.take_finished();
    expect(dynamic2_finished.size() == 3, "dynamic B=2 requests did not finish");
    check_result(dynamic2_finished[0], dynamic2_expected_a, 501);
    check_result(dynamic2_finished[1], dynamic2_expected_b, 502);
    check_result(dynamic2_finished[2], dynamic2_expected_c, 503);
    expect(dynamic2_pool.used_block_count() == 0, "dynamic B=2 leaked blocks");
    std::cout << "B=2 dynamic join/refill FIFO and result alignment passed\n";

    // Cancellation at a batch boundary removes one row before the next CUDA
    // submission; the survivor must continue through the B=1 batch path.
    PagedKvCachePool cancel2_pool({8, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig cancel2_cfg;
    cancel2_cfg.max_seq_len = 32;
    cancel2_cfg.max_decode_batch_size = 2;
    Qwen3PagedRequestScheduler cancel2_scheduler(model, cancel2_pool, cancel2_cfg);
    const auto cancel2_expected = model.generate_greedy(batch2_prompt_b, 3, std::nullopt, 32);
    cancel2_scheduler.submit(request_with_prompt(601, batch2_prompt_a, 3));
    cancel2_scheduler.submit(request_with_prompt(602, batch2_prompt_b, 3));
    expect(cancel2_scheduler.step() && cancel2_scheduler.step(),
           "B=2 cancellation admissions failed");
    const auto cancel2_before = Qwen3PagedSchedulerTestAccess::request_snapshot(
        cancel2_scheduler, 602);
    expect(cancel2_scheduler.cancel_request(601), "B=2 boundary cancellation failed");
    const auto cancel2_finished = cancel2_scheduler.take_finished();
    expect(cancel2_finished.size() == 1 && cancel2_finished[0].request_id == 601 &&
               cancel2_finished[0].stop_reason == "cancelled",
           "B=2 cancelled result mismatch");
    expect(cancel2_pool.used_block_count() == 1 && cancel2_scheduler.active_count() == 1,
           "B=2 cancellation did not reclaim only cancelled row");
    expect(cancel2_scheduler.step(), "surviving B=1 decode failed");
    expect(cancel2_scheduler.metrics().back().decode_batch_size == 1 &&
               cancel2_scheduler.metrics().back().selected_request_ids ==
                   std::vector<uint64_t>{602},
           "cancelled request remained in decode batch");
    while (cancel2_scheduler.has_unfinished()) cancel2_scheduler.step();
    const auto cancel2_survivor = cancel2_scheduler.take_finished();
    expect(cancel2_survivor.size() == 1, "B=1 survivor did not finish");
    check_result(cancel2_survivor[0], cancel2_expected, 602);
    expect(cancel2_pool.used_block_count() == 0, "B=2 cancellation leaked blocks");
    expect(cancel2_before.state == RequestState::Decode,
           "B=2 cancellation precondition was not Decode");
    std::cout << "B=2 batch-boundary cancellation and B=1 survivor passed\n";

    // Timeout is checked before any decode submission and similarly leaves a
    // legal B=1 batch for the non-expired row.
    auto batch_timeout_now = std::chrono::steady_clock::time_point{};
    PagedKvCachePool timeout2_pool({8, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig timeout2_cfg;
    timeout2_cfg.max_seq_len = 32;
    timeout2_cfg.max_decode_batch_size = 2;
    timeout2_cfg.clock_now = [&batch_timeout_now] { return batch_timeout_now; };
    Qwen3PagedRequestScheduler timeout2_scheduler(model, timeout2_pool, timeout2_cfg);
    const auto timeout2_expected = model.generate_greedy(batch2_prompt_b, 3, std::nullopt, 32);
    auto expiring = request_with_prompt(701, batch2_prompt_a, 3);
    expiring.request_timeout_ms = 10;
    timeout2_scheduler.submit(expiring);
    timeout2_scheduler.submit(request_with_prompt(702, batch2_prompt_b, 3));
    expect(timeout2_scheduler.step() && timeout2_scheduler.step(),
           "B=2 timeout admissions failed");
    batch_timeout_now += std::chrono::milliseconds(11);
    expect(timeout2_scheduler.step(), "B=2 timeout survivor did not decode");
    const auto timeout2_finished = timeout2_scheduler.take_finished();
    expect(timeout2_finished.size() == 1 && timeout2_finished[0].request_id == 701 &&
               timeout2_finished[0].stop_reason == "request_timeout",
           "B=2 timeout reason mismatch");
    expect(timeout2_scheduler.metrics().back().decode_batch_size == 1 &&
               timeout2_scheduler.metrics().back().selected_request_ids ==
                   std::vector<uint64_t>{702},
           "B=2 timeout request entered decode batch");
    while (timeout2_scheduler.has_unfinished()) timeout2_scheduler.step();
    const auto timeout2_survivor = timeout2_scheduler.take_finished();
    expect(timeout2_survivor.size() == 1, "B=2 timeout survivor did not finish");
    check_result(timeout2_survivor[0], timeout2_expected, 702);
    expect(timeout2_pool.used_block_count() == 0, "B=2 timeout leaked blocks");
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "B=2 batch-boundary timeout/reclaim and B=1 survivor passed\n";

    // A model-side batch failure must leave every selected request active with
    // its token/cache state untouched; clearing the deterministic hook allows
    // the same batch to retry successfully.
    PagedKvCachePool batch_error_pool({8, 28, 8, 16, 128, DType::F16});
    PagedSchedulerConfig batch_error_cfg;
    batch_error_cfg.max_seq_len = 32;
    batch_error_cfg.max_decode_batch_size = 2;
    Qwen3PagedRequestScheduler batch_error_scheduler(model, batch_error_pool,
                                                       batch_error_cfg);
    const auto batch_error_expected_401 =
        model.generate_greedy(batch2_prompt_a, 3, std::nullopt, 32);
    const auto batch_error_expected_402 =
        model.generate_greedy(batch2_prompt_b, 3, std::nullopt, 32);
    batch_error_scheduler.submit(request_with_prompt(401, batch2_prompt_a, 3));
    batch_error_scheduler.submit(request_with_prompt(402, batch2_prompt_b, 3));
    expect(batch_error_scheduler.step() && batch_error_scheduler.step(),
           "batch-error admissions did not execute");
    expect(batch_error_scheduler.active_count() == 2, "batch-error active count mismatch");
    const auto batch_error_before_401 = Qwen3PagedSchedulerTestAccess::request_snapshot(
        batch_error_scheduler, 401);
    const auto batch_error_before_402 = Qwen3PagedSchedulerTestAccess::request_snapshot(
        batch_error_scheduler, 402);
    const auto batch_error_queue = Qwen3PagedSchedulerTestAccess::active_queue(
        batch_error_scheduler);
    const auto batch_error_free = batch_error_pool.free_block_count();
    const auto batch_error_used = batch_error_pool.used_block_count();
    qwen3_set_paged_batch_fault_for_testing(1, 3);
    const bool failed_batch_step = batch_error_scheduler.step();
    qwen3_clear_paged_batch_fault_for_testing();
    expect(!failed_batch_step && batch_error_scheduler.active_count() == 2 &&
               batch_error_scheduler.waiting_count() == 0 &&
               batch_error_scheduler.has_unfinished(),
           "batch error did not restore active FIFO state");
    expect_snapshot_equal(batch_error_before_401,
                          Qwen3PagedSchedulerTestAccess::request_snapshot(
                              batch_error_scheduler, 401),
                          "batch-error request 401");
    expect_snapshot_equal(batch_error_before_402,
                          Qwen3PagedSchedulerTestAccess::request_snapshot(
                              batch_error_scheduler, 402),
                          "batch-error request 402");
    expect(Qwen3PagedSchedulerTestAccess::active_queue(batch_error_scheduler) ==
               batch_error_queue && batch_error_pool.free_block_count() == batch_error_free &&
               batch_error_pool.used_block_count() == batch_error_used,
           "batch error changed active FIFO or pool snapshot");
    const auto& error_metrics = batch_error_scheduler.metrics();
    expect(!error_metrics.empty() && error_metrics.back().action == "decode_error" &&
               error_metrics.back().decode_batch_size == 2 &&
               error_metrics.back().selected_request_ids == std::vector<uint64_t>{401, 402} &&
               error_metrics.back().decode_batch_error,
           "batch error metric missing selected IDs/batch size");
    while (batch_error_scheduler.has_unfinished()) batch_error_scheduler.step();
    const auto batch_error_finished = batch_error_scheduler.take_finished();
    expect(batch_error_finished.size() == 2,
           "batch error retry did not finish both requests");
    const FinishedRequest* result_401 = nullptr;
    const FinishedRequest* result_402 = nullptr;
    for (const auto& result : batch_error_finished) {
      if (result.request_id == 401) result_401 = &result;
      if (result.request_id == 402) result_402 = &result;
    }
    expect(result_401 && result_402,
           "batch error retry results missing request ID 401 or 402");
    check_result(*result_401, batch_error_expected_401, 401);
    check_result(*result_402, batch_error_expected_402, 402);
    expect(batch_error_pool.used_block_count() == 0,
           "batch error retry did not reclaim both requests");
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "paged decode batch-error rollback/retry reference alignment and CUDA health passed\n";

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_paged_scheduler passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
