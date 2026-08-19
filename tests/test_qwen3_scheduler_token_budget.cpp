#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_scheduler.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& message) {
  throw std::runtime_error("test_qwen3_scheduler_token_budget: " + message);
}

static InferenceRequestConfig request(uint64_t id, size_t length, size_t max_new = 1) {
  InferenceRequestConfig r;
  r.request_id = id;
  r.max_new_tokens = max_new;
  r.max_seq_len = 16;
  for (size_t i = 0; i < length; ++i)
    r.prompt_ids.push_back(static_cast<int32_t>(10 + id * 17 + i * 31));
  return r;
}

static std::map<uint64_t, FinishedRequest> collect(Qwen3RequestScheduler& scheduler) {
  std::map<uint64_t, FinishedRequest> result;
  for (auto& item : scheduler.take_finished()) result.emplace(item.request_id, std::move(item));
  return result;
}

static void check_metrics(const SchedulerBatchMetrics& m, size_t valid, size_t padded,
                          size_t padding, size_t batch, SchedulerBatchKind kind) {
  if (m.kind != kind || m.batch_size != batch || m.valid_tokens != valid ||
      m.padded_tokens != padded || m.padding_tokens != padding ||
      m.selected_request_ids.size() != batch || m.elapsed_ms < 0.0)
    fail("unexpected batch metrics values");
  const double expected_ratio = padded == 0 ? 0.0 : double(padding) / double(padded);
  if (std::fabs(m.padding_ratio - expected_ratio) > 1e-12)
    fail("unexpected padding ratio");
}

int main() {
  try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability="
              << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    const std::vector<size_t> lengths{4, 6, 5, 8};

    StaticBatchBudget budget;
    budget.max_batch_size = 4;
    budget.max_prefill_tokens = 32;
    budget.max_decode_tokens = 4;
    budget.max_context_len = 16;
    budget.max_padding_ratio = 1.0;

    Qwen3RequestScheduler prefill(model);
    for (size_t i = 0; i < lengths.size(); ++i) prefill.submit(request(i + 1, lengths[i]));
    if (prefill.prefill_waiting_budgeted_static_batch(budget) != 4 ||
        prefill.waiting_count() != 0)
      fail("budgeted prefill did not select FIFO B=4");
    const auto& prefill_metrics = prefill.batch_metrics().back();
    check_metrics(prefill_metrics, 23, 32, 9, 4, SchedulerBatchKind::Prefill);
    if (prefill_metrics.selected_request_ids != std::vector<uint64_t>({1, 2, 3, 4}))
      fail("prefill selected IDs are not FIFO");
    std::map<uint64_t, GreedyGenerationResult> prefill_baselines;
    for (size_t i = 0; i < lengths.size(); ++i) {
      prefill_baselines.emplace(i + 1, model.generate_greedy(
          request(i + 1, lengths[i]).prompt_ids, 1, std::nullopt, 16));
      const auto timing_round = prefill.first_token_round(i + 1);
      const auto ttft = prefill.ttft_ms(i + 1);
      if (!timing_round || !ttft || *ttft < 0.0 || *timing_round != prefill_metrics.round_id)
        fail("first-token round/TTFT was not recorded exactly once");
      if (prefill.state(i + 1) != RequestState::Finished)
        fail("max_new_tokens=1 request did not finish after prefill");
    }
    const auto prefill_results = collect(prefill);
    for (size_t i = 0; i < lengths.size(); ++i)
      if (prefill_results.at(i + 1).generated_ids != prefill_baselines.at(i + 1).generated_ids)
        fail("budgeted prefill first token differs from serial baseline");
    std::cout << "prefill FIFO B=4 valid_tokens=23 padded_tokens=32 padding_tokens=9 "
                 "padding_ratio=0.28125 TTFT-first-token recording passed\n";
    const std::string csv = prefill.export_batch_metrics_csv();
    if (csv.find("round_id,kind,selected_request_ids,batch_size") != 0 ||
        csv.find("1,prefill,1|2|3|4,4,4,0,0,23,32,9,0.28125,0,0,") == std::string::npos)
      fail("metrics CSV header or prefill row mismatch");
    std::cout << "metrics CSV header/IDs/fields passed\n";

    Qwen3RequestScheduler reject_ratio(model);
    reject_ratio.submit(request(10, 2));
    reject_ratio.submit(request(11, 16));
    reject_ratio.submit(request(12, 4));
    StaticBatchBudget ratio_budget = budget;
    ratio_budget.max_padding_ratio = 0.10;
    if (reject_ratio.prefill_waiting_budgeted_static_batch(ratio_budget) != 1 ||
        reject_ratio.waiting_count() != 2 || reject_ratio.state(11) != RequestState::Waiting ||
        reject_ratio.state(12) != RequestState::Waiting)
      fail("padding-ratio rejection skipped FIFO request or changed queue order");
    std::cout << "padding-ratio rejection preserved FIFO and did not skip long request passed\n";

    Qwen3RequestScheduler reject_tokens(model);
    reject_tokens.submit(request(20, 4));
    StaticBatchBudget tiny = budget;
    tiny.max_prefill_tokens = 3;
    if (reject_tokens.prefill_waiting_budgeted_static_batch(tiny) != 0 ||
        reject_tokens.waiting_count() != 1 || reject_tokens.batch_metrics().back().skipped_budget_count < 1)
      fail("prefill token-budget rejection mismatch");
    std::cout << "prefill token-budget rejection returned empty FIFO batch passed\n";

    Qwen3RequestScheduler decode(model);
    for (size_t i = 0; i < lengths.size(); ++i) decode.submit(request(30 + i, lengths[i], 2));
    if (decode.prefill_waiting_budgeted_static_batch(budget) != 4)
      fail("decode setup prefill mismatch");
    if (decode.decode_active_budgeted_static_batch(budget) != 4)
      fail("budgeted decode did not select FIFO B=4");
    const auto& decode_metrics = decode.batch_metrics().back();
    check_metrics(decode_metrics, 4, 0, 0, 4, SchedulerBatchKind::Decode);
    if (decode_metrics.decode_tokens != 4 || decode_metrics.round_id <= prefill_metrics.round_id)
      fail("decode metrics round or token count mismatch");
    auto decode_results = collect(decode);
    if (decode_results.size() != 4)
      fail("decode requests did not stop independently at max_new_tokens");
    for (size_t i = 0; i < lengths.size(); ++i) {
      auto baseline = model.generate_greedy(request(30 + i, lengths[i], 2).prompt_ids, 2,
                                            std::nullopt, 16);
      const auto& got = decode_results.at(30 + i);
      if (got.generated_ids != baseline.generated_ids || got.stop_reason != "max_new_tokens" ||
          got.final_cache_length != lengths[i] + 1)
        fail("mixed-length decode result/cache mismatch");
    }
    std::cout << "decode FIFO B=4 decode_tokens=4 padding=0 independent L->L+1 and baseline passed\n";

    Qwen3RequestScheduler decode_reject(model);
    decode_reject.submit(request(50, 4, 3));
    decode_reject.submit(request(51, 6, 3));
    if (decode_reject.prefill_waiting_budgeted_static_batch(budget) != 2)
      fail("decode rejection setup prefill mismatch");
    StaticBatchBudget context_too_small = budget;
    context_too_small.max_context_len = 4;
    if (decode_reject.decode_active_budgeted_static_batch(context_too_small) != 0 ||
        decode_reject.decode_count() != 2 || decode_reject.batch_metrics().back().skipped_budget_count < 1)
      fail("decode context-budget rejection mismatch");
    StaticBatchBudget one = budget;
    one.max_decode_tokens = 1;
    if (decode_reject.decode_active_budgeted_static_batch(one) != 1 || decode_reject.decode_count() != 2)
      fail("decode token-budget rejection/selection mismatch");
    std::cout << "decode context/token-budget rejection and FIFO preservation passed\n";

    Qwen3RequestScheduler three(model);
    three.submit(request(60, 4, 2));
    three.submit(request(61, 6, 2));
    three.submit(request(62, 5, 2));
    if (three.prefill_waiting_budgeted_static_batch(budget) != 2 || three.waiting_count() != 1 ||
        three.state(62) != RequestState::Waiting)
      fail("three-request prefill did not select FIFO legal B=2");
    if (three.decode_active_budgeted_static_batch(budget) != 2 || three.decode_count() != 0 ||
        three.waiting_count() != 1 || three.state(62) != RequestState::Waiting)
      fail("three-request decode did not select FIFO legal B=2");
    std::cout << "three-request FIFO legal-prefix B=2 and third Waiting passed\n";

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_scheduler_token_budget passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
