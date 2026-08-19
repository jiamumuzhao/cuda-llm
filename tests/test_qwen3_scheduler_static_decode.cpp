#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_scheduler.h"

#include <cuda_runtime.h>

#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
static void fail(const std::string& message) {
  throw std::runtime_error("test_qwen3_scheduler_static_decode: " + message);
}
static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); }
  catch (const std::exception& error) {
    std::cout << "rejected " << name << ": " << error.what() << "\n";
    return;
  }
  fail(name + " was accepted");
}
static std::vector<int32_t> prompt(size_t index, size_t length) {
  std::vector<int32_t> result(length);
  for (size_t i = 0; i < length; ++i)
    result[i] = int32_t(1 + ((index * 97 + i * 31 + 11) % 150000));
  return result;
}
static InferenceRequestConfig greedy(uint64_t id, const std::vector<int32_t>& p,
                                     size_t max_new, size_t max_seq) {
  InferenceRequestConfig result;
  result.request_id = id; result.prompt_ids = p;
  result.max_new_tokens = max_new; result.max_seq_len = max_seq;
  return result;
}
static InferenceRequestConfig sampled(uint64_t id, const std::vector<int32_t>& p,
                                      size_t max_new, size_t max_seq) {
  auto result = greedy(id, p, max_new, max_seq);
  result.sampling.temperature = 0.8f; result.sampling.top_k = 50;
  result.sampling.top_p = 0.9f; result.sampling.seed = 20260727;
  return result;
}
static void compare(const FinishedRequest& actual,
                    const GreedyGenerationResult& expected,
                    const std::string& name) {
  if (actual.generated_ids != expected.generated_ids ||
      actual.stop_reason != expected.stop_reason ||
      actual.final_cache_length != expected.final_cache_length)
    fail(name + " differs from serial baseline");
}
static void print_result(const FinishedRequest& result) {
  std::cout << "request_id=" << result.request_id << " generated_ids=[";
  for (size_t i = 0; i < result.generated_ids.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << result.generated_ids[i];
  }
  std::cout << "] stop_reason=" << result.stop_reason
            << " final_cache_length=" << result.final_cache_length << "\n";
}

int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability="
              << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    const auto p4 = prompt(0, 4);
    const auto p8 = prompt(1, 8);
    const auto base1 = model.generate_greedy(p4, 3, std::nullopt, 16);
    const auto base2 = model.generate_sampled(p4, 3, std::nullopt, 16,
                                              sampled(2, p4, 3, 16).sampling);
    const auto base3 = model.generate_greedy(p8, 3, std::nullopt, 16);
    const auto base4 = model.generate_greedy(p4, 3, std::nullopt, 16);

    Qwen3RequestScheduler scheduler(model);
    scheduler.submit(greedy(1, p4, 3, 16));
    scheduler.submit(sampled(2, p4, 3, 16));
    scheduler.submit(greedy(3, p8, 3, 16));
    scheduler.submit(greedy(4, p4, 3, 16));
    if (scheduler.prefill_waiting_static_batch(4) != 2 ||
        scheduler.prefill_waiting_static_batch(4) != 1 ||
        scheduler.prefill_waiting_static_batch(4) != 1)
      fail("static prefill grouping mismatch");
    if (scheduler.waiting_count() != 0 || scheduler.decode_count() != 4)
      fail("prefill did not establish four decode requests");

    size_t round = 1;
    std::cout << "round=" << round++ << " selected_ids=[1,2] anchor_cache_length=4 batch_size=2\n";
    if (scheduler.decode_active_static_batch(4) != 2) fail("round 1 static decode mismatch");
    std::cout << "round=" << round++ << " selected_ids=[3] anchor_cache_length=8 batch_size=1\n";
    if (scheduler.decode_active_static_batch(4) != 1) fail("round 2 static decode mismatch");
    std::cout << "round=" << round++ << " selected_ids=[4] anchor_cache_length=4 batch_size=1\n";
    if (scheduler.decode_active_static_batch(4) != 1) fail("round 3 static decode mismatch");
    std::cout << "round=" << round++ << " selected_ids=[1,2] anchor_cache_length=5 batch_size=2\n";
    if (scheduler.decode_active_static_batch(4) != 2) fail("round 4 static decode mismatch");
    std::cout << "round=" << round++ << " selected_ids=[3] anchor_cache_length=9 batch_size=1\n";
    if (scheduler.decode_active_static_batch(4) != 1) fail("round 5 static decode mismatch");
    std::cout << "round=" << round++ << " selected_ids=[4] anchor_cache_length=5 batch_size=1\n";
    if (scheduler.decode_active_static_batch(4) != 1) fail("round 6 static decode mismatch");
    if (scheduler.decode_active_static_batch(4) != 0 || scheduler.has_unfinished())
      fail("static decode did not drain all requests");
    std::map<uint64_t, FinishedRequest> results;
    for (const auto& result : scheduler.take_finished()) {
      print_result(result); results.emplace(result.request_id, result);
    }
    if (results.size() != 4) fail("finished result count mismatch");
    compare(results.at(1), base1, "greedy request 1");
    compare(results.at(2), base2, "sampled request 2");
    compare(results.at(3), base3, "different-length request 3");
    compare(results.at(4), base4, "greedy request 4");
    if (scheduler.step()) fail("ordinary step changed drained static decode");
    std::cout << "static decode serial greedy/sampled and FIFO grouping passed\n";

    const int32_t eos = base1.generated_ids[1];
    Qwen3RequestScheduler stops(model);
    auto eos_request = greedy(11, p4, 8, 16); eos_request.eos_token_id = eos;
    stops.submit(eos_request);
    stops.submit(greedy(12, p4, 2, 16));
    stops.submit(greedy(13, p4, 3, 5));
    if (stops.prefill_waiting_static_batch(4) != 2 ||
        stops.prefill_waiting_static_batch(4) != 1)
      fail("stop cases prefill grouping mismatch");
    const size_t stop_round1 = stops.decode_active_static_batch(4);
    const size_t stop_round2 = stops.decode_active_static_batch(4);
    std::cout << "stop_round_counts=" << stop_round1 << "," << stop_round2 << "\n";
    if (stop_round1 != 2 || stop_round2 != 1)
      fail("stop cases static decode mismatch");
    std::map<uint64_t, FinishedRequest> stop_results;
    for (const auto& result : stops.take_finished()) {
      print_result(result); stop_results.emplace(result.request_id, result);
    }
    if (stop_results.at(11).stop_reason != "eos" || stop_results.at(11).generated_ids.size() != 2)
      fail("EOS stopping mismatch");
    if (stop_results.at(12).stop_reason != "max_new_tokens" || stop_results.at(12).generated_ids.size() != 2)
      fail("max_new_tokens stopping mismatch");
    if (stop_results.at(13).stop_reason != "cache_capacity" || stop_results.at(13).generated_ids.size() != 2 ||
        stop_results.at(13).final_cache_length != 5)
      fail("cache capacity stopping mismatch");
    std::cout << "static decode EOS/max_new_tokens/cache_capacity passed\n";

    Qwen3RequestScheduler empty(model);
    if (empty.decode_active_static_batch(1) != 0) fail("empty static decode was executable");
    expect_throw("invalid static decode batch size", [&] { empty.decode_active_static_batch(3); });
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_scheduler_static_decode passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
