#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_scheduler.h"

#include <cuda_runtime.h>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
static void fail(const std::string& m) { throw std::runtime_error("test_qwen3_scheduler_variable_length_decode: " + m); }
static std::vector<int32_t> prompt(size_t row, size_t n) {
  std::vector<int32_t> r(n); for (size_t i = 0; i < n; ++i) r[i] = int32_t(1 + ((row * 101 + i * 17 + 7) % 150000)); return r;
}
static InferenceRequestConfig request(uint64_t id, const std::vector<int32_t>& p) {
  InferenceRequestConfig r; r.request_id = id; r.prompt_ids = p; r.max_new_tokens = 4; r.max_seq_len = 16; return r;
}
static std::map<uint64_t, FinishedRequest> collect(Qwen3RequestScheduler& s) {
  std::map<uint64_t, FinishedRequest> r; for (auto& x : s.take_finished()) r.emplace(x.request_id, std::move(x)); return r;
}
int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability=" << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    const std::vector<size_t> lengths{4, 6, 5, 8};
    Qwen3RequestScheduler scheduler(model);
    for (size_t i = 0; i < lengths.size(); ++i) scheduler.submit(request(i + 1, prompt(i, lengths[i])));
    if (scheduler.prefill_waiting_padded_static_batch(4) != 4) fail("padded prefill did not select all requests");
    if (scheduler.decode_count() != 4) fail("decode queue count mismatch after prefill");
    reset_variable_decode_transfer_stats();
    if (scheduler.decode_active_variable_length_batch(4) != 4) fail("mixed B=4 selection mismatch");
    if (variable_decode_transfer_stats().h2d_uploads != 1 || variable_decode_transfer_stats().d2h_validation_copies != 0) fail("scheduler metadata transfer mismatch");
    std::cout << "scheduler mixed lengths=[4,6,5,8] FIFO B=4 and metadata_h2d=1 metadata_d2h=0 passed\n";
    while (scheduler.has_unfinished()) scheduler.decode_active_variable_length_batch(4);
    auto results = collect(scheduler);
    if (results.size() != 4) fail("finished result count mismatch");
    for (size_t i = 0; i < lengths.size(); ++i)
      if (results.at(i + 1).final_cache_length != lengths[i] + 3) fail("independent cache length mismatch");
    std::cout << "scheduler independent stopping and final cache lengths passed\n";

    Qwen3RequestScheduler three(model);
    for (size_t i = 0; i < 3; ++i) three.submit(request(20 + i, prompt(20 + i, 4 + i)));
    if (three.prefill_waiting_padded_static_batch(4) != 2 ||
        three.prefill_waiting_padded_static_batch(4) != 1)
      fail("three-request prefill mismatch");
    if (three.decode_active_variable_length_batch(4) != 2 || three.decode_count() != 3)
      fail("three-request FIFO prefix rule mismatch");
    std::cout << "scheduler three-request FIFO prefix selected=2 third remains queued passed\n";
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_scheduler_variable_length_decode passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
