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
static void fail(const std::string& s) { throw std::runtime_error("test_qwen3_scheduler_padded_prefill: " + s); }
static InferenceRequestConfig request(uint64_t id, size_t length) {
  InferenceRequestConfig r; r.request_id = id; r.max_new_tokens = 1; r.max_seq_len = length;
  for (size_t i = 0; i < length; ++i) r.prompt_ids.push_back(int32_t(10 + id * 17 + i * 31));
  return r;
}

int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability=" << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    Qwen3RequestScheduler scheduler(model);
    const std::vector<size_t> lengths{4, 6, 5, 8};
    for (size_t i = 0; i < lengths.size(); ++i) scheduler.submit(request(i + 1, lengths[i]));
    if (scheduler.prefill_waiting_padded_static_batch(4) != 4 || scheduler.waiting_count() != 0)
      fail("padded scheduler did not consume FIFO batch of four");
    std::map<uint64_t, FinishedRequest> results;
    for (const auto& r : scheduler.take_finished()) results.emplace(r.request_id, r);
    if (results.size() != 4) fail("padded scheduler result count mismatch");
    for (size_t i = 0; i < lengths.size(); ++i) {
      const auto& got = results.at(i + 1);
      auto baseline = model.generate_greedy(request(i + 1, lengths[i]).prompt_ids, 1, std::nullopt, lengths[i]);
      if (got.generated_ids != baseline.generated_ids || got.stop_reason != "max_new_tokens" ||
          got.final_cache_length != lengths[i])
        fail("padded scheduler result/cache length mismatch for request " + std::to_string(i + 1));
      std::cout << "request_id=" << i + 1 << " valid_length=" << lengths[i]
                << " final_cache_length=" << got.final_cache_length << " first_token=" << got.generated_ids.front() << " passed\n";
    }

    Qwen3RequestScheduler fifo(model);
    fifo.submit(request(11, 4)); fifo.submit(request(12, 6)); fifo.submit(request(13, 5));
    if (fifo.prefill_waiting_padded_static_batch(4) != 2 || fifo.waiting_count() != 1 ||
        fifo.state(13) != RequestState::Waiting)
      fail("three selectable requests did not leave third FIFO request waiting");
    std::cout << "three-request FIFO rule selected=2 third=Waiting passed\n";
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_scheduler_padded_prefill passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
