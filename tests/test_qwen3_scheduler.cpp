#include "llm/cuda_check.h"
#include "llm/qwen3_scheduler.h"
#include "llm/qwen3_cuda_model.h"

#include <cuda_runtime.h>

#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
static void fail(const std::string& message) { throw std::runtime_error("test_qwen3_scheduler: " + message); }
static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); } catch (const std::exception& error) { std::cout << "rejected " << name << ": " << error.what() << "\n"; return; }
  fail(name + " was accepted");
}
static void print_result(const FinishedRequest& result) {
  std::cout << "request_id=" << result.request_id << " finished stop_reason=" << result.stop_reason
            << " final_cache_length=" << result.final_cache_length << " generated_ids=[";
  for (size_t i=0;i<result.generated_ids.size();++i) { if (i) std::cout << ','; std::cout << result.generated_ids[i]; }
  std::cout << "]\n";
}
static InferenceRequestConfig greedy(uint64_t id, const std::vector<int32_t>& prompt, size_t max_new, size_t max_seq) {
  InferenceRequestConfig r; r.request_id=id; r.prompt_ids=prompt; r.max_new_tokens=max_new; r.max_seq_len=max_seq; return r;
}
static InferenceRequestConfig sampled(uint64_t id, const std::vector<int32_t>& prompt, size_t max_new, size_t max_seq) {
  auto r=greedy(id,prompt,max_new,max_seq); r.sampling.temperature=.8f; r.sampling.top_k=50; r.sampling.top_p=.9f; r.sampling.seed=20260727; return r;
}
static std::map<uint64_t, FinishedRequest> run(Qwen3CudaModel& model, std::vector<InferenceRequestConfig> requests) {
  Qwen3RequestScheduler scheduler(model);
  for (const auto& request : requests) { scheduler.submit(request); std::cout << "request_id=" << request.request_id << " event=submit state=Waiting\n"; }
  while (scheduler.step()) {
    std::cout << "event=step waiting=" << scheduler.waiting_count() << " decode=" << scheduler.decode_count() << "\n";
  }
  if (scheduler.has_unfinished()) fail("scheduler reported unfinished after idle step");
  std::map<uint64_t, FinishedRequest> results;
  for (const auto& result : scheduler.take_finished()) { print_result(result); results.emplace(result.request_id, result); }
  if (scheduler.step()) fail("idle scheduler executed a unit");
  return results;
}
static void compare(const FinishedRequest& a, const GreedyGenerationResult& b, const std::string& name) {
  if (a.generated_ids != b.generated_ids || a.stop_reason != b.stop_reason || a.final_cache_length != b.final_cache_length)
    fail(name + " differs from greedy baseline");
}
static void compare(const FinishedRequest& a, const GreedyGenerationResult& b, const std::string& name, int) {
  compare(a,b,name);
}

int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout << "CUDA device=" << prop.name << " compute_capability=" << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    const std::vector<int32_t> prompt{1,17,257,4097};
    const auto greedy_base = model.generate_greedy(prompt,2,std::nullopt,16);
    const auto sampled_base = model.generate_sampled(prompt,2,std::nullopt,16,sampled(2,prompt,2,16).sampling);

    Qwen3RequestScheduler observable(model);
    auto r1=greedy(1,prompt,2,16); auto r2=sampled(2,prompt,2,16); auto r0=greedy(3,prompt,0,16);
    observable.submit(r1); observable.submit(r2); observable.submit(r0);
    if (observable.state(1)!=RequestState::Waiting || observable.state(2)!=RequestState::Waiting || observable.state(3)!=RequestState::Finished || observable.waiting_count()!=2) fail("submit state/count mismatch");
    std::cout << "request_id=3 event=finished stop_reason=max_new_tokens generated_ids=[]\n";
    if (!observable.step() || observable.state(1)!=RequestState::Decode || observable.waiting_count()!=1) fail("FIFO prefill observability mismatch");
    std::cout << "request_id=1 event=prefill state=Decode\n";
    if (!observable.step() || observable.state(2)!=RequestState::Decode || observable.decode_count()!=2) fail("second prefill observability mismatch");
    std::cout << "request_id=2 event=prefill state=Decode\n";
    while (observable.step()) std::cout << "event=decode round_robin\n";
    std::map<uint64_t, FinishedRequest> observed;
    for (const auto& result : observable.take_finished()) { print_result(result); observed.emplace(result.request_id,result); }
    compare(observed.at(1), greedy_base, "greedy request"); compare(observed.at(2), sampled_base, "sampled request");
    if (observable.has_unfinished() || observable.waiting_count()!=0 || observable.decode_count()!=0) fail("finished counts mismatch");
    expect_throw("collected state", [&] { observable.state(1); });
    if (observable.step()) fail("idle scheduler was executable");

    auto order_a=run(model,{greedy(11,prompt,2,16),sampled(12,prompt,2,16),greedy(13,prompt,2,16)});
    auto order_b=run(model,{greedy(13,prompt,2,16),sampled(12,prompt,2,16),greedy(11,prompt,2,16)});
    for (uint64_t id : {11ULL,12ULL,13ULL}) {
      if (order_a.at(id).generated_ids!=order_b.at(id).generated_ids || order_a.at(id).stop_reason!=order_b.at(id).stop_reason || order_a.at(id).final_cache_length!=order_b.at(id).final_cache_length) fail("submission order changed request result");
    }
    std::cout << "submission-order and sampled interleaving invariance: passed\n";

    const int32_t eos = model.generate_greedy(prompt,1,std::nullopt,16).generated_ids.front();
    Qwen3RequestScheduler eos_scheduler(model); auto eos_request=greedy(22,prompt,8,16); eos_request.eos_token_id=eos; eos_scheduler.submit(eos_request); while(eos_scheduler.step()){} auto eos_done=eos_scheduler.take_finished();
    if (eos_done.size()!=1 || eos_done[0].generated_ids.size()!=1 || eos_done[0].generated_ids[0]!=eos || eos_done[0].stop_reason!="eos" || eos_done[0].final_cache_length!=prompt.size()) fail("EOS stopping mismatch");
    print_result(eos_done[0]);
    Qwen3RequestScheduler capacity(model); capacity.submit(greedy(23,prompt,2,prompt.size())); while(capacity.step()){} auto cap_done=capacity.take_finished();
    if (cap_done.size()!=1 || cap_done[0].generated_ids.size()!=1 || cap_done[0].stop_reason!="cache_capacity" || cap_done[0].final_cache_length!=prompt.size()) fail("cache capacity stopping mismatch");
    print_result(cap_done[0]);

    Qwen3RequestScheduler errors(model);
    expect_throw("zero request ID", [&] { errors.submit(greedy(0,prompt,1,16)); });
    errors.submit(greedy(30,prompt,1,16));
    expect_throw("duplicate request ID", [&] { errors.submit(greedy(30,prompt,1,16)); });
    expect_throw("empty prompt", [&] { errors.submit(greedy(31,{},1,16)); });
    auto bad=greedy(32,prompt,1,16); bad.prompt_ids[0]=151936; expect_throw("prompt out of range", [&] { errors.submit(bad); });
    bad=greedy(33,prompt,1,16); bad.eos_token_id=151936; expect_throw("EOS out of range", [&] { errors.submit(bad); });
    bad=greedy(34,prompt,1,3); expect_throw("short max_seq_len", [&] { errors.submit(bad); });
    bad=sampled(35,prompt,1,16); bad.sampling.temperature=-1; expect_throw("invalid sampling", [&] { errors.submit(bad); });
    expect_throw("unknown state", [&] { errors.state(9999); });
    while(errors.step()){} errors.take_finished(); expect_throw("collected state", [&] { errors.state(30); });
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_scheduler passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << "\n"; return 1; }
}
