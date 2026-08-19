#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_scheduler.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
static void fail(const std::string& s) { throw std::runtime_error("test_qwen3_scheduler_static_prefill: " + s); }
static InferenceRequestConfig req(uint64_t id, std::vector<int32_t> p, size_t max_new, size_t max_seq) {
  InferenceRequestConfig r; r.request_id=id; r.prompt_ids=std::move(p); r.max_new_tokens=max_new; r.max_seq_len=max_seq; return r;
}
static InferenceRequestConfig sampled(uint64_t id, std::vector<int32_t> p) {
  auto r=req(id,std::move(p),2,16); r.sampling.temperature=.8f; r.sampling.top_k=50; r.sampling.top_p=.9f; r.sampling.seed=20260727; return r;
}
static void equal(const FinishedRequest& a, const GreedyGenerationResult& b, const std::string& name) {
  if (a.generated_ids != b.generated_ids || a.stop_reason != b.stop_reason || a.final_cache_length != b.final_cache_length) fail(name + " differs from baseline");
}
static void print(const FinishedRequest& r) { std::cout << "request_id=" << r.request_id << " generated_ids=["; for (size_t i=0;i<r.generated_ids.size();++i) { if(i) std::cout<<','; std::cout<<r.generated_ids[i]; } std::cout << "] stop_reason=" << r.stop_reason << " final_cache_length=" << r.final_cache_length << "\n"; }

int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    std::cout << "CUDA device=" << prop.name << " compute_capability=" << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    const std::vector<int32_t> p4{1,17,257,4097};
    const std::vector<int32_t> p8{3,19,259,4099,5,23,263,4103};
    const auto base1=model.generate_greedy(p4,2,std::nullopt,16);
    const auto base2=model.generate_sampled(p4,2,std::nullopt,16,sampled(2,p4).sampling);
    const auto base4=model.generate_greedy(p4,2,std::nullopt,16);
    const auto base3=model.generate_greedy(p8,2,std::nullopt,16);

    Qwen3RequestScheduler scheduler(model);
    scheduler.submit(req(1,p4,2,16));
    scheduler.submit(sampled(2,p4));
    scheduler.submit(req(3,p8,2,16));
    scheduler.submit(req(4,p4,2,16));
    if (scheduler.prefill_waiting_static_batch(4) != 3) fail("static batch did not select three equal-length requests");
    if (scheduler.waiting_count()!=1 || scheduler.state(3)!=RequestState::Waiting || scheduler.decode_count()!=3) fail("FIFO/static grouping state mismatch");
    std::cout << "static_prefill selected=3 anchor_seq_len=4 max_batch_size=4 unselected_fifo_preserved=passed\n";
    while (scheduler.step()) {}
    if (scheduler.prefill_waiting_static_batch(4) != 1 || scheduler.decode_count()!=1) fail("second static batch did not select remaining seq_len=8 request");
    while (scheduler.step()) {}
    std::map<uint64_t,FinishedRequest> got;
    for (const auto& r:scheduler.take_finished()) { print(r); got.emplace(r.request_id,r); }
    if (got.size()!=4) fail("static scheduler result count mismatch");
    equal(got.at(1),base1,"greedy request 1"); equal(got.at(2),base2,"sampled request 2"); equal(got.at(4),base4,"greedy request 4"); equal(got.at(3),base3,"seq_len 8 request");
    std::cout << "sampled result invariant under static batch composition/order: passed\n";

    // Verify EOS, max-new-tokens, and capacity handling after batched prefill.
    const int32_t eos = base1.generated_ids.front();
    Qwen3RequestScheduler stops(model);
    auto eos_req=req(11,p4,8,16); eos_req.eos_token_id=eos;
    stops.submit(eos_req); stops.submit(req(12,p4,1,16)); stops.submit(req(13,p4,2,4));
    if (stops.prefill_waiting_static_batch(4)!=3) fail("stop cases were not batched");
    while(stops.step()){}
    std::map<uint64_t,FinishedRequest> stop_results; for(const auto& r:stops.take_finished()) { print(r); stop_results.emplace(r.request_id,r); }
    if (stop_results.at(11).stop_reason!="eos" || stop_results.at(11).generated_ids.size()!=1 || stop_results.at(11).final_cache_length!=p4.size()) fail("EOS stop mismatch");
    if (stop_results.at(12).stop_reason!="max_new_tokens" || stop_results.at(12).generated_ids.size()!=1) fail("max_new_tokens stop mismatch");
    if (stop_results.at(13).stop_reason!="cache_capacity" || stop_results.at(13).generated_ids.size()!=1 || stop_results.at(13).final_cache_length!=p4.size()) fail("cache capacity stop mismatch");
    std::cout << "static prefill EOS/max_new_tokens/cache_capacity stopping: passed\n";
    for (const auto& [id,r] : stop_results) for (int32_t token:r.generated_ids) if(token<0 || token>=151936) fail("generated token out of range");
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_scheduler_static_prefill passed\n";
    return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
