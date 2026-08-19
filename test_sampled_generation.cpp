#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

using namespace llm;
static void fail(const std::string& s) { throw std::runtime_error("test_sampled_generation: "+s); }
int main() {
  try {
    cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,0));
    std::cout << "CUDA device=" << p.name << " compute_capability=" << p.major << "." << p.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    const std::vector<int32_t> prompt{1,17,257,4097};
    SamplingConfig cfg; cfg.temperature=.8f; cfg.top_k=50; cfg.top_p=.9f; cfg.seed=20260727;
    auto a=model.generate_sampled(prompt,3,std::nullopt,16,cfg), b=model.generate_sampled(prompt,3,std::nullopt,16,cfg);
    if(a.generated_ids!=b.generated_ids || a.stop_reason!=b.stop_reason || a.final_cache_length!=b.final_cache_length) fail("sampled generation is not deterministic");
    SamplingConfig zero; zero.temperature=0; zero.top_k=999; zero.top_p=0; zero.seed=4;
    auto greedy=model.generate_greedy(prompt,3,std::nullopt,16), z=model.generate_sampled(prompt,3,std::nullopt,16,zero);
    if(greedy.generated_ids!=z.generated_ids || greedy.stop_reason!=z.stop_reason || greedy.final_cache_length!=z.final_cache_length) fail("temperature zero differs from greedy");
    SamplingConfig k1=cfg; k1.top_k=1; k1.top_p=1;
    auto k=model.generate_sampled(prompt,3,std::nullopt,16,k1);
    if(k.generated_ids!=greedy.generated_ids) fail("top_k=1 differs from greedy");
    auto cap=model.generate_sampled(prompt,2,std::nullopt,4,cfg);
    if(cap.generated_ids.size()!=1 || cap.stop_reason!="cache_capacity" || cap.final_cache_length!=4) fail("cache capacity stopping mismatch");
    for(auto id:a.generated_ids) if(id<0 || id>=151936) fail("sampled id out of range");
    CUDA_CHECK(cudaDeviceSynchronize()); std::cout << "sampled_ids="; for(size_t i=0;i<a.generated_ids.size();++i){if(i)std::cout<<",";std::cout<<a.generated_ids[i];} std::cout << "\n";
    std::cout << "test_sampled_generation passed\n"; return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
