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
    if(a.generated_ids.size()!=3 || a.stop_reason!="max_new_tokens") fail("sampled max-new-tokens stopping mismatch");
    const int32_t first_sampled_token = a.generated_ids.front();
    auto sampled_eos=model.generate_sampled(prompt,8,first_sampled_token,16,cfg);
    if(sampled_eos.generated_ids.size()!=1 || sampled_eos.generated_ids.front()!=first_sampled_token ||
       sampled_eos.stop_reason!="eos" || sampled_eos.final_cache_length!=prompt.size()) fail("sampled EOS stopping mismatch");
    std::cout << "sampled EOS stopping: first token=" << first_sampled_token << "\n";
    SamplingConfig zero; zero.temperature=0; zero.top_k=999; zero.top_p=0; zero.seed=4;
    auto greedy=model.generate_greedy(prompt,3,std::nullopt,16), z=model.generate_sampled(prompt,3,std::nullopt,16,zero);
    if(greedy.generated_ids!=z.generated_ids || greedy.stop_reason!=z.stop_reason || greedy.final_cache_length!=z.final_cache_length) fail("temperature zero differs from greedy");
    SamplingConfig k1=cfg; k1.top_k=1; k1.top_p=1;
    auto k=model.generate_sampled(prompt,3,std::nullopt,16,k1);
    if(k.generated_ids!=greedy.generated_ids) fail("top_k=1 differs from greedy");
    auto cap=model.generate_sampled(prompt,2,std::nullopt,4,cfg);
    if(cap.generated_ids.size()!=1 || cap.stop_reason!="cache_capacity" || cap.final_cache_length!=4) fail("cache capacity stopping mismatch");
    PagedKvCachePool paged_pool(
        PagedKvCachePoolConfig{32, 28, 8, 16, 128, DType::F16});
    auto paged = model.generate_sampled_paged(
        paged_pool, prompt, 3, std::nullopt, 16, cfg);
    if (paged.generated_ids != a.generated_ids ||
        paged.stop_reason != a.stop_reason ||
        paged.final_cache_length != a.final_cache_length ||
        paged_pool.used_block_count() != 0)
      fail("paged sampled generation differs from contiguous or leaked blocks");
    std::cout << "paged sampled generation: passed\n";
    std::cout << "sampled cache-capacity stopping: passed\n";
    for (const auto* ids : {&a.generated_ids, &sampled_eos.generated_ids, &k.generated_ids, &cap.generated_ids})
      for (auto id : *ids) if(id<0 || id>=151936) fail("sampled id out of range");
    CUDA_CHECK(cudaDeviceSynchronize()); std::cout << "sampled_ids="; for(size_t i=0;i<a.generated_ids.size();++i){if(i)std::cout<<",";std::cout<<a.generated_ids[i];} std::cout << "\n";
    std::cout << "test_sampled_generation passed\n"; return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
