#include "llm/qwen3_cuda_model.h"
#include "llm/cuda_check.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
using namespace llm;
static void compare(const Tensor& a, const Tensor& b, const char* name) {
  Tensor ac = a.to(DeviceType::CPU), bc = b.to(DeviceType::CPU);
  float mx = 0.0f; double mean = 0.0, dot = 0.0, aa = 0.0, bb = 0.0;
  for (size_t i = 0; i < ac.numel(); ++i) { const float x=ac.get_f32(i), y=bc.get_f32(i), d=std::fabs(x-y); mx=std::max(mx,d); mean+=d; dot+=double(x)*y; aa+=double(x)*x; bb+=double(y)*y; }
  const double cosine=dot/std::sqrt(aa*bb);
  std::cout << name << " max_abs_error=" << mx << " mean_abs_error=" << mean/ac.numel() << " cosine=" << cosine << "\n";
  if (mx > 3e-2f || cosine < 0.99999) throw std::runtime_error(std::string(name)+" numerical mismatch");
}
static int32_t top1(const Tensor& x) {
  Tensor cpu=x.to(DeviceType::CPU); int32_t best=0; float value=cpu.get_f32(0);
  for(size_t i=1;i<cpu.numel();++i) if(cpu.get_f32(i)>value){value=cpu.get_f32(i);best=static_cast<int32_t>(i);}
  return best;
}
template<class F> static double measure(F&& fn, size_t iters) {
  fn(); CUDA_CHECK(cudaDeviceSynchronize());
  const auto start=std::chrono::steady_clock::now();
  for(size_t i=0;i<iters;++i) fn();
  CUDA_CHECK(cudaDeviceSynchronize());
  return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/iters;
}
int main() {
  try {
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    std::vector<int32_t> prompt(128); for(size_t i=0;i<prompt.size();++i) prompt[i]=int32_t(17+i*37)%150000;
    PagedKvCachePool pool({128,28,8,16,128,DType::F16});
    Qwen3PagedKvCache ref_cache(pool,512), flash_cache(pool,512);
    model.prefill_logits_paged(prompt,ref_cache); model.prefill_logits_paged(prompt,flash_cache);
    Tensor ref(DType::F16,{1,151936},DeviceType::CUDA), flash(DType::F16,{1,151936},DeviceType::CUDA);
    auto ref_ms=measure([&]{model.decode_logits_paged_into(701,ref_cache,ref);},5);
    auto flash_ms=measure([&]{model.decode_logits_paged_flash_into(701,flash_cache,flash);},5);
    compare(ref,flash,"paged_b1_flash_vs_reference");
    std::cout << "paged_b1_reference_ms=" << ref_ms << " paged_b1_flash_ms=" << flash_ms << " speedup=" << ref_ms/flash_ms << "\n";
    Qwen3PagedKvCache gen_ref(pool,512), gen_flash(pool,512);
    model.prefill_logits_paged(prompt,gen_ref); model.prefill_logits_paged(prompt,gen_flash);
    int32_t token=701;
    for(int step=0;step<5;++step){
      model.decode_logits_paged_into(token,gen_ref,ref);
      model.decode_logits_paged_flash_into(token,gen_flash,flash);
      const int32_t r=top1(ref), f=top1(flash);
      std::cout << "paged_b1_greedy_step=" << step << " reference_token=" << r << " flash_token=" << f << "\n";
      if(r!=f) throw std::runtime_error("paged_b1 greedy token mismatch");
      token=r;
    }
    std::cout << "paged_b1 greedy generation equivalence passed\n";

    std::vector<std::vector<int32_t>> prompts{{11,22,33,44},{55,66,77,88}};
    Qwen3PagedKvCache r0(pool,512), r1(pool,512), f0(pool,512), f1(pool,512);
    std::vector<Qwen3PagedKvCache*> rc{&r0,&r1}, fc{&f0,&f1};
    model.prefill_logits_paged(prompts[0],r0); model.prefill_logits_paged(prompts[1],r1);
    model.prefill_logits_paged(prompts[0],f0); model.prefill_logits_paged(prompts[1],f1);
    auto ref_batch=model.decode_logits_paged_batch({701,702},rc);
    auto flash_batch=model.decode_logits_paged_flash_batch({701,702},fc);
    compare(ref_batch,flash_batch,"paged_batch_flash_vs_reference");

    std::vector<int32_t> variable_prompt(prompt.begin(), prompt.begin() + 32);
    Qwen3KvCache vr(512), vf(512); model.prefill_logits_with_cache(variable_prompt,vr); model.prefill_logits_with_cache(variable_prompt,vf);
    auto ref_var=model.decode_logits_variable_length_batch_with_caches({703},{&vr});
    auto flash_var=model.decode_logits_variable_length_flash_batch_with_caches({703},{&vf});
    compare(ref_var,flash_var,"variable_flash_vs_reference");
    std::cout << "flash attention path validation passed\n";
    return 0;
  } catch(const std::exception& e) { std::cerr << "benchmark_flash_attention_paths: " << e.what() << "\n"; return 1; }
}
