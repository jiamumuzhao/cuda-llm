#include "llm/cuda_check.h"
#include "llm/ops_cuda.h"
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace llm;
static void fail(const std::string& s) { throw std::runtime_error("test_sampling_cuda: " + s); }
static Tensor f16(const Tensor& x) { Tensor y(DType::F16, x.shape()); for (size_t i=0;i<x.numel();++i) y.set_f32(i,x.get_f32(i)); return y; }
static void reject(const std::string& n, const std::function<void()>& f) { try { f(); } catch (const std::exception& e) { std::cout << "rejected " << n << ": " << e.what() << "\n"; return; } fail(n+" was accepted"); }
static bool in(const std::vector<int32_t>& ids, int32_t x) { for (auto v:ids) if(v==x) return true; return false; }
int main() {
  try {
    cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,0));
    std::cout << "CUDA device=" << p.name << " compute_capability=" << p.major << "." << p.minor << "\n";
    CudaSampler sampler(151936);
    Tensor x = Tensor::from_f32({2,6}, {-3,1,8,2,4,0, 2,9,1,7,3,5});
    auto xf = x.to(DeviceType::CUDA); auto x16 = f16(x).to(DeviceType::CUDA);
    SamplingConfig zero; zero.temperature=0; zero.top_k=0; zero.top_p=0.0f; // ignored at temperature zero
    if (sampler.sample_last_row(xf,zero,0) != cuda_argmax_last_row(xf) || sampler.sample_last_row(x16,zero,0) != cuda_argmax_last_row(x16)) fail("temperature zero is not argmax");
    SamplingConfig one; one.temperature=1; one.top_k=1; one.top_p=1;
    if (sampler.sample_last_row(xf,one,0) != cuda_argmax_last_row(xf)) fail("top_k=1 is not greedy");
    SamplingConfig d; d.temperature=1; d.top_k=3; d.top_p=1; d.seed=42;
    auto a=sampler.sample_last_row(xf,d,7), b=sampler.sample_last_row(xf,d,7); if(a!=b) fail("same seed/offset is not deterministic");
    Tensor small = Tensor::from_f32({1,6},{10,9,0,-1,-2,-3}).to(DeviceType::CUDA);
    SamplingConfig k2; k2.temperature=1; k2.top_k=2; k2.top_p=1; k2.seed=1;
    for(uint64_t i=0;i<20;++i) if(!in({0,1},sampler.sample_last_row(small,k2,i))) fail("top_k boundary escaped");
    SamplingConfig combo; combo.temperature=1; combo.top_k=3; combo.top_p=.5f; combo.seed=2;
    for(uint64_t i=0;i<20;++i) if(!in({0,1},sampler.sample_last_row(small,combo,i))) fail("top_k/top_p combination escaped");
    std::vector<float> large(151936,-2); large[123456]=8;
    Tensor lv=Tensor::from_f32({1,151936},large).to(DeviceType::CUDA);
    if(sampler.sample_last_row(lv,zero,0)!=123456 || sampler.sample_last_row(f16(Tensor::from_f32({1,151936},large)).to(DeviceType::CUDA),zero,0)!=123456) fail("large vocab argmax failed");
    reject("CPU", [&]{sampler.sample_last_row(x,one,0);});
    reject("rank", [&]{sampler.sample_last_row(Tensor(DType::F32,{1,2,1},DeviceType::CUDA),one,0);});
    reject("BF16", [&]{sampler.sample_last_row(Tensor(DType::BF16,{1,2},DeviceType::CUDA),one,0);});
    reject("zero shape", [&]{sampler.sample_last_row(Tensor(DType::F32,{0,2},DeviceType::CUDA),one,0);});
    SamplingConfig bad=one; bad.temperature=-1; reject("negative temperature", [&]{sampler.sample_last_row(xf,bad,0);});
    bad=one; bad.top_k=7; reject("top_k too large", [&]{sampler.sample_last_row(xf,bad,0);});
    bad=one; bad.top_p=0; reject("top_p zero", [&]{sampler.sample_last_row(xf,bad,0);});
    CUDA_CHECK(cudaDeviceSynchronize()); std::cout << "test_sampling_cuda passed\n"; return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
