#include "llm/tensor.h"
#include "llm/cuda_check.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace llm;
static void ok(bool x,const char*s){if(!x)throw std::runtime_error(std::string("test_tensor_cuda: ")+s);}
int main(){try{cudaDeviceProp p{};CUDA_CHECK(cudaGetDeviceProperties(&p,0));std::cout<<"CUDA device="<<p.name<<" compute_capability="<<p.major<<"."<<p.minor<<"\n";
 auto cpu=Tensor::from_f32({2,5},{0,-1.25f,2.5f,0,9,-7,3.125f,0.5f,-2,4});auto gpu=cpu.to(DeviceType::CUDA);CUDA_CHECK(cudaDeviceSynchronize());ok(gpu.device()==DeviceType::CUDA&&gpu.shape()==cpu.shape()&&gpu.strides()==cpu.strides()&&gpu.nbytes()==cpu.nbytes(),"F32 metadata");auto back=gpu.to(DeviceType::CPU);for(size_t i=0;i<cpu.numel();++i)ok(back.get_f32(i)==cpu.get_f32(i),"F32 round trip");
 Tensor h(DType::F16,{2,4});float vals[]={0,-1.25f,2.5f,9,-7,3.125f,.5f,-2};for(size_t i=0;i<8;++i)h.set_f32(i,vals[i]);auto hg=h.to(DeviceType::CUDA);auto hb=hg.to(DeviceType::CPU);float mx=0;for(size_t i=0;i<8;++i)mx=std::max(mx,std::fabs(hb.get_f32(i)-h.get_f32(i)));std::cout<<"F16 round_trip max_abs_error="<<mx<<"\n";ok(mx<=1e-3f,"F16 round trip");auto view=gpu.reshape({5,2});auto copy=view.to(DeviceType::CUDA);ok(copy.storage_bytes()==gpu.storage_bytes()&&view.shape()==std::vector<int64_t>({5,2}),"CUDA view/copy lifetime");
 bool rejected=false;try{Tensor b(DType::BF16,{2});b.to(DeviceType::CUDA);}catch(const std::exception&){rejected=true;}ok(rejected,"BF16 CUDA rejection");rejected=false;try{gpu.get_f32(0);}catch(const std::exception&){rejected=true;}ok(rejected,"CUDA host access rejection");for(int i=0;i<100;++i){Tensor t=cpu.to(DeviceType::CUDA);auto r=t.reshape({10});(void)r;}CUDA_CHECK(cudaDeviceSynchronize());std::cout<<"test_tensor_cuda passed\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
