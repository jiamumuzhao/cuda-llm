#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
namespace llm {
inline void cuda_check(cudaError_t e,const char*expr,const char*file,int line){if(e!=cudaSuccess)throw std::runtime_error(std::string("CUDA error: ")+expr+" at "+file+":"+std::to_string(line)+": "+cudaGetErrorString(e));}
inline void cuda_check(cublasStatus_t e,const char*expr,const char*file,int line){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string("cuBLAS error: ")+expr+" at "+file+":"+std::to_string(line)+" status="+std::to_string(int(e)));}
}
#define CUDA_CHECK(expr) ::llm::cuda_check((expr),#expr,__FILE__,__LINE__)
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())
