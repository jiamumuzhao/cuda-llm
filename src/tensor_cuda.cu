#include "llm/tensor.h"
#include "llm/cuda_check.h"
#include <exception>
#include <mutex>
#include <unordered_map>
namespace llm {
namespace {
std::mutex g_cuda_stats_mutex;
CudaAllocationStats g_cuda_stats;
std::unordered_map<void*, size_t> g_cuda_allocation_sizes;
}
CudaAllocationStats cuda_allocation_stats() {
  std::lock_guard<std::mutex> lock(g_cuda_stats_mutex);
  return g_cuda_stats;
}
void reset_cuda_allocation_stats() {
  std::lock_guard<std::mutex> lock(g_cuda_stats_mutex);
  g_cuda_stats.cuda_malloc_calls = 0;
  g_cuda_stats.cuda_free_calls = 0;
  g_cuda_stats.cuda_allocated_bytes_total = 0;
  g_cuda_stats.cuda_freed_bytes_total = 0;
  // Allocations which were live before reset form the baseline for this
  // measurement window and must not be attributed to the operation.
  g_cuda_stats.cuda_peak_live_bytes = g_cuda_stats.cuda_live_bytes;
}
namespace detail {
void* cuda_allocate(size_t n) {
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n));
  {
    std::lock_guard<std::mutex> lock(g_cuda_stats_mutex);
    g_cuda_allocation_sizes[p] = n;
    ++g_cuda_stats.cuda_malloc_calls;
    g_cuda_stats.cuda_allocated_bytes_total += n;
    g_cuda_stats.cuda_live_bytes += n;
    if (g_cuda_stats.cuda_live_bytes > g_cuda_stats.cuda_peak_live_bytes)
      g_cuda_stats.cuda_peak_live_bytes = g_cuda_stats.cuda_live_bytes;
  }
  return p;
}
void cuda_release(void* p) {
  if (!p) return;
  cudaError_t e = cudaFree(p);
  if (e != cudaSuccess) std::terminate();
  std::lock_guard<std::mutex> lock(g_cuda_stats_mutex);
  const auto it = g_cuda_allocation_sizes.find(p);
  if (it != g_cuda_allocation_sizes.end()) {
    const size_t n = it->second;
    g_cuda_allocation_sizes.erase(it);
    ++g_cuda_stats.cuda_free_calls;
    g_cuda_stats.cuda_freed_bytes_total += n;
    g_cuda_stats.cuda_live_bytes -= n;
  }
}
}
Tensor Tensor::to(DeviceType target)const{if(target==device_)return *this;if(target==DeviceType::CUDA){if(dtype_==DType::BF16)throw std::runtime_error("Tensor::to CUDA: BF16 is unsupported");Tensor out(dtype_,shape_,DeviceType::CUDA);CUDA_CHECK(cudaMemcpy(out.data(),data(),nbytes(),cudaMemcpyHostToDevice));return out;}Tensor out(dtype_,shape_,DeviceType::CPU);CUDA_CHECK(cudaMemcpy(out.data(),data(),nbytes(),cudaMemcpyDeviceToHost));return out;}
}
