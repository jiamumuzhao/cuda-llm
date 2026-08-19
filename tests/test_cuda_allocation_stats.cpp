#include "llm/tensor.h"
#include "llm/cuda_check.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace llm;

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "SKIP test_cuda_allocation_stats: no CUDA device\n";
    return 0;
  }

  reset_cuda_allocation_stats();
  const CudaAllocationStats reset = cuda_allocation_stats();
  require(reset.cuda_malloc_calls == 0 && reset.cuda_free_calls == 0,
          "allocation stats reset did not clear call counters");
  const size_t baseline_live = reset.cuda_live_bytes;

  {
    Tensor cpu(DType::F32, {16, 16}, DeviceType::CPU);
    const CudaAllocationStats after_cpu = cuda_allocation_stats();
    require(after_cpu.cuda_malloc_calls == 0 &&
                after_cpu.cuda_allocated_bytes_total == 0,
            "CPU Tensor polluted CUDA allocation stats");

    Tensor cuda(DType::F32, {1024, 4}, DeviceType::CUDA);
    const CudaAllocationStats after_alloc = cuda_allocation_stats();
    require(after_alloc.cuda_malloc_calls >= 1,
            "CUDA Tensor did not increment allocation count");
    require(after_alloc.cuda_live_bytes > baseline_live,
            "CUDA Tensor did not increase live bytes");
    require(after_alloc.cuda_allocated_bytes_total >= cuda.nbytes(),
            "CUDA Tensor allocation bytes were not recorded");
    require(after_alloc.cuda_peak_live_bytes >= after_alloc.cuda_live_bytes,
            "peak live bytes is below live bytes");
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  const CudaAllocationStats after_free = cuda_allocation_stats();
  require(after_free.cuda_free_calls >= 1,
          "CUDA Tensor destruction did not increment free count");
  require(after_free.cuda_live_bytes == baseline_live,
          "live bytes did not return to reset baseline");
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "test_cuda_allocation_stats passed: malloc_calls="
            << after_free.cuda_malloc_calls << " free_calls="
            << after_free.cuda_free_calls << " live_bytes="
            << after_free.cuda_live_bytes << "\n";
  return 0;
}
