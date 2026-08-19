#pragma once
#include "dtype.h"
#include "device.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
namespace llm {
struct CudaAllocationStats {
  uint64_t cuda_malloc_calls = 0;
  uint64_t cuda_free_calls = 0;
  uint64_t cuda_allocated_bytes_total = 0;
  uint64_t cuda_freed_bytes_total = 0;
  uint64_t cuda_live_bytes = 0;
  uint64_t cuda_peak_live_bytes = 0;
};

CudaAllocationStats cuda_allocation_stats();
void reset_cuda_allocation_stats();

class Tensor {
 public:
  Tensor() = default;
  Tensor(DType dtype, std::vector<int64_t> shape, DeviceType device=DeviceType::CPU);
  static Tensor from_f32(const std::vector<int64_t>& shape, const std::vector<float>& values);
  DType dtype() const { return dtype_; } DeviceType device() const { return device_; }
  const std::vector<int64_t>& shape() const { return shape_; } const std::vector<int64_t>& strides() const { return strides_; }
  size_t numel() const { return numel_; } size_t nbytes() const { return numel_ * dtype_size(dtype_); }
  size_t storage_bytes() const { return storage_size_; } size_t storage_offset() const { return offset_; }
  bool is_contiguous() const; Tensor reshape(const std::vector<int64_t>& shape) const; Tensor contiguous() const; Tensor to(DeviceType target) const;
  void* data(); const void* data() const; float get_f32(size_t i) const; void set_f32(size_t i, float v);
  float* data_f32(); const float* data_f32() const; uint16_t* data_f16(); uint16_t* data_bf16();
 private:
  DType dtype_=DType::F32; DeviceType device_=DeviceType::CPU; std::vector<int64_t> shape_, strides_;
  std::shared_ptr<void> storage_; size_t storage_size_=0, offset_=0, numel_=0;
  Tensor(DType d, DeviceType dev, std::vector<int64_t> shape, std::vector<int64_t> strides, std::shared_ptr<void> s, size_t storage_size, size_t o, size_t n);
};
}
