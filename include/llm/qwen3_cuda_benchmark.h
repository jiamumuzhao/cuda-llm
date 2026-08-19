#pragma once

#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace llm::benchmark {

struct Options {
  std::filesystem::path model;
  std::filesystem::path output;
  size_t warmup = 5;
  size_t iterations = 30;
  size_t max_seq_len = 2048;
  std::vector<size_t> contexts{4, 16};
  std::vector<size_t> batches{1, 2, 4};
};

constexpr size_t kMaxSupportedContext = 32;

void validate_options(const Options& options);

struct AllocationAggregate {
  uint64_t cuda_malloc_calls = 0;
  uint64_t cuda_free_calls = 0;
  uint64_t cuda_allocated_bytes_total = 0;
  uint64_t cuda_freed_bytes_total = 0;
  uint64_t cuda_live_bytes = 0;
  uint64_t cuda_peak_live_bytes = 0;
  size_t device_free_bytes_before = std::numeric_limits<size_t>::max();
  size_t device_free_bytes_after = std::numeric_limits<size_t>::max();
  size_t device_total_bytes = 0;
  size_t snapshot_count = 0;
};

void add_allocation_snapshot(AllocationAggregate& aggregate,
                             const CudaAllocationStats& stats,
                             size_t device_free_before,
                             size_t device_free_after,
                             size_t device_total_bytes);

}  // namespace llm::benchmark
