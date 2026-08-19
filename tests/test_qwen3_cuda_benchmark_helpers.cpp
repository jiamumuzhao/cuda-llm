#include "llm/qwen3_cuda_benchmark.h"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace llm;
using namespace llm::benchmark;

static void expect_reject(const Options& options, const char* name) {
  try {
    validate_options(options);
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(std::string(name) + " was accepted unexpectedly");
}

int main() {
  Options defaults;
  if (defaults.contexts != std::vector<size_t>{4, 16})
    throw std::runtime_error("default contexts are not {4,16}");
  defaults.model = "mock-model";
  defaults.output = "mock.csv";
  validate_options(defaults);

  Options long_context = defaults;
  long_context.contexts = {4, 128};
  expect_reject(long_context, "context 128");

  Options too_short = defaults;
  too_short.max_seq_len = 10;
  expect_reject(too_short, "insufficient max_seq_len");

  AllocationAggregate aggregate;
  CudaAllocationStats first;
  first.cuda_malloc_calls = 2;
  first.cuda_free_calls = 1;
  first.cuda_allocated_bytes_total = 100;
  first.cuda_freed_bytes_total = 20;
  first.cuda_live_bytes = 80;
  first.cuda_peak_live_bytes = 120;
  add_allocation_snapshot(aggregate, first, 900, 920, 1000);

  CudaAllocationStats second;
  second.cuda_malloc_calls = 3;
  second.cuda_free_calls = 4;
  second.cuda_allocated_bytes_total = 300;
  second.cuda_freed_bytes_total = 280;
  second.cuda_live_bytes = 100;
  second.cuda_peak_live_bytes = 180;
  add_allocation_snapshot(aggregate, second, 850, 880, 1000);

  if (aggregate.snapshot_count != 2 || aggregate.cuda_malloc_calls != 5 ||
      aggregate.cuda_free_calls != 5 || aggregate.cuda_allocated_bytes_total != 400 ||
      aggregate.cuda_freed_bytes_total != 300 || aggregate.cuda_live_bytes != 100 ||
      aggregate.cuda_peak_live_bytes != 180 || aggregate.device_free_bytes_before != 850 ||
      aggregate.device_free_bytes_after != 880)
    throw std::runtime_error("allocation aggregate did not sum/max/min snapshots correctly");

  std::cout << "test_qwen3_cuda_benchmark_helpers passed: defaults={4,16}, "
               "long-context rejection and two-snapshot aggregate verified\n";
  return 0;
}
