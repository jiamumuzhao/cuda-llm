#include "llm/qwen3_cuda_benchmark.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace llm::benchmark {

void validate_options(const Options& options) {
  if (options.model.empty()) throw std::invalid_argument("qwen3_cuda_benchmark: --model is required");
  if (options.output.empty()) throw std::invalid_argument("qwen3_cuda_benchmark: --output is required");
  if (options.iterations == 0) throw std::invalid_argument("qwen3_cuda_benchmark: --iterations must be > 0");
  if (options.warmup == 0) throw std::invalid_argument("qwen3_cuda_benchmark: --warmup must be > 0");
  if (options.max_seq_len == 0) throw std::invalid_argument("qwen3_cuda_benchmark: --max-seq-len must be > 0");
  if (options.contexts.empty()) throw std::invalid_argument("qwen3_cuda_benchmark: --contexts must not be empty");
  std::set<size_t> seen_contexts;
  size_t max_context = 0;
  for (const size_t context : options.contexts) {
    if (context == 0) throw std::invalid_argument("qwen3_cuda_benchmark: context must be > 0");
    if (context > kMaxSupportedContext)
      throw std::invalid_argument("qwen3_cuda_benchmark: context " + std::to_string(context) +
                                  " exceeds current prefill limit 32; long-context benchmarks require long-prefill support");
    if (!seen_contexts.insert(context).second)
      throw std::invalid_argument("qwen3_cuda_benchmark: duplicate context " + std::to_string(context));
    max_context = std::max(max_context, context);
  }
  if (options.max_seq_len < max_context ||
      options.warmup > std::numeric_limits<size_t>::max() - max_context ||
      options.iterations > std::numeric_limits<size_t>::max() - max_context - options.warmup ||
      options.max_seq_len < max_context + options.warmup + options.iterations)
    throw std::invalid_argument("qwen3_cuda_benchmark: --max-seq-len must be at least max_context + warmup + iterations");
  std::set<size_t> seen_batches;
  for (const size_t batch : options.batches) {
    if (batch != 1 && batch != 2 && batch != 4)
      throw std::invalid_argument("qwen3_cuda_benchmark: unsupported batch size " + std::to_string(batch));
    if (!seen_batches.insert(batch).second)
      throw std::invalid_argument("qwen3_cuda_benchmark: duplicate batch " + std::to_string(batch));
  }
  if (options.batches.empty()) throw std::invalid_argument("qwen3_cuda_benchmark: --batches must not be empty");
}

void add_allocation_snapshot(AllocationAggregate& aggregate,
                             const CudaAllocationStats& stats,
                             size_t device_free_before,
                             size_t device_free_after,
                             size_t device_total_bytes) {
  if (aggregate.snapshot_count != 0 && aggregate.device_total_bytes != device_total_bytes)
    throw std::runtime_error("qwen3_cuda_benchmark: device_total_bytes changed between allocation snapshots");
  aggregate.device_total_bytes = device_total_bytes;
  aggregate.cuda_malloc_calls += stats.cuda_malloc_calls;
  aggregate.cuda_free_calls += stats.cuda_free_calls;
  aggregate.cuda_allocated_bytes_total += stats.cuda_allocated_bytes_total;
  aggregate.cuda_freed_bytes_total += stats.cuda_freed_bytes_total;
  aggregate.cuda_live_bytes = std::max(aggregate.cuda_live_bytes, stats.cuda_live_bytes);
  aggregate.cuda_peak_live_bytes = std::max(aggregate.cuda_peak_live_bytes, stats.cuda_peak_live_bytes);
  aggregate.device_free_bytes_before = std::min(aggregate.device_free_bytes_before, device_free_before);
  aggregate.device_free_bytes_after = std::min(aggregate.device_free_bytes_after, device_free_after);
  ++aggregate.snapshot_count;
}

}  // namespace llm::benchmark
