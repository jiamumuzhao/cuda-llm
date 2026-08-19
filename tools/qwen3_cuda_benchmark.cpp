#include "llm/cuda_check.h"
#include "llm/ops_cuda.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_kv_cache.h"
#include "llm/qwen3_cuda_benchmark.h"
#include "llm/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace llm;

namespace {

using llm::benchmark::Options;

[[noreturn]] void usage_error(const std::string& message) {
  throw std::invalid_argument("qwen3_cuda_benchmark: " + message +
                              " (use --help for usage)");
}

size_t parse_positive(const std::string& text, const char* name) {
  if (text.empty() || text[0] == '-') usage_error(std::string(name) + " must be positive");
  size_t pos = 0;
  unsigned long long value = 0;
  try { value = std::stoull(text, &pos, 10); }
  catch (...) { usage_error(std::string(name) + " is not an unsigned integer: " + text); }
  if (pos != text.size() || value == 0 || value > std::numeric_limits<size_t>::max())
    usage_error(std::string(name) + " is not a positive integer: " + text);
  return static_cast<size_t>(value);
}

std::vector<size_t> parse_list(const std::string& text, const char* name,
                               bool validate_batch) {
  if (text.empty()) usage_error(std::string(name) + " must not be empty");
  std::vector<size_t> result;
  std::set<size_t> seen;
  size_t begin = 0;
  while (begin <= text.size()) {
    const size_t end = text.find(',', begin);
    const std::string item = text.substr(begin, end == std::string::npos ? end : end - begin);
    const size_t value = parse_positive(item, name);
    if (!seen.insert(value).second) usage_error(std::string(name) + " contains a duplicate: " + item);
    if (validate_batch && value != 1 && value != 2 && value != 4)
      usage_error("unsupported batch size " + std::to_string(value) + "; only 1,2,4 are allowed");
    result.push_back(value);
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return result;
}

void print_help() {
  std::cout << "Usage: qwen3_cuda_benchmark --model <package_root> --output <csv_path> "
               "[--warmup N] [--iterations N] [--contexts a,b,...] [--batches 1,2,4] "
               "[--max-seq-len N]\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  std::set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") { print_help(); std::exit(0); }
    const std::vector<std::string> names{
        "--model", "--output", "--warmup", "--iterations", "--contexts",
        "--batches", "--max-seq-len"};
    if (std::find(names.begin(), names.end(), arg) == names.end())
      usage_error("unknown argument: " + arg);
    if (!seen.insert(arg).second) usage_error("duplicate argument: " + arg);
    if (i + 1 >= argc) usage_error("missing value for " + arg);
    const std::string value = argv[++i];
    if (arg == "--model") options.model = value;
    else if (arg == "--output") options.output = value;
    else if (arg == "--warmup") options.warmup = parse_positive(value, "--warmup");
    else if (arg == "--iterations") options.iterations = parse_positive(value, "--iterations");
    else if (arg == "--contexts") options.contexts = parse_list(value, "--contexts", false);
    else if (arg == "--batches") options.batches = parse_list(value, "--batches", true);
    else if (arg == "--max-seq-len") options.max_seq_len = parse_positive(value, "--max-seq-len");
  }
  llm::benchmark::validate_options(options);
  return options;
}

struct EventTimer {
  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  EventTimer() { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&end)); }
  ~EventTimer() {
    if (start) cudaEventDestroy(start);
    if (end) cudaEventDestroy(end);
  }
  template <typename Fn> float run(Fn&& fn) {
    CUDA_CHECK(cudaEventRecord(start, 0));
    fn();
    CUDA_CHECK(cudaEventRecord(end, 0));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    return ms;
  }
};

struct Summary {
  size_t count = 0;
  double mean = 0.0, p50 = 0.0, p99 = 0.0, minimum = 0.0, maximum = 0.0;
};

Summary summarize(std::vector<float> samples) {
  if (samples.empty()) throw std::runtime_error("benchmark produced no samples");
  std::sort(samples.begin(), samples.end());
  const auto percentile = [&samples](double p) {
    const double index = p * static_cast<double>(samples.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(index));
    const size_t hi = static_cast<size_t>(std::ceil(index));
    const double fraction = index - static_cast<double>(lo);
    return static_cast<double>(samples[lo]) * (1.0 - fraction) +
           static_cast<double>(samples[hi]) * fraction;
  };
  Summary out;
  out.count = samples.size();
  out.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  out.p50 = percentile(0.50);
  out.p99 = percentile(0.99);
  out.minimum = samples.front();
  out.maximum = samples.back();
  return out;
}

struct RowMeasurement {
  Summary summary;
  llm::benchmark::AllocationAggregate allocation;
  size_t free_before = 0, free_after = 0, total = 0;
};

void mem_info(size_t& free_bytes, size_t& total_bytes) {
  CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
}

std::vector<int32_t> make_prompt(size_t batch_index, size_t context) {
  std::vector<int32_t> prompt(context);
  for (size_t i = 0; i < context; ++i)
    prompt[i] = static_cast<int32_t>(1 + ((batch_index * 97 + i * 13) % 100000));
  return prompt;
}

std::vector<std::vector<int32_t>> make_prompts(size_t batch, size_t context) {
  std::vector<std::vector<int32_t>> prompts;
  for (size_t b = 0; b < batch; ++b) prompts.push_back(make_prompt(b, context));
  return prompts;
}

std::vector<Qwen3KvCache*> cache_ptrs(std::vector<Qwen3KvCache>& caches) {
  std::vector<Qwen3KvCache*> pointers;
  for (auto& cache : caches) pointers.push_back(&cache);
  return pointers;
}

RowMeasurement finish_measurement(const std::vector<float>& samples,
                                  const llm::benchmark::AllocationAggregate& aggregate) {
  RowMeasurement row;
  row.summary = summarize(samples);
  row.allocation = aggregate;
  row.free_before = aggregate.device_free_bytes_before;
  row.free_after = aggregate.device_free_bytes_after;
  row.total = aggregate.device_total_bytes;
  return row;
}

RowMeasurement measure_prefill(const Qwen3CudaModel& model, size_t batch,
                               size_t context, size_t warmup, size_t iterations,
                               size_t max_seq_len) {
  const auto prompts = make_prompts(batch, context);
  for (size_t i = 0; i < warmup; ++i) {
    std::vector<Qwen3KvCache> caches;
    for (size_t b = 0; b < batch; ++b) caches.emplace_back(max_seq_len);
    auto pointers = cache_ptrs(caches);
    { Tensor logits = model.prefill_logits_batch_with_caches(prompts, pointers); }
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  std::vector<float> samples;
  llm::benchmark::AllocationAggregate aggregate;
  for (size_t i = 0; i < iterations; ++i) {
    std::vector<Qwen3KvCache> caches;
    for (size_t b = 0; b < batch; ++b) caches.emplace_back(max_seq_len);
    auto pointers = cache_ptrs(caches);
    reset_cuda_allocation_stats();
    size_t free_before = 0, total_before = 0;
    mem_info(free_before, total_before);
    EventTimer timer;
    {
      Tensor logits;
      samples.push_back(timer.run([&] {
        logits = model.prefill_logits_batch_with_caches(prompts, pointers);
      }));
      logits = Tensor();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t free_after = 0, total_after = 0;
    mem_info(free_after, total_after);
    llm::benchmark::add_allocation_snapshot(
        aggregate, cuda_allocation_stats(), free_before, free_after,
        total_after);
  }
  return finish_measurement(samples, aggregate);
}

int32_t greedy_token(CudaSampler& sampler, const Tensor& logits, size_t row) {
  SamplingConfig greedy;
  greedy.temperature = 0.0f;
  return sampler.sample_row(logits, row, greedy, 0);
}

RowMeasurement measure_decode(const Qwen3CudaModel& model, size_t batch,
                              size_t context, size_t warmup, size_t iterations,
                              size_t max_seq_len, CudaSampler& sampler) {
  const auto prompts = make_prompts(batch, context);
  std::vector<Qwen3KvCache> caches;
  for (size_t b = 0; b < batch; ++b) caches.emplace_back(max_seq_len);
  auto pointers = cache_ptrs(caches);
  { Tensor logits = model.prefill_logits_batch_with_caches(prompts, pointers); }
  std::vector<int32_t> next_ids(batch, 0);
  for (size_t b = 0; b < batch; ++b) next_ids[b] = prompts[b].back();
  for (size_t i = 0; i < warmup; ++i) {
    Tensor logits = model.decode_logits_batch_with_caches(next_ids, pointers);
    for (size_t b = 0; b < batch; ++b) next_ids[b] = greedy_token(sampler, logits, b);
  }
  std::vector<float> samples;
  llm::benchmark::AllocationAggregate aggregate;
  for (size_t i = 0; i < iterations; ++i) {
    // The prior logits is destroyed at the end of the preceding iteration,
    // before this reset, so its free belongs to the prior sample only.
    reset_cuda_allocation_stats();
    size_t free_before = 0, total_before = 0;
    mem_info(free_before, total_before);
    EventTimer timer;
    Tensor logits;
    samples.push_back(timer.run([&] {
      logits = model.decode_logits_batch_with_caches(next_ids, pointers);
    }));
    CUDA_CHECK(cudaDeviceSynchronize());
    // Allocation observation is complete before the required greedy step.
    const CudaAllocationStats operation_stats = cuda_allocation_stats();
    for (size_t b = 0; b < batch; ++b) next_ids[b] = greedy_token(sampler, logits, b);
    logits = Tensor();
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t free_after = 0, total_after = 0;
    mem_info(free_after, total_after);
    CudaAllocationStats cleanup_stats = cuda_allocation_stats();
    cleanup_stats.cuda_malloc_calls = operation_stats.cuda_malloc_calls;
    cleanup_stats.cuda_allocated_bytes_total = operation_stats.cuda_allocated_bytes_total;
    cleanup_stats.cuda_peak_live_bytes = std::max(
        operation_stats.cuda_peak_live_bytes, cleanup_stats.cuda_peak_live_bytes);
    llm::benchmark::add_allocation_snapshot(
        aggregate, cleanup_stats, free_before, free_after, total_after);
  }
  return finish_measurement(samples, aggregate);
}

RowMeasurement measure_sample(const Qwen3CudaModel& model, size_t batch,
                              size_t context, size_t warmup, size_t iterations,
                              size_t max_seq_len, CudaSampler& sampler) {
  const auto prompts = make_prompts(batch, context);
  std::vector<Qwen3KvCache> caches;
  for (size_t b = 0; b < batch; ++b) caches.emplace_back(max_seq_len);
  auto pointers = cache_ptrs(caches);
  { Tensor logits = model.prefill_logits_batch_with_caches(prompts, pointers); }
  std::vector<int32_t> next_ids(batch, 0);
  for (size_t b = 0; b < batch; ++b) next_ids[b] = prompts[b].back();
  Tensor logits;
  for (size_t i = 0; i < warmup + iterations; ++i) {
    logits = model.decode_logits_batch_with_caches(next_ids, pointers);
    for (size_t b = 0; b < batch; ++b) next_ids[b] = greedy_token(sampler, logits, b);
  }
  // Sampling starts only after all model work and cache setup are complete.
  std::vector<float> samples;
  llm::benchmark::AllocationAggregate aggregate;
  for (size_t i = 0; i < iterations; ++i) {
    reset_cuda_allocation_stats();
    size_t free_before = 0, total_before = 0;
    mem_info(free_before, total_before);
    EventTimer timer;
    samples.push_back(timer.run([&] {
      for (size_t b = 0; b < batch; ++b) {
        SamplingConfig sampling;
        sampling.temperature = 1.0f;
        sampling.top_k = 50;
        sampling.top_p = 0.9f;
        sampling.seed = 20260727;
        (void)sampler.sample_row(logits, b, sampling, i * batch + b);
      }
    }));
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t free_after = 0, total_after = 0;
    mem_info(free_after, total_after);
    llm::benchmark::add_allocation_snapshot(
        aggregate, cuda_allocation_stats(), free_before, free_after,
        total_after);
  }
  // This logits Tensor is pre-existing sample input, not a sample temporary;
  // release it only after all snapshots so its free is excluded from sample.
  logits = Tensor();
  CUDA_CHECK(cudaDeviceSynchronize());
  return finish_measurement(samples, aggregate);
}

void write_row(std::ofstream& csv, const char* operation, size_t context_start,
               size_t context_end, size_t batch, size_t warmup, size_t iterations,
               const RowMeasurement& row, double tokens_per_second) {
  const auto& s = row.summary;
  const auto& a = row.allocation;
  csv << operation << ',' << context_start << ',' << context_end << ',' << batch << ','
      << warmup << ',' << iterations << ',' << s.count << ','
      << std::fixed << std::setprecision(6) << s.mean << ',' << s.p50 << ',' << s.p99
      << ',' << s.minimum << ',' << s.maximum << ',' << tokens_per_second << ','
      << a.cuda_malloc_calls << ',' << a.cuda_free_calls << ','
      << a.cuda_allocated_bytes_total << ',' << a.cuda_freed_bytes_total << ','
      << a.cuda_live_bytes << ',' << a.cuda_peak_live_bytes << ','
      << row.free_before << ',' << row.free_after << ',' << row.total << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (!std::filesystem::is_directory(options.model))
      usage_error("model path is not a directory: " + options.model.string());
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    int runtime_version = 0, driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    std::cout << "gpu=" << props.name << " compute_capability=" << props.major << '.'
              << props.minor << " cuda_runtime=" << runtime_version
              << " cuda_driver=" << driver_version << '\n';
    std::cout << "model=" << options.model << " warmup=" << options.warmup
              << " iterations=" << options.iterations << " max_seq_len="
              << options.max_seq_len << " resident_weight_bytes=";
    Qwen3CudaModel model(options.model);
    std::cout << model.resident_weight_bytes() << " decode_workspace=enabled"
              << " workspace_resident_bytes=" << model.decode_workspace_bytes()
              << " output=" << options.output << '\n';

    if (options.output.has_parent_path())
      std::filesystem::create_directories(options.output.parent_path());
    std::ofstream csv(options.output);
    if (!csv) throw std::runtime_error("cannot open CSV output: " + options.output.string());
    csv << "operation,context_start,context_end,batch_size,warmup,iterations,count,mean_ms,p50_ms,p99_ms,min_ms,max_ms,tokens_per_second,cuda_malloc_calls,cuda_free_calls,cuda_allocated_bytes_total,cuda_freed_bytes_total,cuda_live_bytes,cuda_peak_live_bytes,device_free_bytes_before,device_free_bytes_after,device_total_bytes\n";

    CudaSampler sampler(151936);
    for (const size_t context : options.contexts) {
      for (const size_t batch : options.batches) {
        const RowMeasurement prefill = measure_prefill(model, batch, context,
                                                        options.warmup, options.iterations,
                                                        options.max_seq_len);
        write_row(csv, "prefill", 0, context, batch, options.warmup,
                  options.iterations, prefill, 0.0);
        const RowMeasurement decode = measure_decode(model, batch, context,
                                                     options.warmup, options.iterations,
                                                     options.max_seq_len, sampler);
        write_row(csv, "decode", context + options.warmup,
                  context + options.warmup + options.iterations, batch,
                  options.warmup, options.iterations, decode,
                  static_cast<double>(batch) * 1000.0 / decode.summary.mean);
        const RowMeasurement sample = measure_sample(model, batch, context,
                                                      options.warmup, options.iterations,
                                                      options.max_seq_len, sampler);
        write_row(csv, "sample", context + options.warmup + options.iterations,
                  context + options.warmup + options.iterations + 1, batch,
                  options.warmup, options.iterations, sample, 0.0);
        csv.flush();
        std::cout << "completed operation set context=" << context << " batch=" << batch
                  << " decode_mean_ms=" << decode.summary.mean
                  << " decode_tokens_per_second="
                  << (static_cast<double>(batch) * 1000.0 / decode.summary.mean)
                  << " decode_mallocs_per_step="
                  << (decode.allocation.cuda_malloc_calls / decode.summary.count)
                  << " decode_frees_per_step="
                  << (decode.allocation.cuda_free_calls / decode.summary.count)
                  << '\n';
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "benchmark_csv=" << options.output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
