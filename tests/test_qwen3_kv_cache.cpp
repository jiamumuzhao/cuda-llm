#include "llm/cuda_check.h"
#include "llm/model_package.h"
#include "llm/ops_cuda.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_kv_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
namespace fs = std::filesystem;

static void fail(const std::string& message) {
  throw std::runtime_error("test_qwen3_kv_cache: " + message);
}

static fs::path locate(const fs::path& path) {
  if (fs::exists(path)) return path;
  if (fs::exists(fs::path("..") / path)) return fs::path("..") / path;
  fail("package missing: " + path.string() + "; run bash tools/run_phase15.sh first");
  return {};
}

static float max_abs(const Tensor& actual, const Tensor& expected) {
  if (actual.shape() != expected.shape()) fail("comparison shape mismatch");
  float result = 0.0f;
  for (size_t i = 0; i < actual.numel(); ++i)
    result = std::max(result, std::fabs(actual.get_f32(i) - expected.get_f32(i)));
  return result;
}

static double cosine(const Tensor& actual, const Tensor& expected) {
  double dot = 0.0, aa = 0.0, bb = 0.0;
  for (size_t i = 0; i < actual.numel(); ++i) {
    double a = actual.get_f32(i), b = expected.get_f32(i);
    dot += a * b; aa += a * a; bb += b * b;
  }
  return aa == 0.0 || bb == 0.0 ? (aa == bb ? 1.0 : 0.0)
                                : dot / std::sqrt(aa * bb);
}

static std::vector<int32_t> top5(const Tensor& logits_cpu) {
  const size_t base = (logits_cpu.shape()[0] - 1) * 151936;
  std::vector<int32_t> ids(151936);
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + 5, ids.end(),
                    [&](int32_t a, int32_t b) {
                      return logits_cpu.get_f32(base + a) > logits_cpu.get_f32(base + b);
                    });
  ids.resize(5);
  return ids;
}

static void compare_logits(const std::string& name, const Tensor& actual_cuda,
                           const Tensor& expected_cuda) {
  if (actual_cuda.device() != DeviceType::CUDA || actual_cuda.dtype() != DType::F16 ||
      !actual_cuda.is_contiguous() || actual_cuda.shape() != expected_cuda.shape())
    fail(name + " output must be CUDA/F16/contiguous with matching shape");
  Tensor actual = actual_cuda.to(DeviceType::CPU);
  Tensor expected = expected_cuda.to(DeviceType::CPU);
  float max_error = 0.0f, mean_error = 0.0f;
  size_t max_index = 0;
  double max_ratio = 0.0;
  size_t max_ratio_index = 0;
  float max_allowed = 0.0f;
  bool element_failed = false;
  for (size_t i = 0; i < actual.numel(); ++i) {
    float a = actual.get_f32(i), e = expected.get_f32(i);
    if (!std::isfinite(a)) fail(name + " produced NaN/Inf");
    float error = std::fabs(a - e);
    float allowed = std::max(5e-3f, 1e-3f * std::fabs(e));
    if (error > allowed) element_failed = true;
    if (error > max_error) { max_error = error; max_index = i; }
    double ratio = allowed == 0.0f ? 0.0 : double(error) / double(allowed);
    if (ratio > max_ratio) { max_ratio = ratio; max_ratio_index = i; max_allowed = allowed; }
    mean_error += error;
  }
  mean_error /= static_cast<float>(actual.numel());
  const double sim = cosine(actual, expected);
  auto actual_top5 = top5(actual);
  auto expected_top5 = top5(expected);
  const int32_t actual_greedy = actual_top5[0];
  const int32_t expected_greedy = expected_top5[0];
  std::cout << name << " shape=" << actual.shape()[0] << "x" << actual.shape()[1]
            << " max_abs_error=" << max_error << " mean_abs_error=" << mean_error
            << " cosine_similarity=" << sim << " tolerance=0.005/0.001"
            << " max_error_index=" << max_index
            << " max_ratio_index=" << max_ratio_index
            << " max_error_to_allowed_ratio=" << max_ratio
            << " greedy=" << actual_greedy << " expected_greedy=" << expected_greedy
            << " top5=";
  for (auto id : actual_top5) std::cout << id << ",";
  std::cout << " expected_top5=";
  for (auto id : expected_top5) std::cout << id << ",";
  std::cout << "\n";
  if (element_failed) {
    const size_t i = max_ratio_index;
    const float a = actual.get_f32(i), e = expected.get_f32(i);
    const float error = std::fabs(a - e);
    std::cerr << name << " element_failed index=" << i << " actual=" << a
              << " expected=" << e << " abs_error=" << error
              << " relative_error=" << (e == 0.0f ? 0.0f : error / std::fabs(e))
              << " allowed=" << max_allowed << " ratio=" << max_ratio << "\n";
  }
  if (element_failed || sim < 0.9999 || actual_top5 != expected_top5)
    fail(name + " cosine or top5 mismatch");
}

static void expect_throw(const std::string& name,
                         const std::function<void()>& function) {
  try { function(); }
  catch (const std::exception& error) {
    std::cout << "rejected " << name << ": " << error.what() << "\n";
    return;
  }
  fail(name + " was accepted");
}

static Tensor last_row_cuda(const Tensor& full_cuda) {
  Tensor full = full_cuda.to(DeviceType::CPU);
  Tensor row(DType::F16, {1, full.shape()[1]});
  const size_t offset = (full.shape()[0] - 1) * full.shape()[1];
  for (size_t i = 0; i < row.numel(); ++i) row.set_f32(i, full.get_f32(offset + i));
  return row.to(DeviceType::CUDA);
}

int main() {
  try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability="
              << prop.major << "." << prop.minor << "\n";
    const fs::path package = locate("artifacts/phase15/qwen3-0.6b-f32");
    Qwen3CudaModel model(package);
    Qwen3KvCache cache(32);
    const size_t bytes_before = cache.resident_bytes();
    const std::vector<int32_t> prompt{1, 17, 257, 4097};
    const std::vector<int32_t> continuation{8193, 16385, 32769, 65537};
    std::cout << "cache capacity=" << cache.capacity() << " resident_bytes="
              << cache.resident_bytes() << "\n";

    Tensor cached_prompt = model.prefill_logits_with_cache(prompt, cache);
    Tensor plain_prompt = model.prefill_logits(prompt, {0, 1, 2, 3});
    compare_logits("prefill_prompt", cached_prompt, plain_prompt);
    if (cache.length() != 4) fail("prefill cache length is not 4");

    std::vector<int32_t> sequence = prompt;
    for (size_t step = 0; step < continuation.size(); ++step) {
      Tensor decoded = model.decode_logits(continuation[step], cache);
      sequence.push_back(continuation[step]);
      std::vector<int32_t> positions(sequence.size());
      std::iota(positions.begin(), positions.end(), 0);
      Tensor full = model.prefill_logits(sequence, positions);
      compare_logits("decode_step_" + std::to_string(step + 1), decoded,
                     last_row_cuda(full));
      if (cache.length() != sequence.size()) fail("cache length mismatch after decode");
    }
    if (cache.length() != 8) fail("final cache length is not 8");
    if (cache.resident_bytes() != bytes_before) fail("resident bytes changed after decode");

    cache.reset();
    if (cache.length() != 0 || cache.resident_bytes() != bytes_before)
      fail("reset changed logical length or resident bytes");
    Tensor repeat = model.prefill_logits_with_cache(prompt, cache);
    compare_logits("prefill_repeat", repeat, plain_prompt);

    expect_throw("decode empty cache", [&] {
      Qwen3KvCache empty(32);
      model.decode_logits(1, empty);
    });
    expect_throw("prefill non-empty cache", [&] {
      model.prefill_logits_with_cache(prompt, cache);
    });
    expect_throw("decode overflow", [&] {
      Qwen3KvCache small(1);
      model.prefill_logits_with_cache({1}, small);
      model.decode_logits(2, small);
    });
    expect_throw("prefill over capacity", [&] {
      Qwen3KvCache small(3);
      model.prefill_logits_with_cache(prompt, small);
    });
    expect_throw("token out of range", [&] {
      Qwen3KvCache bad(32);
      model.prefill_logits_with_cache({1, 2, 3, 151936}, bad);
    });
    expect_throw("cache layer count mismatch", [&] {
      Qwen3KvCache bad(32, 27);
      model.prefill_logits_with_cache(prompt, bad);
    });
    expect_throw("cache dtype/shape mismatch", [&] {
      Qwen3KvCache bad(4, 28, 8, 128);
      Tensor wrong(DType::F32, {1, 8, 128}, DeviceType::CUDA);
      bad.write_prefill_layer(0, wrong, wrong, 1);
    });
    Tensor q_bad(DType::F16, {1, 15, 128}, DeviceType::CUDA);
    Tensor k_ok(DType::F16, {4, 8, 128}, DeviceType::CUDA);
    expect_throw("decode attention bad q shape", [&] {
      cuda_gqa_decode_attention(q_bad, k_ok, k_ok, 1);
    });
    expect_throw("decode attention empty cache", [&] {
      Tensor q(DType::F16, {1, 16, 128}, DeviceType::CUDA);
      cuda_gqa_decode_attention(q, k_ok, k_ok, 0);
    });
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_kv_cache passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
