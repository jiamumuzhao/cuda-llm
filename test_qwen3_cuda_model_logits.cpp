#include "llm/cuda_check.h"
#include "llm/model_package.h"
#include "llm/ops_cuda.h"
#include "llm/qwen3_cuda_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
namespace fs = std::filesystem;

struct FixtureInfo {
  std::string file;
  std::vector<int64_t> shape;
  std::string dtype;
};

static void fail(const std::string& message) {
  throw std::runtime_error("test_qwen3_cuda_model_logits: " + message);
}

static fs::path locate(const fs::path& path) {
  if (fs::exists(path)) return path;
  if (fs::exists(fs::path("..") / path)) return fs::path("..") / path;
  fail("artifact missing: " + path.string() +
       "; run bash tools/run_phase25c.sh first");
  return {};
}

static std::vector<std::string> expected_names() {
  return {"input_ids", "position_ids", "embedding_output", "final_hidden",
          "logits", "last_token_greedy_id", "last_token_top5_ids",
          "last_token_top5_logits"};
}

static std::map<std::string, FixtureInfo> read_manifest(const fs::path& path) {
  std::ifstream input(path);
  if (!input) fail("manifest missing: " + path.string());
  std::map<std::string, FixtureInfo> result;
  std::string name, file, dtype;
  int rank = 0;
  while (input >> name >> file >> rank) {
    FixtureInfo info{file, {}, {}};
    for (int i = 0; i < rank; ++i) {
      int64_t dimension = 0;
      if (!(input >> dimension) || dimension < 0)
        fail("malformed manifest entry: " + name);
      info.shape.push_back(dimension);
    }
    if (!(input >> dtype) || (dtype != "f32" && dtype != "i32"))
      fail("invalid manifest dtype: " + name);
    info.dtype = dtype;
    if (!result.emplace(name, std::move(info)).second)
      fail("duplicate manifest tensor: " + name);
  }
  for (const auto& name : expected_names())
    if (!result.count(name)) fail("manifest missing tensor: " + name);
  if (result.size() != expected_names().size()) fail("manifest tensor count mismatch");
  return result;
}

static Tensor load_tensor(const fs::path& dir, const FixtureInfo& info) {
  if (info.dtype != "f32") fail("only F32 golden tensors are loadable: " + info.file);
  Tensor result(DType::F32, info.shape);
  std::ifstream input(dir / info.file, std::ios::binary | std::ios::ate);
  if (!input) fail("fixture file missing: " + info.file);
  const size_t bytes = result.nbytes();
  if (input.tellg() != std::streamoff(bytes)) fail("fixture size mismatch: " + info.file);
  input.seekg(0);
  input.read(reinterpret_cast<char*>(result.data()), std::streamsize(bytes));
  if (!input) fail("fixture short read: " + info.file);
  return result;
}

static std::vector<int32_t> read_metadata_vector(const fs::path& path,
                                                 const std::string& wanted) {
  std::ifstream input(path);
  if (!input) fail("case metadata missing: " + path.string());
  std::string key;
  while (input >> key) {
    std::string line;
    std::getline(input, line);
    if (key != wanted) continue;
    std::istringstream values(line);
    std::vector<int32_t> result;
    int32_t value;
    while (values >> value) result.push_back(value);
    return result;
  }
  fail("case metadata missing key: " + wanted);
  return {};
}

static int32_t read_metadata_scalar(const fs::path& path, const std::string& wanted) {
  auto values = read_metadata_vector(path, wanted);
  if (values.size() != 1) fail("metadata scalar malformed: " + wanted);
  return values[0];
}

static void require_cuda_f16(const Tensor& tensor, const std::string& name,
                             const std::vector<int64_t>& shape) {
  if (tensor.device() != DeviceType::CUDA || tensor.dtype() != DType::F16 ||
      !tensor.is_contiguous() || tensor.shape() != shape)
    fail(name + " expected CUDA/F16/contiguous shape");
}

static double cosine(const Tensor& actual, const Tensor& expected) {
  double dot = 0.0, aa = 0.0, bb = 0.0;
  for (size_t i = 0; i < actual.numel(); ++i) {
    const double a = actual.get_f32(i), b = expected.get_f32(i);
    dot += a * b; aa += a * a; bb += b * b;
  }
  return aa == 0.0 || bb == 0.0 ? (aa == bb ? 1.0 : 0.0)
                                : dot / std::sqrt(aa * bb);
}

static void compare_node(const std::string& case_name, const std::string& node,
                         const Tensor& actual_cuda, const Tensor& expected,
                         float abs_tol, float rel_tol, double cosine_tol) {
  require_cuda_f16(actual_cuda, case_name + "/" + node, expected.shape());
  Tensor actual = actual_cuda.to(DeviceType::CPU);
  float max_abs = 0.0f, mean_abs = 0.0f;
  double max_ratio = -1.0;
  size_t max_abs_index = 0, max_ratio_index = 0;
  float max_ratio_allowed = 0.0f;
  bool failed = false;
  for (size_t i = 0; i < actual.numel(); ++i) {
    const float a = actual.get_f32(i), e = expected.get_f32(i);
    if (!std::isfinite(a)) fail(case_name + "/" + node + " contains NaN/Inf");
    const float error = std::fabs(a - e);
    const float allowed = std::max(abs_tol, rel_tol * std::fabs(e));
    const double ratio = allowed == 0.0 ? (error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                                        : double(error) / double(allowed);
    if (error > max_abs) { max_abs = error; max_abs_index = i; }
    if (ratio > max_ratio) { max_ratio = ratio; max_ratio_index = i; max_ratio_allowed = allowed; }
    if (error > allowed) failed = true;
    mean_abs += error;
  }
  mean_abs /= static_cast<float>(actual.numel());
  const double similarity = cosine(actual, expected);
  std::cout << case_name << " node=" << node << " shape=";
  for (auto dimension : actual.shape()) std::cout << dimension << "x";
  std::cout << " dtype=F16 device=CUDA max_abs_error=" << max_abs
            << " mean_abs_error=" << mean_abs
            << " cosine_similarity=" << similarity
            << " absolute_tolerance=" << abs_tol
            << " relative_tolerance=" << rel_tol
            << " max_error_index=" << max_abs_index
            << " max_ratio_index=" << max_ratio_index
            << " max_error_to_allowed_ratio=" << max_ratio << "\n";
  if (failed || similarity < cosine_tol) {
    const size_t i = max_ratio_index;
    const float a = actual.get_f32(i), e = expected.get_f32(i);
    const float error = std::fabs(a - e);
    const float relative = e == 0.0f ? 0.0f : error / std::fabs(e);
    std::cerr << case_name << " node=" << node << " threshold failed index=" << i
              << " actual=" << a << " expected=" << e << " abs_error=" << error
              << " relative_error=" << relative << " allowed=" << max_ratio_allowed
              << " error_to_allowed_ratio=" << max_ratio
              << " cosine_similarity=" << similarity << " cosine_tolerance="
              << cosine_tol << "\n";
    fail(case_name + "/" + node + " golden mismatch");
  }
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

int main() {
  try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability="
              << prop.major << "." << prop.minor << "\n";
    const fs::path package_root = locate("artifacts/phase15/qwen3-0.6b-f32");
    const fs::path golden_root = locate("artifacts/phase25/qwen3_full_prefill");
    ModelPackage package(package_root);
    Qwen3CudaModel model(package_root);
    Tensor embedding = Tensor(DType::F16, {151936, 1024});
    Tensor embedding_f32 = package.load_tensor("token_embedding");
    for (size_t i = 0; i < embedding.numel(); ++i) embedding.set_f32(i, embedding_f32.get_f32(i));
    Tensor embedding_cuda = embedding.to(DeviceType::CUDA);

    for (const std::string& case_name : {"case_seq4", "case_seq8"}) {
      const fs::path case_dir = golden_root / case_name;
      auto manifest = read_manifest(case_dir / "manifest.txt");
      const auto ids = read_metadata_vector(case_dir / "metadata.txt", "token_ids");
      const auto positions = read_metadata_vector(case_dir / "metadata.txt", "position_ids");
      const int32_t golden_greedy = read_metadata_scalar(case_dir / "metadata.txt", "last_token_greedy_id");
      const auto golden_top5 = read_metadata_vector(case_dir / "metadata.txt", "last_token_top5_ids");
      if (ids.size() != positions.size() || ids.empty() || golden_top5.size() != 5)
        fail(case_name + " metadata shape invalid");
      Tensor expected_embedding = load_tensor(case_dir, manifest.at("embedding_output"));
      Tensor expected_final = load_tensor(case_dir, manifest.at("final_hidden"));
      Tensor expected_logits = load_tensor(case_dir, manifest.at("logits"));
      Tensor actual_embedding = cuda_embedding_lookup(embedding_cuda, ids);
      Tensor actual_final = model.prefill_final_hidden(ids, positions);
      Tensor actual_logits = model.prefill_logits(ids, positions);
      compare_node(case_name, "embedding_output", actual_embedding, expected_embedding,
                   5e-3f, 1e-3f, .999);
      compare_node(case_name, "final_hidden", actual_final, expected_final,
                   5e-2f, 2e-2f, .995);
      compare_node(case_name, "logits", actual_logits, expected_logits,
                   1e-1f, 2e-2f, .99);

      Tensor logits_cpu = actual_logits.to(DeviceType::CPU);
      const size_t offset = (ids.size() - 1) * 151936;
      std::vector<int32_t> order(151936);
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                        [&](int32_t a, int32_t b) {
                          return logits_cpu.get_f32(offset + a) > logits_cpu.get_f32(offset + b);
                        });
      const int32_t cuda_greedy = order[0];
      std::cout << case_name << " CUDA top5=";
      for (int i = 0; i < 5; ++i)
        std::cout << "(" << order[i] << "," << logits_cpu.get_f32(offset + order[i]) << ") ";
      std::cout << "golden top5=";
      for (int i = 0; i < 5; ++i)
        std::cout << "(" << golden_top5[i] << "," << load_tensor(case_dir, manifest.at("last_token_top5_logits")).get_f32(i) << ") ";
      std::cout << " greedy_cuda=" << cuda_greedy << " greedy_golden=" << golden_greedy << "\n";
      if (cuda_greedy != golden_greedy || !std::equal(order.begin(), order.begin() + 5, golden_top5.begin()))
        fail(case_name + " greedy/top5 mismatch");
    }

    expect_throw("token length mismatch", [&] { model.prefill_logits({1, 2, 3, 4}, {0, 1}); });
    expect_throw("position length mismatch", [&] { model.prefill_logits({1, 2, 3, 4}, {0, 1, 2}); });
    expect_throw("missing case manifest", [&] { read_manifest(golden_root / "missing/manifest.txt"); });
    expect_throw("missing package", [&] { Qwen3CudaModel missing("artifacts/phase15/missing"); });
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_cuda_model_logits passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
