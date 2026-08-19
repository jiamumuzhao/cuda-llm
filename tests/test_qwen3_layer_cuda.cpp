#include "llm/cuda_check.h"
#include "llm/model_package.h"
#include "llm/qwen3_layer_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
namespace fs = std::filesystem;

struct FixtureEntry { std::string file; std::vector<int64_t> shape; };

static void fail(const std::string& message) {
  throw std::runtime_error("test_qwen3_layer_cuda: " + message);
}

static std::map<std::string, FixtureEntry> read_manifest(const fs::path& path) {
  std::ifstream in(path);
  if (!in) fail("fixture manifest missing: " + path.string() +
               "; run the existing Phase 1.5 pipeline first");
  std::map<std::string, FixtureEntry> result;
  std::string name, file;
  int rank = 0;
  while (in >> name >> file >> rank) {
    FixtureEntry entry{file, {}};
    for (int i = 0; i < rank; ++i) {
      int64_t dimension = 0;
      if (!(in >> dimension)) fail("malformed fixture manifest entry " + name);
      entry.shape.push_back(dimension);
    }
    result.emplace(name, std::move(entry));
  }
  return result;
}

static Tensor load_fixture(const fs::path& dir,
                           const std::map<std::string, FixtureEntry>& manifest,
                           const std::string& name) {
  auto it = manifest.find(name);
  if (it == manifest.end()) fail("fixture tensor missing: " + name);
  std::ifstream in(dir / it->second.file, std::ios::binary | std::ios::ate);
  if (!in) fail("fixture file missing for " + name + ": " + (dir / it->second.file).string());
  const auto bytes = in.tellg();
  size_t count = 1;
  for (int64_t d : it->second.shape) count *= static_cast<size_t>(d);
  if (bytes != std::streamoff(count * sizeof(float))) fail("fixture byte size mismatch: " + name);
  in.seekg(0);
  std::vector<float> values(count);
  in.read(reinterpret_cast<char*>(values.data()), bytes);
  return Tensor::from_f32(it->second.shape, values);
}

static fs::path locate(const fs::path& relative) {
  if (fs::exists(relative)) return relative;
  if (fs::exists(fs::path("..") / relative)) return fs::path("..") / relative;
  fail("artifact missing: " + relative.string() +
       "; run bash tools/run_phase15.sh first");
  return {};
}

static Qwen3CudaLayerWeights load_weights(const ModelPackage& package) {
  return {package.load_tensor("layers.0.input_norm").to(DeviceType::CUDA),
          package.load_tensor("layers.0.q_proj").to(DeviceType::CUDA),
          package.load_tensor("layers.0.k_proj").to(DeviceType::CUDA),
          package.load_tensor("layers.0.v_proj").to(DeviceType::CUDA),
          package.load_tensor("layers.0.q_norm").to(DeviceType::CUDA),
          package.load_tensor("layers.0.k_norm").to(DeviceType::CUDA),
          package.load_tensor("layers.0.o_proj").to(DeviceType::CUDA),
          package.load_tensor("layers.0.post_attention_norm").to(DeviceType::CUDA),
          package.load_tensor("layers.0.gate_proj").to(DeviceType::CUDA),
          package.load_tensor("layers.0.up_proj").to(DeviceType::CUDA),
          package.load_tensor("layers.0.down_proj").to(DeviceType::CUDA)};
}

static double cosine(const Tensor& actual, const Tensor& expected) {
  double dot = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < actual.numel(); ++i) {
    const double a = actual.get_f32(i), b = expected.get_f32(i);
    dot += a * b; aa += a * a; bb += b * b;
  }
  return (aa == 0 || bb == 0) ? (aa == bb ? 1.0 : 0.0) : dot / std::sqrt(aa * bb);
}

static float node_max_abs_tolerance(const std::string& name) {
  if (name == "k_norm_output" || name == "k_rope") return 5e-4f;
  return 1e-4f;
}

static void compare_node(const std::string& name, const Tensor& actual,
                         const Tensor& expected) {
  if (actual.device() != DeviceType::CUDA || actual.dtype() != DType::F32 ||
      !actual.is_contiguous()) fail(name + " trace state is not contiguous CUDA F32");
  if (actual.shape() != expected.shape()) fail(name + " shape mismatch");
  Tensor actual_cpu = actual.to(DeviceType::CPU);
  float max_error = 0;
  size_t max_index = 0;
  double mean_error = 0;
  for (size_t i = 0; i < actual.numel(); ++i) {
    const float error = std::fabs(actual_cpu.get_f32(i) - expected.get_f32(i));
    if (error > max_error) { max_error = error; max_index = i; }
    mean_error += error;
  }
  mean_error /= actual.numel();
  const double similarity = cosine(actual_cpu, expected);
  const float max_tolerance = node_max_abs_tolerance(name);
  constexpr double cosine_tolerance = 0.99999;
  std::cout << name << " shape=";
  for (auto d : actual.shape()) std::cout << d << "x";
  std::cout << " dtype=F32 device=CUDA max_abs_error=" << max_error
            << " mean_abs_error=" << mean_error
            << " cosine_similarity=" << similarity
            << " max_abs_tolerance=" << max_tolerance
            << " cosine_tolerance=" << cosine_tolerance << "\n";
  if (max_error > max_tolerance || similarity < cosine_tolerance)
    fail(name + " threshold failed actual=" + std::to_string(max_error) +
         " actual_value=" + std::to_string(actual_cpu.get_f32(max_index)) +
         " expected_value=" + std::to_string(expected.get_f32(max_index)) +
         " max_abs_tolerance=" + std::to_string(max_tolerance) +
         " cosine=" + std::to_string(similarity) +
         " cosine_tolerance=" + std::to_string(cosine_tolerance));
}

static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); }
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
    const fs::path package_path = locate("artifacts/phase15/qwen3-0.6b-f32");
    const fs::path fixture_path = locate("artifacts/phase1/qwen3_layer0");
    ModelPackage package(package_path);
    const float eps = std::stof(package.config("rms_norm_eps"));
    const float theta = std::stof(package.config("rope_theta"));
    std::cout << "rms_norm_eps=" << eps << " rope_theta=" << theta << "\n";
    auto fixture_manifest = read_manifest(fixture_path / "manifest.txt");
    Tensor hidden = load_fixture(fixture_path, fixture_manifest, "hidden_states");
    Tensor position_tensor = load_fixture(fixture_path, fixture_manifest, "position_ids");
    std::vector<int32_t> positions(position_tensor.numel());
    for (size_t i = 0; i < positions.size(); ++i) positions[i] = static_cast<int32_t>(position_tensor.get_f32(i));
    Qwen3CudaLayerWeights weights = load_weights(package);
    Tensor hidden_cuda = hidden.to(DeviceType::CUDA);
    Qwen3LayerTrace trace = qwen3_decoder_layer_cuda_trace(hidden_cuda, positions, weights, eps, theta);

    std::map<std::string, const Tensor*> nodes = {
      {"input_norm", &trace.input_norm}, {"q_linear", &trace.q_linear},
      {"k_linear", &trace.k_linear}, {"v_linear", &trace.v_linear},
      {"q_norm_output", &trace.q_norm_output}, {"k_norm_output", &trace.k_norm_output},
      {"q_rope", &trace.q_rope}, {"k_rope", &trace.k_rope},
      {"attention_output", &trace.attention_output}, {"o_proj_output", &trace.o_proj_output},
      {"attention_residual", &trace.attention_residual},
      {"post_attention_norm", &trace.post_attention_norm},
      {"gate_proj_output", &trace.gate_proj_output}, {"up_proj_output", &trace.up_proj_output},
      {"swiglu_output", &trace.swiglu_output}, {"down_proj_output", &trace.down_proj_output},
      {"layer_output", &trace.layer_output}};
    for (const auto& node : nodes) compare_node(node.first, *node.second, load_fixture(fixture_path, fixture_manifest, node.first));

    expect_throw("CPU hidden_states", [&] { qwen3_decoder_layer_cuda_trace(hidden, positions, weights, eps, theta); });
    Tensor hidden_f16(DType::F16, hidden.shape());
    for (size_t i = 0; i < hidden.numel(); ++i) hidden_f16.set_f32(i, hidden.get_f32(i));
    expect_throw("F16 hidden_states", [&] { qwen3_decoder_layer_cuda_trace(hidden_f16.to(DeviceType::CUDA), positions, weights, eps, theta); });
    expect_throw("invalid hidden shape", [&] { qwen3_decoder_layer_cuda_trace(Tensor(DType::F32, {4, 1023}, DeviceType::CUDA), positions, weights, eps, theta); });
    expect_throw("position_ids length mismatch", [&] { qwen3_decoder_layer_cuda_trace(hidden_cuda, {0, 1}, weights, eps, theta); });
    expect_throw("unsupported seq_len", [&] { qwen3_decoder_layer_cuda_trace(Tensor(DType::F32, {33, 1024}, DeviceType::CUDA), std::vector<int32_t>(33), weights, eps, theta); });
    auto bad_weights = weights;
    bad_weights.q_proj = Tensor(DType::F32, {2047, 1024}, DeviceType::CUDA);
    expect_throw("q_proj shape mismatch", [&] { qwen3_decoder_layer_cuda_trace(hidden_cuda, positions, bad_weights, eps, theta); });
    expect_throw("non-positive epsilon", [&] { qwen3_decoder_layer_cuda_trace(hidden_cuda, positions, weights, 0, theta); });
    expect_throw("non-positive rope theta", [&] { qwen3_decoder_layer_cuda_trace(hidden_cuda, positions, weights, eps, 0); });
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_layer_cuda passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
