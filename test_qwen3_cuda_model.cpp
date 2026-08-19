#include "llm/cuda_check.h"
#include "llm/model_package.h"
#include "llm/ops_cuda.h"
#include "llm/qwen3_cuda_model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
namespace fs = std::filesystem;

static void fail(const std::string& message) {
  throw std::runtime_error("test_qwen3_cuda_model: " + message);
}

static fs::path locate(const fs::path& path) {
  if (fs::exists(path)) return path;
  if (fs::exists(fs::path("..") / path)) return fs::path("..") / path;
  fail("artifact missing: " + path.string() +
       "; run bash tools/run_phase15.sh first");
  return {};
}

static Tensor to_f16_cpu(const Tensor& input) {
  Tensor output(DType::F16, input.shape());
  for (size_t i = 0; i < input.numel(); ++i)
    output.set_f32(i, input.get_f32(i));
  return output;
}

static Qwen3CudaLayerWeights load_layer0_f16(const ModelPackage& package) {
  auto load = [&](const std::string& name) {
    return to_f16_cpu(package.load_tensor("layers.0." + name))
        .to(DeviceType::CUDA);
  };
  return {load("input_norm"), load("q_proj"), load("k_proj"),
          load("v_proj"), load("q_norm"), load("k_norm"), load("o_proj"),
          load("post_attention_norm"), load("gate_proj"), load("up_proj"),
          load("down_proj")};
}

static Tensor load_embedding_f16(const ModelPackage& package) {
  return to_f16_cpu(package.load_tensor("token_embedding"))
      .to(DeviceType::CUDA);
}

static void require_cuda_f16(const Tensor& tensor, const std::string& name,
                             const std::vector<int64_t>& shape) {
  if (tensor.device() != DeviceType::CUDA || tensor.dtype() != DType::F16 ||
      !tensor.is_contiguous() || tensor.shape() != shape) {
    fail(name + " expected CUDA/F16/contiguous shape [" +
         std::to_string(shape[0]) + "," + std::to_string(shape[1]) + "]");
  }
}

static float max_abs_error(const Tensor& a, const Tensor& b) {
  if (a.numel() != b.numel() || a.shape() != b.shape()) fail("shape mismatch");
  float result = 0.0f;
  for (size_t i = 0; i < a.numel(); ++i)
    result = std::max(result, std::fabs(a.get_f32(i) - b.get_f32(i)));
  return result;
}

static void require_finite(const Tensor& tensor, const std::string& name) {
  Tensor cpu = tensor.to(DeviceType::CPU);
  float lo = std::numeric_limits<float>::infinity();
  float hi = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < cpu.numel(); ++i) {
    float value = cpu.get_f32(i);
    if (!std::isfinite(value)) fail(name + " contains NaN/Inf");
    lo = std::min(lo, value);
    hi = std::max(hi, value);
  }
  std::cout << name << " shape=" << cpu.shape()[0] << "x" << cpu.shape()[1]
            << " dtype=F16 device=CUDA min=" << lo << " max=" << hi
            << " finite=true\n";
}

static void expect_throw(const std::string& name,
                         const std::function<void()>& function) {
  try {
    function();
  } catch (const std::exception& error) {
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
    ModelPackage package(package_root);
    Qwen3CudaModel model(package_root);
    if (model.num_layers() != 28) fail("num_layers actual=" +
                                       std::to_string(model.num_layers()) +
                                       " expected=28");
    size_t expected_bytes = 0;
    for (const auto& info : package.tensors()) {
      size_t elements = 1;
      for (int64_t dimension : info.shape) elements *= size_t(dimension);
      expected_bytes += elements * 2;
    }
    if (model.resident_weight_bytes() != expected_bytes)
      fail("resident_weight_bytes actual=" +
           std::to_string(model.resident_weight_bytes()) + " expected=" +
           std::to_string(expected_bytes));
    std::cout << "resident_weight_bytes=" << model.resident_weight_bytes()
              << " tied_embedding_storage=single\n";

    const std::vector<int32_t> ids{0, 1, 2, 3};
    const std::vector<int32_t> positions{0, 1, 2, 3};
    const float eps = std::stof(package.config("rms_norm_eps"));
    const float theta = std::stof(package.config("rope_theta"));

    Tensor layer_runner = model.prefill_hidden_layers(ids, positions, 1);
    require_cuda_f16(layer_runner, "layer0 runner", {4, 1024});
    Tensor hidden = cuda_embedding_lookup(load_embedding_f16(package), ids);
    auto layer0 = load_layer0_f16(package);
    Tensor direct = qwen3_decoder_layer_cuda_trace_fp16(
                        hidden, positions, layer0, eps, theta)
                        .layer_output;
    require_cuda_f16(direct, "layer0 direct", {4, 1024});
    float layer_error = max_abs_error(layer_runner.to(DeviceType::CPU),
                                      direct.to(DeviceType::CPU));
    std::cout << "layer0_runner_vs_direct max_abs_error=" << layer_error
              << " tolerance=1e-6\n";
    if (layer_error > 1e-6f) fail("layer0 runner mismatch");

    Tensor all_layers = model.prefill_hidden_layers(ids, positions, 28);
    Tensor final_hidden = model.prefill_final_hidden(ids, positions);
    Tensor final_hidden_again = model.prefill_final_hidden(ids, positions);
    require_cuda_f16(all_layers, "28-layer hidden", {4, 1024});
    require_cuda_f16(final_hidden, "final hidden", {4, 1024});
    require_cuda_f16(final_hidden_again, "final hidden repeat", {4, 1024});
    require_finite(all_layers, "28-layer hidden");
    require_finite(final_hidden, "final hidden");
    float repeat_error = max_abs_error(final_hidden.to(DeviceType::CPU),
                                       final_hidden_again.to(DeviceType::CPU));
    std::cout << "final_hidden_repeat max_abs_error=" << repeat_error
              << " tolerance=1e-6\n";
    if (repeat_error > 1e-6f) fail("repeated final hidden mismatch");

    expect_throw("empty token_ids", [&] {
      model.prefill_hidden_layers({}, {}, 1);
    });
    expect_throw("negative token id", [&] {
      model.prefill_hidden_layers({-1, 1, 2, 3}, positions, 1);
    });
    expect_throw("token id out of range", [&] {
      model.prefill_hidden_layers({0, 1, 2, 151936}, positions, 1);
    });
    expect_throw("position length mismatch", [&] {
      model.prefill_hidden_layers(ids, {0, 1}, 1);
    });
    expect_throw("unsupported seq_len", [&] {
      model.prefill_hidden_layers({0, 1, 2}, {0, 1, 2}, 1);
    });
    expect_throw("layer_count zero", [&] {
      model.prefill_hidden_layers(ids, positions, 0);
    });
    expect_throw("layer_count too large", [&] {
      model.prefill_hidden_layers(ids, positions, 29);
    });
    expect_throw("missing package directory", [&] {
      Qwen3CudaModel missing("artifacts/phase15/does-not-exist");
    });

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_cuda_model passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
