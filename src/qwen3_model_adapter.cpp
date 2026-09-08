#include "llm/qwen3_model_adapter.h"
#include "llm/qwen3_layer_executor.h"

#include "llm/model_package.h"

#include <cmath>
#include <stdexcept>

namespace llm {
namespace {

[[noreturn]] void adapter_error(const std::string& message) {
  throw std::runtime_error("Qwen3ModelAdapter: " + message);
}

Tensor f32_to_f16_cpu(const Tensor& input, const std::string& name) {
  if (input.device() != DeviceType::CPU || input.dtype() != DType::F32 || !input.is_contiguous())
    adapter_error(name + " must be CPU/F32/contiguous at load boundary");
  Tensor output(DType::F16, input.shape(), DeviceType::CPU);
  for (std::size_t i = 0; i < input.numel(); ++i) output.set_f32(i, input.get_f32(i));
  return output;
}

Tensor load_f16_cuda(const ModelPackage& package, const std::string& name,
                     const std::vector<int64_t>& expected_shape) {
  Tensor cpu = package.load_tensor(name);
  if (cpu.shape() != expected_shape) adapter_error(name + " shape mismatch");
  return f32_to_f16_cpu(cpu, name).to(DeviceType::CUDA);
}

class Qwen3ModelAdapter final : public DecoderModelAdapter {
 public:
  explicit Qwen3ModelAdapter(const std::filesystem::path& root) : package_(root) {
    if (package_.config("export_dtype") != "f32") adapter_error("config export_dtype must be f32");
    if (package_.config("tie_word_embeddings") != "true") adapter_error("config tie_word_embeddings must be true");
    spec_.attention = AttentionConfig{16, 8, 128, 16, 512, true};
    spec_.hidden_size = 1024;
    spec_.intermediate_size = 3072;
    spec_.vocab_size = 151936;
    spec_.num_layers = 28;
    spec_.rms_norm_eps = std::stof(package_.config("rms_norm_eps"));
    spec_.rope_theta = std::stof(package_.config("rope_theta"));
    spec_.tie_word_embeddings = true;
    if (!std::isfinite(spec_.rms_norm_eps) || spec_.rms_norm_eps <= 0.0f || !std::isfinite(spec_.rope_theta) || spec_.rope_theta <= 0.0f)
      adapter_error("invalid normalization or RoPE configuration");
  }
  DecoderModelSpec spec() const override { return spec_; }
  std::unique_ptr<DecoderLayerExecutor> create_layer_executor() const override {
    return std::make_unique<Qwen3LayerExecutor>();
  }
  Tensor load_token_embedding() const override {
    return load_f16_cuda(package_, "token_embedding", {static_cast<int64_t>(spec_.vocab_size), static_cast<int64_t>(spec_.hidden_size)});
  }
  Tensor load_final_norm() const override {
    return load_f16_cuda(package_, "final_norm", {static_cast<int64_t>(spec_.hidden_size)});
  }
  DecoderLayerWeights load_layer(std::size_t index) const override {
    const std::string prefix = "layers." + std::to_string(index) + ".";
    const auto hidden = static_cast<int64_t>(spec_.hidden_size);
    const auto q_dim = static_cast<int64_t>(spec_.attention.num_q_heads * spec_.attention.head_dim);
    const auto kv_dim = static_cast<int64_t>(spec_.attention.num_kv_heads * spec_.attention.head_dim);
    const auto head_dim = static_cast<int64_t>(spec_.attention.head_dim);
    const auto intermediate = static_cast<int64_t>(spec_.intermediate_size);
    DecoderLayerWeights result{
        load_f16_cuda(package_, prefix + "input_norm", {hidden}),
        load_f16_cuda(package_, prefix + "q_proj", {q_dim, hidden}),
        load_f16_cuda(package_, prefix + "k_proj", {kv_dim, hidden}),
        load_f16_cuda(package_, prefix + "v_proj", {kv_dim, hidden}),
        load_f16_cuda(package_, prefix + "q_norm", {head_dim}),
        load_f16_cuda(package_, prefix + "k_norm", {head_dim}),
        load_f16_cuda(package_, prefix + "o_proj", {hidden, q_dim}),
        load_f16_cuda(package_, prefix + "post_attention_norm", {hidden}),
        load_f16_cuda(package_, prefix + "gate_proj", {intermediate, hidden}),
        load_f16_cuda(package_, prefix + "up_proj", {intermediate, hidden}),
        load_f16_cuda(package_, prefix + "down_proj", {hidden, intermediate}),
        spec_};
    return result;
  }
 private:
  ModelPackage package_;
  DecoderModelSpec spec_{};
};

}  // namespace

std::unique_ptr<DecoderModelAdapter> make_qwen3_model_adapter(const std::filesystem::path& package_root) {
  return std::make_unique<Qwen3ModelAdapter>(package_root);
}

}  // namespace llm
