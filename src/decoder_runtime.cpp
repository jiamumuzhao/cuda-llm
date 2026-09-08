#include "llm/decoder_runtime.h"

#include <stdexcept>

namespace llm {

DecoderRuntime::DecoderRuntime(std::unique_ptr<DecoderModelAdapter> adapter) {
  if (!adapter) throw std::invalid_argument("DecoderRuntime: null model adapter");
  model_spec_ = adapter->spec();
  if (model_spec_.num_layers == 0 || model_spec_.hidden_size == 0 || model_spec_.vocab_size == 0)
    throw std::invalid_argument("DecoderRuntime: invalid model specification");
  token_embedding_ = adapter->load_token_embedding();
  final_norm_ = adapter->load_final_norm();
  layer_executor_ = adapter->create_layer_executor();
  if (!layer_executor_)
    throw std::invalid_argument("DecoderRuntime: adapter returned null layer executor");
  layers_.reserve(model_spec_.num_layers);
  for (std::size_t i = 0; i < model_spec_.num_layers; ++i) layers_.push_back(adapter->load_layer(i));
  account_weight_bytes();
  decode_workspace_ = std::make_unique<DecodeWorkspace>(model_spec_);
}

void DecoderRuntime::account_weight_bytes() {
  resident_weight_bytes_ = token_embedding_.nbytes() + final_norm_.nbytes();
  for (const auto& layer : layers_) {
    resident_weight_bytes_ += layer.input_norm.nbytes();
    resident_weight_bytes_ += layer.q_proj.nbytes() + layer.k_proj.nbytes();
    resident_weight_bytes_ += layer.v_proj.nbytes() + layer.q_norm.nbytes();
    resident_weight_bytes_ += layer.k_norm.nbytes() + layer.o_proj.nbytes();
    resident_weight_bytes_ += layer.post_attention_norm.nbytes();
    resident_weight_bytes_ += layer.gate_proj.nbytes() + layer.up_proj.nbytes();
    resident_weight_bytes_ += layer.down_proj.nbytes();
  }
}

}  // namespace llm
