#pragma once

#include "decoder_layer_weights.h"
#include "decoder_layer_executor.h"
#include "decoder_model_spec.h"
#include "qwen3_decode_workspace.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace llm {

class DecoderModelAdapter {
 public:
  virtual ~DecoderModelAdapter() = default;
  virtual DecoderModelSpec spec() const = 0;
  virtual Tensor load_token_embedding() const = 0;
  virtual Tensor load_final_norm() const = 0;
  virtual DecoderLayerWeights load_layer(std::size_t index) const = 0;
  virtual std::unique_ptr<DecoderLayerExecutor> create_layer_executor() const = 0;
};

class DecoderRuntime {
 public:
  explicit DecoderRuntime(std::unique_ptr<DecoderModelAdapter> adapter);
  virtual ~DecoderRuntime() = default;

  const DecoderModelSpec& model_spec() const noexcept { return model_spec_; }
  std::size_t num_layers() const noexcept { return layers_.size(); }
  std::size_t resident_weight_bytes() const noexcept { return resident_weight_bytes_; }
  std::size_t decode_workspace_bytes() const {
    return decode_workspace_ ? decode_workspace_->resident_bytes() : 0;
  }
  std::size_t paged_decode_metadata_workspace_bytes() const {
    return paged_decode_metadata_workspace_ ? paged_decode_metadata_workspace_->resident_bytes() : 0;
  }
  const DecoderLayerExecutor& layer_executor() const { return *layer_executor_; }

 protected:
  Tensor token_embedding_;
  Tensor final_norm_;
  std::vector<DecoderLayerWeights> layers_;
  DecoderModelSpec model_spec_{};
  std::size_t resident_weight_bytes_ = 0;
  std::unique_ptr<DecodeWorkspace> decode_workspace_;
  mutable std::unique_ptr<PagedDecodeMetadataWorkspace> paged_decode_metadata_workspace_;
  std::unique_ptr<DecoderLayerExecutor> layer_executor_;

 private:
  void account_weight_bytes();
};

}  // namespace llm
