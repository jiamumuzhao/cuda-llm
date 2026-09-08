#pragma once

#include "decoder_runtime.h"

#include <filesystem>
#include <memory>

namespace llm {

std::unique_ptr<DecoderModelAdapter> make_qwen3_model_adapter(const std::filesystem::path& package_root);

}  // namespace llm
