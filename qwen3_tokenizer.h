#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llm {

// Read-only wrapper around the Hugging Face tokenizers runtime.  The tokenizer
// JSON is parsed once by the constructor and reused by encode/decode.
class Qwen3Tokenizer {
 public:
  explicit Qwen3Tokenizer(const std::string& tokenizer_json_path);
  ~Qwen3Tokenizer();

  Qwen3Tokenizer(const Qwen3Tokenizer&) = delete;
  Qwen3Tokenizer& operator=(const Qwen3Tokenizer&) = delete;
  Qwen3Tokenizer(Qwen3Tokenizer&&) = delete;
  Qwen3Tokenizer& operator=(Qwen3Tokenizer&&) = delete;

  std::vector<int32_t> encode(const std::string& text) const;
  std::string decode(const std::vector<int32_t>& ids) const;
  std::optional<int32_t> eos_token_id() const;

 private:
  void* tokenizer_ = nullptr;
  int32_t vocab_size_ = 0;
};

}  // namespace llm
