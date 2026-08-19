#include "llm/qwen3_tokenizer.h"

#include <tokenizers_cpp.h>

#include <filesystem>
#include <cctype>
#include <climits>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace llm {
namespace {

std::string read_file(const std::filesystem::path& path, const char* what) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error(std::string("Qwen3Tokenizer: cannot open ") + what + " '" + path.string() + "'");
  std::ostringstream buffer; buffer << input.rdbuf();
  if (!input.good() && !input.eof()) throw std::runtime_error(std::string("Qwen3Tokenizer: failed reading ") + what + " '" + path.string() + "'");
  return buffer.str();
}

std::string json_unescape(const std::string& value) {
  std::string result;
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] != '\\') { result += value[i]; continue; }
    if (++i >= value.size()) throw std::invalid_argument("Qwen3Tokenizer: invalid JSON escape in eos_token");
    switch (value[i]) {
      case '"': result += '"'; break; case '\\': result += '\\'; break;
      case '/': result += '/'; break; case 'b': result += '\b'; break;
      case 'f': result += '\f'; break; case 'n': result += '\n'; break;
      case 'r': result += '\r'; break; case 't': result += '\t'; break;
      default: throw std::invalid_argument("Qwen3Tokenizer: unsupported JSON escape in eos_token");
    }
  }
  return result;
}

std::optional<std::string> configured_eos(const std::string& config) {
  const std::regex key("\\\"eos_token\\\"\\s*:");
  std::smatch match;
  if (!std::regex_search(config, match, key)) return std::nullopt;
  size_t p = static_cast<size_t>(match.position() + match.length());
  while (p < config.size() && std::isspace(static_cast<unsigned char>(config[p]))) ++p;
  if (p >= config.size() || config[p] != '"') throw std::invalid_argument("Qwen3Tokenizer: eos_token must be a JSON string");
  ++p; std::string raw;
  bool closed = false;
  for (; p < config.size(); ++p) {
    if (config[p] == '\\') { if (p + 1 >= config.size()) break; raw += config[p++]; raw += config[p]; continue; }
    if (config[p] == '"') { closed = true; break; }
    raw += config[p];
  }
  if (!closed) throw std::invalid_argument("Qwen3Tokenizer: unterminated eos_token JSON string");
  return json_unescape(raw);
}

}  // namespace

Qwen3Tokenizer::Qwen3Tokenizer(const std::string& path) {
  if (path.empty()) throw std::invalid_argument("Qwen3Tokenizer: tokenizer JSON path is empty");
  const std::filesystem::path tokenizer_path(path);
  const std::string blob = read_file(tokenizer_path, "tokenizer JSON");
  try {
    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
  } catch (const std::exception& error) {
    throw std::runtime_error("Qwen3Tokenizer: failed to parse tokenizer JSON '" + path + "': " + error.what());
  }
  if (!tokenizer_) throw std::runtime_error("Qwen3Tokenizer: tokenizer JSON produced a null tokenizer");
  const size_t size = tokenizer_->GetVocabSize();
  if (size == 0 || size > static_cast<size_t>(INT32_MAX)) throw std::runtime_error("Qwen3Tokenizer: invalid vocabulary size");
  vocab_size_ = static_cast<int32_t>(size);

  const auto config_path = tokenizer_path.parent_path() / "tokenizer_config.json";
  if (std::filesystem::exists(config_path)) {
    const auto eos_text = configured_eos(read_file(config_path, "tokenizer config"));
    if (eos_text.has_value()) {
      const auto ids = tokenizer_->Encode(*eos_text);
      if (ids.size() != 1) throw std::invalid_argument("Qwen3Tokenizer: configured eos_token must encode to exactly one token");
      if (ids[0] < 0 || ids[0] >= vocab_size_) throw std::invalid_argument("Qwen3Tokenizer: configured eos_token produced an invalid token id");
      eos_token_id_ = ids[0];
    }
  }
}

Qwen3Tokenizer::~Qwen3Tokenizer() = default;

std::vector<int32_t> Qwen3Tokenizer::encode(const std::string& text) const {
  if (!tokenizer_) throw std::runtime_error("Qwen3Tokenizer::encode: tokenizer is not initialized");
  return tokenizer_->Encode(text);
}

std::string Qwen3Tokenizer::decode(const std::vector<int32_t>& ids) const {
  if (!tokenizer_) throw std::runtime_error("Qwen3Tokenizer::decode: tokenizer is not initialized");
  for (const int32_t id : ids) if (id < 0 || id >= vocab_size_)
    throw std::out_of_range("Qwen3Tokenizer::decode: token id " + std::to_string(id) + " outside [0," + std::to_string(vocab_size_) + ")");
  return tokenizer_->Decode(ids);
}

std::optional<int32_t> Qwen3Tokenizer::eos_token_id() const { return eos_token_id_; }

}  // namespace llm
