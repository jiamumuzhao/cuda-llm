#include "llm/qwen3_chat_template.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace llm {
namespace {

using json = nlohmann::json;

[[noreturn]] void invalid(const std::string& message) {
  throw std::invalid_argument("Qwen3ChatTemplate: " + message);
}

bool valid_utf8(const std::string& value) {
  for (size_t i = 0; i < value.size();) {
    const unsigned char c = static_cast<unsigned char>(value[i]);
    size_t width = 0;
    if (c <= 0x7f) width = 1;
    else if (c >= 0xc2 && c <= 0xdf) width = 2;
    else if (c >= 0xe0 && c <= 0xef) width = 3;
    else if (c >= 0xf0 && c <= 0xf4) width = 4;
    else return false;
    if (i + width > value.size()) return false;
    for (size_t j = 1; j < width; ++j)
      if ((static_cast<unsigned char>(value[i + j]) & 0xc0) != 0x80) return false;
    if (width == 3) {
      const unsigned char b1 = static_cast<unsigned char>(value[i + 1]);
      if ((c == 0xe0 && b1 < 0xa0) || (c == 0xed && b1 >= 0xa0)) return false;
    }
    if (width == 4) {
      const unsigned char b1 = static_cast<unsigned char>(value[i + 1]);
      if ((c == 0xf0 && b1 < 0x90) || (c == 0xf4 && b1 >= 0x90)) return false;
    }
    i += width;
  }
  return true;
}

std::vector<ChatMessage> validate(const json& root) {
  if (!root.is_array()) invalid("messages top level must be a JSON array");
  if (root.empty()) invalid("messages must be non-empty");
  std::vector<ChatMessage> messages;
  messages.reserve(root.size());
  for (size_t i = 0; i < root.size(); ++i) {
    const json& item = root[i];
    if (!item.is_object()) invalid("messages[" + std::to_string(i) + "] must be a JSON object");
    if (!item.contains("role") || !item["role"].is_string())
      invalid("messages[" + std::to_string(i) + "].role must be a string");
    if (!item.contains("content") || !item["content"].is_string())
      invalid("messages[" + std::to_string(i) + "].content must be a string");
    ChatMessage message{item["role"].get<std::string>(), item["content"].get<std::string>()};
    if (message.role != "system" && message.role != "user" && message.role != "assistant")
      invalid("messages[" + std::to_string(i) + "].role is unsupported: " + message.role);
    if (!valid_utf8(message.role) || !valid_utf8(message.content))
      invalid("messages[" + std::to_string(i) + "] contains invalid UTF-8");
    if (message.role == "system" && i != 0) invalid("system message must be at index 0");
    messages.push_back(std::move(message));
  }
  size_t system_count = 0;
  for (const auto& message : messages) if (message.role == "system") ++system_count;
  if (system_count > 1) invalid("messages may contain at most one system message");
  if (messages.back().role != "user") invalid("messages must end with a user message");
  return messages;
}

}  // namespace

std::vector<ChatMessage> Qwen3ChatTemplate::parse_messages_json(const std::string& json_text) {
  if (json_text.empty()) invalid("messages JSON is empty");
  try {
    return validate(json::parse(json_text));
  } catch (const json::parse_error& error) {
    throw std::invalid_argument(std::string("Qwen3ChatTemplate: invalid messages JSON: ") + error.what());
  } catch (const json::type_error& error) {
    throw std::invalid_argument(std::string("Qwen3ChatTemplate: invalid messages JSON types: ") + error.what());
  }
}

std::vector<ChatMessage> Qwen3ChatTemplate::load_messages_file(const std::string& path) {
  if (path.empty()) invalid("messages file path is empty");
  std::ifstream input(std::filesystem::path(path), std::ios::binary);
  if (!input) throw std::runtime_error("Qwen3ChatTemplate: cannot open messages file '" + path + "'");
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (!input.good() && !input.eof()) throw std::runtime_error("Qwen3ChatTemplate: failed reading messages file '" + path + "'");
  try { return parse_messages_json(buffer.str()); }
  catch (const std::exception& error) { throw std::invalid_argument("Qwen3ChatTemplate: messages file '" + path + "': " + error.what()); }
}

std::string Qwen3ChatTemplate::render_generation_prompt(const std::vector<ChatMessage>& messages) {
  if (messages.empty()) invalid("messages must be non-empty");
  std::vector<ChatMessage> checked;
  checked.reserve(messages.size());
  for (size_t i = 0; i < messages.size(); ++i) {
    const auto& message = messages[i];
    if (message.role != "system" && message.role != "user" && message.role != "assistant")
      invalid("messages[" + std::to_string(i) + "].role is unsupported: " + message.role);
    if (!valid_utf8(message.role) || !valid_utf8(message.content)) invalid("messages contains invalid UTF-8");
    if (message.role == "system" && i != 0) invalid("system message must be at index 0");
    checked.push_back(message);
  }
  size_t systems = 0; for (const auto& message : checked) if (message.role == "system") ++systems;
  if (systems > 1) invalid("messages may contain at most one system message");
  if (checked.back().role != "user") invalid("messages must end with a user message");
  std::string rendered;
  for (const auto& message : checked) {
    rendered += "<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>\n";
  }
  rendered += "<|im_start|>assistant\n";
  return rendered;
}

}  // namespace llm
