#pragma once

#include <string>
#include <vector>

namespace llm {

struct ChatMessage {
  std::string role;
  std::string content;
};

class Qwen3ChatTemplate {
 public:
  static std::vector<ChatMessage> parse_messages_json(const std::string& json_text);
  static std::vector<ChatMessage> load_messages_file(const std::string& path);
  static std::string render_generation_prompt(const std::vector<ChatMessage>& messages);
};

}  // namespace llm
