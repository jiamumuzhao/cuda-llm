#include "llm/qwen3_chat_template.h"
#include "llm/qwen3_tokenizer.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace llm;
using json = nlohmann::json;

static void fail(const std::string& message) { throw std::runtime_error("test_qwen3_chat_template: " + message); }
static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); } catch (const std::exception& error) {
    std::cout << "rejected " << name << ": " << error.what() << "\n"; return;
  }
  fail(name + " was accepted");
}

int main() {
  try {
    std::ifstream input("artifacts/phase29/qwen3_chat_template_reference/reference.json");
    if (!input) fail("reference.json is missing");
    json reference = json::parse(input);
    if (!reference.at("add_generation_prompt").get<bool>()) fail("reference generation-prompt flag is false");
    Qwen3Tokenizer tokenizer(reference.at("tokenizer_file").get<std::string>());
    const auto& cases = reference.at("cases");
    if (!cases.is_array() || cases.size() != 4) fail("expected four reference cases");
    for (const auto& record : cases) {
      std::vector<ChatMessage> messages;
      for (const auto& item : record.at("messages"))
        messages.push_back({item.at("role").get<std::string>(), item.at("content").get<std::string>()});
      const std::string actual_text = Qwen3ChatTemplate::render_generation_prompt(messages);
      const std::string expected_text = record.at("rendered_prompt").get<std::string>();
      if (actual_text != expected_text) fail("render mismatch for " + record.at("name").get<std::string>());
      const auto actual_ids = tokenizer.encode(actual_text);
      const auto expected_ids = record.at("apply_chat_template_ids").get<std::vector<int32_t>>();
      if (actual_ids != expected_ids) fail("token ID mismatch for " + record.at("name").get<std::string>());
      std::cout << "case=" << record.at("name").get<std::string>()
                << " rendered_tokens=" << actual_ids.size() << "\n";
    }
    std::cout << "template/reference alignment: passed\n";

    expect_throw("empty array", [] { Qwen3ChatTemplate::parse_messages_json("[]"); });
    expect_throw("corrupt JSON", [] { Qwen3ChatTemplate::parse_messages_json("[{]"); });
    expect_throw("top-level object", [] { Qwen3ChatTemplate::parse_messages_json("{}"); });
    expect_throw("missing role", [] { Qwen3ChatTemplate::parse_messages_json("[{\"content\":\"x\"}]"); });
    expect_throw("wrong field type", [] { Qwen3ChatTemplate::parse_messages_json("[{\"role\":1,\"content\":\"x\"}]"); });
    expect_throw("unknown role", [] { Qwen3ChatTemplate::parse_messages_json("[{\"role\":\"tool\",\"content\":\"x\"}]"); });
    expect_throw("system position", [] { Qwen3ChatTemplate::parse_messages_json("[{\"role\":\"user\",\"content\":\"x\"},{\"role\":\"system\",\"content\":\"s\"}]"); });
    expect_throw("multiple system", [] { Qwen3ChatTemplate::parse_messages_json("[{\"role\":\"system\",\"content\":\"a\"},{\"role\":\"system\",\"content\":\"b\"},{\"role\":\"user\",\"content\":\"x\"}]"); });
    expect_throw("non-user final", [] { Qwen3ChatTemplate::parse_messages_json("[{\"role\":\"user\",\"content\":\"x\"},{\"role\":\"assistant\",\"content\":\"a\"}]"); });
    expect_throw("invalid UTF-8", [] { Qwen3ChatTemplate::parse_messages_json(std::string("[{\"role\":\"user\",\"content\":\"") + char(0xff) + "\"}]"); });
    expect_throw("missing file", [] { Qwen3ChatTemplate::load_messages_file("/no/such/messages.json"); });
    expect_throw("empty path", [] { Qwen3ChatTemplate::load_messages_file(""); });
    std::cout << "message validation errors: passed\n";
    std::cout << "test_qwen3_chat_template passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << "\n"; return 1; }
}
