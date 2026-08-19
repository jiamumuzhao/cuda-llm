#include "llm/qwen3_tokenizer.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
namespace {
void fail(const std::string& s) { throw std::runtime_error("test_qwen3_tokenizer: " + s); }
void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); } catch (const std::exception& e) { std::cout << "rejected " << name << ": " << e.what() << "\n"; return; }
  fail(name + " was accepted");
}
std::string unescape(std::string s) {
  std::string out;
  for (size_t i=0;i<s.size();++i) {
    if (s[i] != '\\') { out += s[i]; continue; }
    if (++i >= s.size()) fail("invalid reference escape");
    if (s[i]=='n') out+='\n'; else if(s[i]=='t') out+='\t'; else if(s[i]=='r') out+='\r';
    else if(s[i]=='"') out+='"'; else if(s[i]=='\\') out+='\\'; else fail("unsupported reference escape");
  }
  return out;
}
std::string field(const std::string& object, const std::string& key) {
  const std::regex re("\\\"" + key + "\\\"\\s*:\\s*\\\"((?:\\\\.|[^\\\"\\\\])*)\\\"");
  std::smatch m; if (!std::regex_search(object,m,re)) fail("missing reference field " + key); return unescape(m[1].str());
}
std::vector<int32_t> ids_field(const std::string& object) {
  const std::string key="\"input_ids\""; size_t p=object.find(key); if(p==std::string::npos) fail("missing input_ids");
  p=object.find('[',p); size_t end=object.find(']',p); if(p==std::string::npos||end==std::string::npos) fail("invalid input_ids");
  std::vector<int32_t> ids; size_t i=p+1;
  while(i<end) { while(i<end && (object[i]==' '||object[i]==','||object[i]=='\n')) ++i; if(i>=end) break; size_t j=i; while(j<end && object[j]>='0'&&object[j]<='9') ++j; if(j==i) fail("invalid input id"); ids.push_back(std::stoi(object.substr(i,j-i))); i=j; }
  return ids;
}
int32_t integer_field(const std::string& json, const std::string& key) {
  const std::regex re("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+)"); std::smatch m;
  if (!std::regex_search(json, m, re)) fail("missing integer field " + key); return std::stoi(m[1].str());
}
}

int main() {
  try {
    std::ifstream in("artifacts/phase28/qwen3_tokenizer_reference/reference.json");
    if (!in) fail("reference.json is missing");
    std::string json((std::istreambuf_iterator<char>(in)), {});
    const std::string tokenizer_path = field(json, "tokenizer_file");
    Qwen3Tokenizer tokenizer(tokenizer_path);
    size_t samples = 0; size_t pos=json.find("\"samples\"");
    while ((pos=json.find("\"name\"",pos)) != std::string::npos) {
      const size_t begin=json.rfind('{',pos); size_t end=json.find("\n    }",pos);
      if(begin==std::string::npos || end==std::string::npos) break;
      const std::string object=json.substr(begin,end+6-begin);
      const auto expected_ids=ids_field(object); const auto text=field(object,"text"); const auto expected_decoded=field(object,"decoded");
      const auto actual_ids=tokenizer.encode(text); if(actual_ids!=expected_ids) fail("encode mismatch for " + text);
      const auto actual_decoded=tokenizer.decode(expected_ids); if(actual_decoded!=expected_decoded) fail("decode mismatch for " + text);
      if(tokenizer.encode(text)!=actual_ids || tokenizer.decode(actual_ids)!=actual_decoded) fail("repeatability mismatch");
      std::cout << "sample text=" << text << " tokens=" << actual_ids.size() << " decoded=" << actual_decoded << "\n";
      ++samples; pos=end+6;
    }
    if(samples != 4) fail("expected 4 reference samples");
    if (!tokenizer.eos_token_id().has_value() || *tokenizer.eos_token_id() != integer_field(json, "eos_token_id"))
      fail("EOS id mismatch");
    const auto empty = tokenizer.encode("");
    const auto empty_pos = json.find("\"empty_input_ids\"");
    if (empty_pos == std::string::npos) fail("reference missing empty_input_ids");
    const auto empty_end = json.find(']', empty_pos); const auto empty_begin = json.find('[', empty_pos);
    if (empty_begin == std::string::npos || empty_end == std::string::npos || empty_end != empty_begin + 1 || !empty.empty())
      fail("empty encode mismatch");
    if(tokenizer.decode({}) != "") fail("empty decode should return empty text");
    expect_throw("missing file", [] { Qwen3Tokenizer t("/no/such/tokenizer.json"); });
    const auto bad=std::filesystem::temp_directory_path()/"cuda_llm_bad_tokenizer.json";
    { std::ofstream out(bad); out << "not json"; }
    expect_throw("corrupt JSON", [&] { Qwen3Tokenizer t(bad.string()); });
    std::filesystem::remove(bad);
    expect_throw("empty path", [] { Qwen3Tokenizer t(""); });
    expect_throw("invalid decode id", [&] { tokenizer.decode({-1}); });
    std::cout << "eos_token_id=" << (tokenizer.eos_token_id().has_value() ? std::to_string(*tokenizer.eos_token_id()) : "none") << "\n";
    std::cout << "test_qwen3_tokenizer passed\n"; return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
