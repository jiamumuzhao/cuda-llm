#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_chat_template.h"
#include "llm/qwen3_tokenizer.h"

#include <charconv>
#include <climits>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

namespace {
struct Args { std::string model, tokenizer, prompt, messages_file; size_t max_new = 0, max_seq = 0; std::optional<int32_t> eos; SamplingConfig sampling; bool sampled=false; };
[[noreturn]] void usage_error(const std::string& s) { throw std::invalid_argument("cuda_llm_chat: " + s); }
int64_t integer(const std::string& s, const char* name) {
  if (s.empty() || s[0] == '-') usage_error(std::string(name) + " must be a non-negative integer");
  int64_t v = 0; auto r = std::from_chars(s.data(), s.data() + s.size(), v);
  if (r.ec != std::errc() || r.ptr != s.data() + s.size()) usage_error(std::string(name) + " is invalid");
  return v;
}
std::string value(int& i, int argc, char** argv, const char* name) {
  if (++i >= argc || std::string(argv[i]).rfind("--", 0) == 0) usage_error(std::string(name) + " is missing");
  return argv[i];
}
float real(const std::string& s, const char* name) {
  if (s.empty()) usage_error(std::string(name) + " is invalid");
  char* end=nullptr; errno=0; float v=std::strtof(s.c_str(), &end);
  if (errno==ERANGE || end != s.c_str()+s.size() || !std::isfinite(v)) usage_error(std::string(name)+" must be finite");
  return v;
}
uint64_t u64(const std::string& s, const char* name) {
  if (s.empty() || s[0]=='-') usage_error(std::string(name)+" is invalid");
  uint64_t v=0; auto r=std::from_chars(s.data(),s.data()+s.size(),v);
  if (r.ec != std::errc() || r.ptr != s.data()+s.size()) usage_error(std::string(name)+" is invalid");
  return v;
}
Args parse(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::cout << "usage: cuda_llm_chat --model <package_dir> --tokenizer <tokenizer.json> "
                 "(--prompt <UTF-8 text> | --messages-file <messages.json>) "
                 "--max-new-tokens <N> --max-seq-len <N> [--eos-id <id>] [--temperature <float>] [--top-k <int>] [--top-p <float>] [--seed <uint64>]\n";
    std::exit(0);
  }
  Args a; bool model=false, tokenizer=false, prompt=false, messages_file=false, max_new=false, max_seq=false, eos=false, temperature=false, top_k=false, top_p=false, seed=false;
  for (int i=1; i<argc; ++i) {
    const std::string key = argv[i];
    if (key == "--model") { if (model) usage_error("duplicate --model"); a.model=value(i,argc,argv,"--model"); model=true; }
    else if (key == "--tokenizer") { if (tokenizer) usage_error("duplicate --tokenizer"); a.tokenizer=value(i,argc,argv,"--tokenizer"); tokenizer=true; }
    else if (key == "--prompt") { if (prompt) usage_error("duplicate --prompt"); a.prompt=value(i,argc,argv,"--prompt"); if (a.prompt.empty()) usage_error("--prompt must not be empty"); prompt=true; }
    else if (key == "--messages-file") { if (messages_file) usage_error("duplicate --messages-file"); a.messages_file=value(i,argc,argv,"--messages-file"); messages_file=true; }
    else if (key == "--max-new-tokens") { if (max_new) usage_error("duplicate --max-new-tokens"); auto n=integer(value(i,argc,argv,"--max-new-tokens"),"--max-new-tokens"); a.max_new=static_cast<size_t>(n); max_new=true; }
    else if (key == "--max-seq-len") { if (max_seq) usage_error("duplicate --max-seq-len"); auto n=integer(value(i,argc,argv,"--max-seq-len"),"--max-seq-len"); if(n==0) usage_error("--max-seq-len must be > 0"); a.max_seq=static_cast<size_t>(n); max_seq=true; }
    else if (key == "--eos-id") { if (eos) usage_error("duplicate --eos-id"); auto n=integer(value(i,argc,argv,"--eos-id"),"--eos-id"); if(n > INT32_MAX) usage_error("--eos-id exceeds int32_t"); a.eos=static_cast<int32_t>(n); eos=true; }
    else if (key == "--temperature") { if (temperature) usage_error("duplicate --temperature"); a.sampling.temperature=real(value(i,argc,argv,"--temperature"),"--temperature"); temperature=a.sampled=true; }
    else if (key == "--top-k") { if (top_k) usage_error("duplicate --top-k"); auto n=integer(value(i,argc,argv,"--top-k"),"--top-k"); if(n>INT32_MAX) usage_error("--top-k exceeds int32_t"); a.sampling.top_k=static_cast<int32_t>(n); top_k=a.sampled=true; }
    else if (key == "--top-p") { if (top_p) usage_error("duplicate --top-p"); a.sampling.top_p=real(value(i,argc,argv,"--top-p"),"--top-p"); top_p=a.sampled=true; }
    else if (key == "--seed") { if (seed) usage_error("duplicate --seed"); a.sampling.seed=u64(value(i,argc,argv,"--seed"),"--seed"); seed=a.sampled=true; }
    else usage_error("unknown argument " + key);
  }
  if (!model || !tokenizer || (!prompt && !messages_file) || (prompt && messages_file) || !max_new || !max_seq)
    usage_error("exactly one of --prompt and --messages-file is required, plus all other required arguments");
  if (a.sampling.temperature < 0.0f) usage_error("--temperature must be >= 0");
  if (top_p && (a.sampling.top_p <= 0.0f || a.sampling.top_p > 1.0f)) usage_error("--top-p must be in (0,1]");
  return a;
}
void print_ids(const char* name, const std::vector<int32_t>& ids) {
  std::cout << name << "=["; for (size_t i=0;i<ids.size();++i) { if(i) std::cout << ','; std::cout << ids[i]; } std::cout << "]\n";
}
}

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);
    Qwen3Tokenizer tokenizer(a.tokenizer);
    std::string prompt_text = a.prompt;
    if (!a.messages_file.empty()) {
      const auto messages = Qwen3ChatTemplate::load_messages_file(a.messages_file);
      prompt_text = Qwen3ChatTemplate::render_generation_prompt(messages);
    }
    const auto prompt_ids = tokenizer.encode(prompt_text);
    if (prompt_ids.empty()) throw std::invalid_argument("cuda_llm_chat: prompt encoded to zero tokens");
    if (a.max_seq < prompt_ids.size()) throw std::invalid_argument("cuda_llm_chat: max_seq_len is smaller than encoded prompt length");
    if (a.eos.has_value() && (*a.eos < 0 || *a.eos >= tokenizer.vocab_size()))
      throw std::invalid_argument("cuda_llm_chat: --eos-id outside tokenizer vocabulary [0," + std::to_string(tokenizer.vocab_size()) + ")");
    const auto eos = a.eos.has_value() ? a.eos : tokenizer.eos_token_id();
    Qwen3CudaModel model(a.model);
    const auto result = a.sampled && a.sampling.temperature != 0.0f
        ? model.generate_sampled(prompt_ids, a.max_new, eos, a.max_seq, a.sampling)
        : model.generate_greedy(prompt_ids, a.max_new, eos, a.max_seq);
    print_ids("prompt_ids", prompt_ids);
    print_ids("generated_ids", result.generated_ids);
    std::cout << "generated_text=" << tokenizer.decode(result.generated_ids) << "\n";
    std::cout << "stop_reason=" << result.stop_reason << "\n";
    std::cout << "final_cache_length=" << result.final_cache_length << "\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
