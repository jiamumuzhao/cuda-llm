#include "llm/qwen3_cuda_model.h"

#include <charconv>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void usage() {
  std::cout << "usage: cuda_llm_generate --model <package_dir> --ids <id,id,...> "
               "--max-new-tokens <N> --max-seq-len <N> [--eos-id <id>] "
               "[--temperature <float>] [--top-k <int>] [--top-p <float>] [--seed <uint64>] [--paged]\n";
}

static int64_t parse_integer(const std::string& text, const std::string& name) {
  if (text.empty()) throw std::invalid_argument(name + " must not be empty");
  int64_t value = 0;
  const char* first = text.data();
  const char* last = first + text.size();
  auto result = std::from_chars(first, last, value);
  if (result.ec != std::errc() || result.ptr != last)
    throw std::invalid_argument(name + " invalid integer: " + text);
  return value;
}

static float parse_float(const std::string& text, const std::string& name) {
  if (text.empty()) throw std::invalid_argument(name + " must not be empty");
  char* end = nullptr; errno = 0;
  const float value = std::strtof(text.c_str(), &end);
  if (errno == ERANGE || end != text.c_str() + text.size() || !std::isfinite(value))
    throw std::invalid_argument(name + " invalid finite float: " + text);
  return value;
}
static uint64_t parse_u64(const std::string& text, const std::string& name) {
  if (text.empty() || text[0] == '-') throw std::invalid_argument(name + " invalid uint64");
  uint64_t value = 0; auto r = std::from_chars(text.data(), text.data()+text.size(), value);
  if (r.ec != std::errc() || r.ptr != text.data()+text.size()) throw std::invalid_argument(name + " invalid uint64");
  return value;
}

static std::vector<int32_t> parse_ids(const std::string& text) {
  if (text.empty()) throw std::invalid_argument("--ids must not be empty");
  std::vector<int32_t> ids;
  size_t begin = 0;
  while (begin <= text.size()) {
    size_t end = text.find(',', begin);
    std::string field = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    int64_t value = parse_integer(field, "--ids");
    if (value < 0 || value > 2147483647) throw std::invalid_argument("--ids value out of int32 range");
    ids.push_back(static_cast<int32_t>(value));
    if (end == std::string::npos) break;
    begin = end + 1;
    if (begin == text.size()) throw std::invalid_argument("--ids contains an empty field");
  }
  if (ids.empty()) throw std::invalid_argument("--ids must not be empty");
  return ids;
}

int main(int argc, char** argv) {
  try {
    std::string model;
    std::vector<int32_t> ids;
    size_t max_new_tokens = 0, max_seq_len = 0;
    bool have_model = false, have_ids = false, have_new = false, have_seq = false;
    bool have_temp = false, have_top_k = false, have_top_p = false, have_seed = false;
    bool paged = false;
    std::optional<int32_t> eos;
    SamplingConfig sampling;
    bool sampled = false;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help") { usage(); return 0; }
      if (option == "--paged") {
        if (paged) throw std::invalid_argument("duplicate --paged");
        paged = true;
        continue;
      }
      if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)
        throw std::invalid_argument("missing value for " + option);
      const std::string value = argv[++i];
      if (option == "--model") {
        if (have_model) throw std::invalid_argument("duplicate --model");
        model = value; have_model = true;
      } else if (option == "--ids") {
        if (have_ids) throw std::invalid_argument("duplicate --ids");
        ids = parse_ids(value); have_ids = true;
      } else if (option == "--max-new-tokens") {
        if (have_new) throw std::invalid_argument("duplicate --max-new-tokens");
        int64_t parsed = parse_integer(value, option);
        if (parsed < 0) throw std::invalid_argument(option + " must be >= 0");
        max_new_tokens = static_cast<size_t>(parsed); have_new = true;
      } else if (option == "--max-seq-len") {
        if (have_seq) throw std::invalid_argument("duplicate --max-seq-len");
        int64_t parsed = parse_integer(value, option);
        if (parsed <= 0) throw std::invalid_argument(option + " must be > 0");
        max_seq_len = static_cast<size_t>(parsed); have_seq = true;
      } else if (option == "--eos-id") {
        if (eos) throw std::invalid_argument("duplicate --eos-id");
        int64_t parsed = parse_integer(value, option);
        if (parsed < 0 || parsed >= 151936) throw std::invalid_argument(option + " out of range");
        eos = static_cast<int32_t>(parsed);
      } else if (option == "--temperature") {
        if (have_temp) throw std::invalid_argument("duplicate --temperature");
        sampling.temperature = parse_float(value, option); have_temp = sampled = true;
      } else if (option == "--top-k") {
        if (have_top_k) throw std::invalid_argument("duplicate --top-k");
        int64_t parsed = parse_integer(value, option);
        if (parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) throw std::invalid_argument("--top-k out of range");
        sampling.top_k = static_cast<int32_t>(parsed); have_top_k = sampled = true;
      } else if (option == "--top-p") {
        if (have_top_p) throw std::invalid_argument("duplicate --top-p");
        sampling.top_p = parse_float(value, option); have_top_p = sampled = true;
      } else if (option == "--seed") {
        if (have_seed) throw std::invalid_argument("duplicate --seed");
        sampling.seed = parse_u64(value, option); have_seed = sampled = true;
      } else {
        throw std::invalid_argument("unsupported argument: " + option);
      }
    }
    if (!have_model || !have_ids || !have_new || !have_seq)
      throw std::invalid_argument("--model, --ids, --max-new-tokens, and --max-seq-len are required");
    if (sampling.temperature < 0.0f) throw std::invalid_argument("--temperature must be >= 0");
    if ((have_top_p && (sampling.top_p <= 0.0f || sampling.top_p > 1.0f)) ||
        !std::isfinite(sampling.top_p)) throw std::invalid_argument("--top-p must be finite in (0,1]");
    if (max_seq_len < ids.size()) throw std::invalid_argument("--max-seq-len is smaller than prompt length");
    Qwen3CudaModel model_object(model);
    if (paged && max_seq_len > 512)
      throw std::invalid_argument("--paged supports --max-seq-len <= 512");
    const auto started = std::chrono::steady_clock::now();
    GreedyGenerationResult result;
    if (paged) {
      const size_t blocks = (max_seq_len + 15) / 16 + 8;
      PagedKvCachePool pool(
          PagedKvCachePoolConfig{blocks, 28, 8, 16, 128, DType::F16});
      result = sampled && sampling.temperature != 0.0f
          ? model_object.generate_sampled_paged(
                pool, ids, max_new_tokens, eos, max_seq_len, sampling)
          : model_object.generate_greedy_paged(
                pool, ids, max_new_tokens, eos, max_seq_len);
    } else {
      result = sampled && sampling.temperature != 0.0f
          ? model_object.generate_sampled(ids, max_new_tokens, eos, max_seq_len, sampling)
          : model_object.generate_greedy(ids, max_new_tokens, eos, max_seq_len);
    }
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    auto print_ids = [](const std::vector<int32_t>& values) {
      std::cout << "[";
      for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ","; std::cout << values[i]; }
      std::cout << "]";
    };
    std::cout << "prompt_ids="; print_ids(ids); std::cout << "\n";
    std::cout << "generated_ids="; print_ids(result.generated_ids); std::cout << "\n";
    std::cout << "stop_reason=" << result.stop_reason << "\n";
    std::cout << "final_cache_length=" << result.final_cache_length << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "cuda_llm_generate: " << error.what() << "\n";
    usage();
    return 2;
  }
}
