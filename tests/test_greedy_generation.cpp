#include "llm/cuda_check.h"
#include "llm/ops_cuda.h"
#include "llm/qwen3_cuda_model.h"

#include <cuda_runtime.h>

#include <iostream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& message) { throw std::runtime_error("test_greedy_generation: " + message); }
static void print_ids(const std::string& name, const std::vector<int32_t>& ids) {
  std::cout << name << "=[";
  for (size_t i = 0; i < ids.size(); ++i) { if (i) std::cout << ","; std::cout << ids[i]; }
  std::cout << "]";
}
static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); }
  catch (const std::exception& error) { std::cout << "rejected " << name << ": " << error.what() << "\n"; return; }
  fail(name + " was accepted");
}

int main() {
  try {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability=" << prop.major << "." << prop.minor << "\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    const std::vector<int32_t> prompt{1, 17, 257, 4097};
    Qwen3KvCache manual_cache(16);
    Tensor logits = model.prefill_logits_with_cache(prompt, manual_cache);
    std::vector<int32_t> manual;
    for (size_t i = 0; i < 3; ++i) {
      int32_t token = cuda_argmax_last_row(logits);
      manual.push_back(token);
      if (i + 1 < 3) logits = model.decode_logits(token, manual_cache);
    }
    auto generated = model.generate_greedy(prompt, 3, std::nullopt, 16);
    print_ids("manual_generated_ids", manual); std::cout << "\n";
    print_ids("generated_ids", generated.generated_ids);
    std::cout << " stop_reason=" << generated.stop_reason << " final_cache_length=" << generated.final_cache_length << "\n";
    if (generated.generated_ids != manual || generated.stop_reason != "max_new_tokens" || generated.final_cache_length != 6)
      fail("generate_greedy differs from manual GPU baseline");

    int32_t first_token = manual.front();
    auto eos = model.generate_greedy(prompt, 8, first_token, 16);
    print_ids("eos_generated_ids", eos.generated_ids); std::cout << " stop_reason=" << eos.stop_reason
                                                            << " final_cache_length=" << eos.final_cache_length << "\n";
    if (eos.generated_ids != std::vector<int32_t>{first_token} || eos.stop_reason != "eos") fail("EOS stopping mismatch");
    auto zero = model.generate_greedy(prompt, 0, std::nullopt, 16);
    if (!zero.generated_ids.empty() || zero.stop_reason != "max_new_tokens" || zero.final_cache_length != 0) fail("zero-token generation mismatch");
    auto capacity = model.generate_greedy(prompt, 2, std::nullopt, 4);
    print_ids("capacity_generated_ids", capacity.generated_ids); std::cout << " stop_reason=" << capacity.stop_reason
                                                                  << " final_cache_length=" << capacity.final_cache_length << "\n";
    if (capacity.generated_ids.size() != 1 || capacity.stop_reason != "cache_capacity" || capacity.final_cache_length != 4)
      fail("cache capacity stopping mismatch");
    auto repeat = model.generate_greedy(prompt, 3, std::nullopt, 16);
    if (repeat.generated_ids != generated.generated_ids) fail("generation is not repeatable");

    expect_throw("invalid eos", [&] { model.generate_greedy(prompt, 1, 151936, 16); });
    expect_throw("invalid max_seq_len", [&] { model.generate_greedy(prompt, 1, std::nullopt, 3); });
    expect_throw("out of range prompt", [&] { model.generate_greedy({1, 151936}, 1, std::nullopt, 16); });
    expect_throw("empty prompt", [&] { model.generate_greedy({}, 1, std::nullopt, 16); });
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_greedy_generation passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << "\n"; return 1; }
}
