#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_padded_prefill.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("test_qwen3_paged_padded_prefill: " + message);
}

void expect(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

void expect_throw(const std::string& name, const std::function<void()>& function) {
  try {
    function();
  } catch (const std::exception& error) {
    std::cout << "rejected " << name << ": " << error.what() << "\n";
    return;
  }
  fail(name + " was accepted");
}

std::vector<int32_t> prompt(size_t count, int salt) {
  std::vector<int32_t> ids(count);
  for (size_t i = 0; i < count; ++i)
    ids[i] = 1 + (salt + static_cast<int>(i * 43)) % 150000;
  return ids;
}

PagedKvCachePoolConfig pool_config(size_t blocks) {
  return {blocks, 28, 8, 16, 128, DType::F16};
}

void compare(const Tensor& actual, const Tensor& expected, const std::string& name) {
  Tensor actual_cpu = actual.to(DeviceType::CPU);
  Tensor expected_cpu = expected.to(DeviceType::CPU);
  expect(actual_cpu.shape() == expected_cpu.shape(), name + " shape mismatch");
  float max_abs = 0.0f;
  for (size_t i = 0; i < actual_cpu.numel(); ++i)
    max_abs = std::max(max_abs, std::fabs(actual_cpu.get_f32(i) - expected_cpu.get_f32(i)));
  expect(max_abs <= 5e-3f, name + " max_abs_error=" + std::to_string(max_abs));
  std::cout << name << " max_abs_error=" << max_abs << "\n";
}

void compare_kv(const Qwen3PagedKvCache& paged, const Qwen3KvCache& contiguous,
                size_t length, const std::string& name) {
  for (size_t layer = 0; layer < 28; ++layer) {
    Tensor key = contiguous.key_cache(layer).to(DeviceType::CPU);
    Tensor value = contiguous.value_cache(layer).to(DeviceType::CPU);
    for (size_t token = 0; token < length; ++token) {
      std::vector<uint16_t> paged_key(1024), paged_value(1024);
      paged.copy_layer_kv_to_host(layer, token, paged_key.data(), paged_value.data(), 1024);
      for (size_t i = 0; i < 1024; ++i)
        if (paged_key[i] != key.data_f16()[token * 1024 + i] ||
            paged_value[i] != value.data_f16()[token * 1024 + i])
          fail(name + " K/V mismatch layer=" + std::to_string(layer) +
               " token=" + std::to_string(token) + " index=" + std::to_string(i));
    }
  }
}

}  // namespace

int main() {
  int devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&devices));
  if (devices == 0) {
    std::cout << "SKIP test_qwen3_paged_padded_prefill: no CUDA device\n";
    return 0;
  }
  try {
    struct FaultGuard {
      ~FaultGuard() noexcept { qwen3_clear_paged_prefill_fault_for_testing(); }
    } fault_guard;
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");

    for (const std::vector<size_t>& lengths :
         std::vector<std::vector<size_t>>{{4, 6}, {3, 4, 6, 8}}) {
      const size_t batch = lengths.size();
      PagedKvCachePool pool(pool_config(32));
      std::vector<std::vector<int32_t>> prompts;
      std::vector<Qwen3KvCache> contiguous;
      std::vector<Qwen3KvCache*> contiguous_ptrs;
      std::vector<std::unique_ptr<Qwen3PagedKvCache>> paged;
      std::vector<Qwen3PagedKvCache*> paged_ptrs;
      prompts.reserve(batch);
      contiguous.reserve(batch);
      contiguous_ptrs.reserve(batch);
      paged.reserve(batch);
      paged_ptrs.reserve(batch);
      for (size_t b = 0; b < batch; ++b) {
        prompts.push_back(prompt(lengths[b], 100 + static_cast<int>(b) * 97));
        contiguous.emplace_back(32);
        contiguous_ptrs.push_back(&contiguous.back());
        paged.emplace_back(std::make_unique<Qwen3PagedKvCache>(pool, 32));
        paged_ptrs.push_back(paged.back().get());
      }

      PaddedPrefillBatch contiguous_batch = build_padded_prefill_batch(prompts, contiguous_ptrs);
      PagedPaddedPrefillBatch paged_batch = build_paged_padded_prefill_batch(prompts, paged_ptrs);
      expect(paged_batch.valid_lengths == lengths &&
             paged_batch.max_seq_len == contiguous_batch.max_seq_len &&
             paged_batch.padded_token_ids == contiguous_batch.padded_token_ids,
             "paged padded metadata mismatch");
      Tensor expected = model.prefill_logits_padded_batch_with_caches(contiguous_batch);
      Tensor actual = model.prefill_logits_paged_padded_batch(paged_batch);
      compare(select_last_valid_logits(actual, lengths),
              select_last_valid_logits(expected, lengths),
              "B=" + std::to_string(batch) + " padded last-valid logits");
      for (size_t b = 0; b < batch; ++b) {
        expect(paged[b]->length() == lengths[b], "paged valid length was not committed");
        compare_kv(*paged[b], contiguous[b], lengths[b],
                   "B=" + std::to_string(batch) + " row=" + std::to_string(b));
      }
      std::vector<int32_t> next_ids(batch);
      for (size_t b = 0; b < batch; ++b) next_ids[b] = 901 + static_cast<int32_t>(b);
      compare(model.decode_logits_paged_batch(next_ids, paged_ptrs),
              model.decode_logits_variable_length_batch_with_caches(next_ids, contiguous_ptrs),
              "B=" + std::to_string(batch) + " decode after padded prefill");
      for (size_t b = 0; b < batch; ++b) {
        expect(paged[b]->length() == lengths[b] + 1, "paged decode length mismatch");
        paged[b]->release_all();
      }
      expect(pool.used_block_count() == 0, "paged padded prefill leaked blocks");
    }

    {
      PagedKvCachePool pool(pool_config(16));
      Qwen3PagedKvCache left(pool, 32), right(pool, 32);
      PagedPaddedPrefillBatch invalid = build_paged_padded_prefill_batch(
          {prompt(4, 1), prompt(6, 2)}, {&left, &right});
      invalid.padded_token_ids[4] = 123;
      expect_throw("non-zero right padding", [&] {
        model.prefill_logits_paged_padded_batch(invalid);
      });
      expect(left.length() == 0 && right.length() == 0 && pool.used_block_count() == 0,
             "invalid metadata changed paged cache state");
    }

    {
      // The first row allocates two blocks, then the second row cannot begin.
      // The model must abort the already-begun first row as well.
      PagedKvCachePool pool(pool_config(3));
      Qwen3PagedKvCache left(pool, 32), right(pool, 32);
      PagedPaddedPrefillBatch batch = build_paged_padded_prefill_batch(
          {prompt(17, 11), prompt(18, 12)}, {&left, &right});
      expect_throw("padded prefill begin exhaustion", [&] {
        model.prefill_logits_paged_padded_batch(batch);
      });
      expect(left.length() == 0 && right.length() == 0 &&
             !left.in_prefill_transaction() && !right.in_prefill_transaction() &&
             pool.used_block_count() == 0, "begin exhaustion did not roll back every paged row");
    }

    {
      PagedKvCachePool pool(pool_config(16));
      Qwen3PagedKvCache left(pool, 32), right(pool, 32);
      PagedPaddedPrefillBatch batch = build_paged_padded_prefill_batch(
          {prompt(4, 3), prompt(6, 4)}, {&left, &right});
      qwen3_set_paged_prefill_fault_for_testing(0);
      expect_throw("padded prefill fault", [&] {
        model.prefill_logits_paged_padded_batch(batch);
      });
      expect(left.length() == 0 && right.length() == 0 &&
             !left.in_prefill_transaction() && !right.in_prefill_transaction() &&
             pool.used_block_count() == 0, "fault did not roll back every paged row");
      qwen3_clear_paged_prefill_fault_for_testing();
      Tensor logits = model.prefill_logits_paged_padded_batch(batch);
      expect(logits.shape() == std::vector<int64_t>{2, 6, 151936},
             "cache was not reusable after rollback");
      left.release_all();
      right.release_all();
      expect(pool.used_block_count() == 0, "rollback/retry leaked paged blocks");
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_qwen3_paged_padded_prefill passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
