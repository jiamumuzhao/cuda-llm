#include "llm/cuda_check.h"
#include "llm/ops_cuda.h"
#include "llm/paged_kv_cache_pool.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& message) {
  throw std::runtime_error("test_paged_gqa_attention_cuda: " + message);
}

static void expect_throw(const std::function<void()>& fn, const std::string& name) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  fail(name + " did not throw");
}

static PagedKvCachePoolConfig config() {
  PagedKvCachePoolConfig c;
  c.total_blocks = 8;
  c.num_layers = 2;
  c.num_kv_heads = 2;
  c.block_size = 4;
  c.head_dim = 8;
  c.dtype = DType::F16;
  return c;
}

static Tensor make_q(size_t q_heads, size_t dim) {
  Tensor q(DType::F16, {static_cast<int64_t>(q_heads), static_cast<int64_t>(dim)});
  for (size_t h = 0; h < q_heads; ++h)
    for (size_t d = 0; d < dim; ++d)
      q.set_f32(h * dim + d, 0.17f + 0.031f * static_cast<float>(h) +
                              0.007f * static_cast<float>(d));
  return q;
}

static void make_kv(Tensor& k, Tensor& v, size_t length, size_t kv_heads, size_t dim) {
  for (size_t t = 0; t < length; ++t) {
    for (size_t h = 0; h < kv_heads; ++h) {
      for (size_t d = 0; d < dim; ++d) {
        const size_t index = (t * kv_heads + h) * dim + d;
        k.set_f32(index, 0.11f * static_cast<float>(t + 1) +
                           0.23f * static_cast<float>(h + 1) +
                           0.013f * static_cast<float>(d));
        v.set_f32(index, -0.37f + 0.19f * static_cast<float>(t) +
                           0.41f * static_cast<float>(h) +
                           0.017f * static_cast<float>(d));
      }
    }
  }
}

static std::vector<float> reference(const Tensor& q, const Tensor& k,
                                    const Tensor& v, size_t q_heads,
                                    size_t kv_heads, size_t dim) {
  std::vector<float> out(q_heads * dim);
  const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
  const size_t length = static_cast<size_t>(k.shape()[0]);
  const size_t group = q_heads / kv_heads;
  for (size_t h = 0; h < q_heads; ++h) {
    const size_t kv_head = h / group;
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t t = 0; t < length; ++t) {
      float score = 0.0f;
      for (size_t d = 0; d < dim; ++d)
        score += q.get_f32(h * dim + d) * k.get_f32((t * kv_heads + kv_head) * dim + d);
      maximum = std::max(maximum, score * scale);
    }
    float denominator = 0.0f;
    for (size_t t = 0; t < length; ++t) {
      float score = 0.0f;
      for (size_t d = 0; d < dim; ++d)
        score += q.get_f32(h * dim + d) * k.get_f32((t * kv_heads + kv_head) * dim + d);
      denominator += std::exp(score * scale - maximum);
    }
    for (size_t d = 0; d < dim; ++d) {
      float value = 0.0f;
      for (size_t t = 0; t < length; ++t) {
        float score = 0.0f;
        for (size_t e = 0; e < dim; ++e)
          score += q.get_f32(h * dim + e) * k.get_f32((t * kv_heads + kv_head) * dim + e);
        value += std::exp(score * scale - maximum) *
                 v.get_f32((t * kv_heads + kv_head) * dim + d);
      }
      out[h * dim + d] = value / denominator;
    }
  }
  return out;
}

static void compare(const Tensor& actual_cuda, const std::vector<float>& expected,
                    const std::string& label) {
  Tensor actual = actual_cuda.to(DeviceType::CPU);
  float max_error = 0.0f;
  for (size_t i = 0; i < expected.size(); ++i)
    max_error = std::max(max_error, std::fabs(actual.get_f32(i) - expected[i]));
  if (max_error > 5e-3f)
    fail(label + " max_abs_error=" + std::to_string(max_error) + " expected<=0.005");
  std::cout << label << " max_abs_error=" << max_error << " tolerance=0.005\n";
}

static void compare_tensors(const Tensor& lhs_cuda, const Tensor& rhs_cuda,
                            const std::string& label) {
  Tensor lhs = lhs_cuda.to(DeviceType::CPU);
  Tensor rhs = rhs_cuda.to(DeviceType::CPU);
  float max_error = 0.0f;
  for (size_t i = 0; i < lhs.numel(); ++i)
    max_error = std::max(max_error, std::fabs(lhs.get_f32(i) - rhs.get_f32(i)));
  if (max_error > 5e-3f)
    fail(label + " max_abs_error=" + std::to_string(max_error) + " expected<=0.005");
  std::cout << label << " max_abs_error=" << max_error << " tolerance=0.005\n";
}

static void run_batch_case(std::size_t batch,
                           const std::vector<std::size_t>& lengths) {
  PagedKvCachePool pool(config());
  std::vector<std::unique_ptr<PagedSequenceKvCache>> sequences;
  sequences.reserve(batch);
  Tensor q_host(DType::F16, {static_cast<int64_t>(batch), 4, 8});
  Tensor tables_host(DType::I32, {static_cast<int64_t>(batch), 2});
  Tensor positions_host(DType::I32, {static_cast<int64_t>(batch)});
  std::vector<Tensor> keys;
  std::vector<Tensor> values;
  keys.reserve(batch); values.reserve(batch);
  for (std::size_t b = 0; b < batch; ++b) {
    sequences.push_back(std::make_unique<PagedSequenceKvCache>(pool, 8));
    sequences.back()->append_tokens(lengths[b]);
    positions_host.data_i32()[b] = static_cast<int32_t>(lengths[b] - 1);
    for (std::size_t i = 0; i < 4 * 8; ++i)
      q_host.set_f32(b * 4 * 8 + i, 0.17f + 0.031f * static_cast<float>(i / 8) +
                                      0.007f * static_cast<float>(i % 8) +
                                      0.013f * static_cast<float>(b));
    keys.emplace_back(Tensor(
        DType::F16, {static_cast<int64_t>(lengths[b]), 2, 8}));
    values.emplace_back(DType::F16, keys.back().shape());
    make_kv(keys.back(), values.back(), lengths[b], 2, 8);
    for (std::size_t t = 0; t < lengths[b]; ++t) {
      sequences.back()->copy_token_from_host(
          0, t, PagedKvKind::Key, keys.back().data_f16() + t * 16, 16);
      sequences.back()->copy_token_from_host(
          0, t, PagedKvKind::Value, values.back().data_f16() + t * 16, 16);
    }
    const std::uint16_t sentinel = 0x3555;
    std::vector<std::uint16_t> tail(16, sentinel);
    for (std::size_t t = lengths[b];
         t < sequences.back()->block_table().size() * 4; ++t) {
      const BlockId block = sequences.back()->block_table()[t / 4];
      const std::size_t offset = t % 4;
      CUDA_CHECK(cudaMemcpy(pool.device_token_ptr(0, block, PagedKvKind::Key, offset),
                            tail.data(), tail.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(pool.device_token_ptr(0, block, PagedKvKind::Value, offset),
                            tail.data(), tail.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice));
    }
    for (std::size_t i = 0; i < 2; ++i)
      tables_host.data_i32()[b * 2 + i] =
          i < sequences.back()->block_table().size()
              ? static_cast<int32_t>(sequences.back()->block_table()[i])
              : 0;
  }
  Tensor q_cuda = q_host.to(DeviceType::CUDA);
  Tensor tables_cuda = tables_host.to(DeviceType::CUDA);
  Tensor positions_cuda = positions_host.to(DeviceType::CUDA);
  Tensor batch_out = cuda_paged_gqa_attention_decode_batch(
      q_cuda, pool, 0, tables_cuda, positions_cuda, 4, 2, 8);
  Tensor batch_cpu = batch_out.to(DeviceType::CPU);
  for (std::size_t b = 0; b < batch; ++b) {
    std::vector<float> actual(32);
    for (std::size_t i = 0; i < actual.size(); ++i)
      actual[i] = batch_cpu.get_f32(b * 32 + i);
    const Tensor q_row = q_host.slice_first_dim(b, 1).reshape({4, 8});
    const auto expected = reference(q_row, keys[b], values[b], 4, 2, 8);
    float max_error = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i)
      max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
    if (max_error > 5e-3f)
      fail("batch B=" + std::to_string(batch) + " row=" + std::to_string(b) +
           " max_abs_error=" + std::to_string(max_error));
    Tensor row_q_cuda = q_row.to(DeviceType::CUDA);
    Tensor row_table = sequences[b]->make_device_block_table_i32();
    Tensor single = cuda_paged_gqa_attention_decode(
        row_q_cuda, pool, 0, row_table, lengths[b], 4, 2, 8);
    compare_tensors(single, batch_out.slice_first_dim(b, 1).reshape({4, 8}),
                    "batch B=" + std::to_string(batch) + " row=" + std::to_string(b) +
                    " vs single");
  }
  Tensor out(DType::F16, {static_cast<int64_t>(batch), 4, 8}, DeviceType::CUDA);
  reset_cuda_allocation_stats();
  cuda_paged_gqa_attention_decode_batch_out(
      q_cuda, pool, 0, tables_cuda, positions_cuda, 4, 2, 8, out);
  CUDA_CHECK(cudaDeviceSynchronize());
  const CudaAllocationStats stats = cuda_allocation_stats();
  if (stats.cuda_malloc_calls != 0 || stats.cuda_free_calls != 0)
    fail("batch _out allocated CUDA Tensor for B=" + std::to_string(batch));
  compare_tensors(out, batch_out,
                  "batch B=" + std::to_string(batch) + " _out");
  std::cout << "paged batch B=" << batch << " mixed lengths=" << lengths[0]
            << "," << lengths[batch - 1]
            << " CPU/single/_out alignment and zero allocation passed\n";
}

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    std::cout << "SKIP test_paged_gqa_attention_cuda: no CUDA device\n";
    return 0;
  }
  try {
    const size_t q_heads = 4, kv_heads = 2, dim = 8;
    for (size_t length : {size_t(1), size_t(4), size_t(5), size_t(8)}) {
      PagedKvCachePool pool(config());
      PagedSequenceKvCache target(pool, 8);
      target.append_tokens(4);
      PagedSequenceKvCache interposer(pool, 4);
      interposer.append_tokens(4);
      if (length > 4) target.append_tokens(length - 4);
      if (length > 4 && target.block_table().size() != 2)
        fail("cross-block test did not allocate two logical blocks");
      if (length > 4 && target.block_table()[1] == target.block_table()[0] + 1)
        fail("test setup did not produce a non-contiguous physical block table");

      Tensor q_host = make_q(q_heads, dim);
      Tensor k_host(DType::F16, {static_cast<int64_t>(length), static_cast<int64_t>(kv_heads),
                                 static_cast<int64_t>(dim)});
      Tensor v_host = Tensor(DType::F16, k_host.shape());
      make_kv(k_host, v_host, length, kv_heads, dim);
      for (size_t t = 0; t < length; ++t) {
        target.copy_token_from_host(0, t, PagedKvKind::Key,
                                    k_host.data_f16() + t * kv_heads * dim,
                                    kv_heads * dim);
        target.copy_token_from_host(0, t, PagedKvKind::Value,
                                    v_host.data_f16() + t * kv_heads * dim,
                                    kv_heads * dim);
      }
      if (length == 5) {
        std::vector<std::uint16_t> sentinel(kv_heads * dim, 0x7bff);
        for (size_t offset = 1; offset < 4; ++offset) {
          CUDA_CHECK(cudaMemcpy(pool.device_token_ptr(0, target.block_table()[1],
                                                       PagedKvKind::Key, offset),
                                sentinel.data(), sentinel.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(pool.device_token_ptr(0, target.block_table()[1],
                                                       PagedKvKind::Value, offset),
                                sentinel.data(), sentinel.size() * sizeof(std::uint16_t),
                                cudaMemcpyHostToDevice));
        }
      }

      Tensor q_cuda = q_host.to(DeviceType::CUDA);
      Tensor k_cuda = k_host.to(DeviceType::CUDA);
      Tensor v_cuda = v_host.to(DeviceType::CUDA);
      Tensor table_cuda = target.make_device_block_table_i32();
      Tensor paged = cuda_paged_gqa_attention_decode(
          q_cuda, pool, 0, table_cuda, length, q_heads, kv_heads, dim);
      compare(paged, reference(q_host, k_host, v_host, q_heads, kv_heads, dim),
              "length=" + std::to_string(length) + " CPU");

      Tensor q_sequence_host(DType::F16,
                             {static_cast<int64_t>(length), static_cast<int64_t>(q_heads),
                              static_cast<int64_t>(dim)});
      for (size_t i = 0; i < q_sequence_host.numel(); ++i) q_sequence_host.set_f32(i, 0.0f);
      for (size_t i = 0; i < q_host.numel(); ++i)
        q_sequence_host.set_f32((length - 1) * q_host.numel() + i, q_host.get_f32(i));
      Tensor q3 = q_sequence_host.to(DeviceType::CUDA);
      Tensor contiguous = cuda_gqa_attention(q3, k_cuda, v_cuda);
      Tensor contiguous_cpu = contiguous.to(DeviceType::CPU);
      Tensor contiguous_last_cpu(DType::F16,
                                 {static_cast<int64_t>(q_heads), static_cast<int64_t>(dim)});
      for (size_t i = 0; i < contiguous_last_cpu.numel(); ++i)
        contiguous_last_cpu.set_f32(i, contiguous_cpu.get_f32((length - 1) * q_heads * dim + i));
      compare(contiguous_last_cpu, reference(q_host, k_host, v_host, q_heads, kv_heads, dim),
              "length=" + std::to_string(length) + " contiguous-last");
      compare_tensors(paged, contiguous_last_cpu.to(DeviceType::CUDA),
                      "length=" + std::to_string(length) + " paged-vs-contiguous");
      target.release_all();
      interposer.release_all();
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::cout << "cross-block physical lookup, GQA mapping and length=5 tail masking passed\n";

    run_batch_case(1, {1});
    run_batch_case(2, {4, 5});
    run_batch_case(4, {1, 4, 5, 8});

    PagedKvCachePool pool(config());
    PagedSequenceKvCache sequence(pool, 4);
    sequence.append_tokens(1);
    Tensor table = sequence.make_device_block_table_i32();
    Tensor q = make_q(4, 8);
    Tensor q_cuda = q.to(DeviceType::CUDA);
    Tensor q_f32 = Tensor(DType::F32, q.shape());
    for (size_t i = 0; i < q.numel(); ++i) q_f32.set_f32(i, q.get_f32(i));
    Tensor table_cpu(DType::I32, {1});
    table_cpu.data_i32()[0] = static_cast<int32_t>(sequence.block_table()[0]);
    Tensor table_f32(DType::F32, {1});
    Tensor table_rank2(DType::I32, {1, 1}, DeviceType::CUDA);
    Tensor free_table(DType::I32, {1}, DeviceType::CUDA);
    const int32_t free_id = 7;
    CUDA_CHECK(cudaMemcpy(free_table.data(), &free_id, sizeof(free_id), cudaMemcpyHostToDevice));

    expect_throw([&] { cuda_paged_gqa_attention_decode(q, pool, 0, table, 1, 4, 2, 8); }, "CPU q");
    expect_throw([&] { cuda_paged_gqa_attention_decode(q_f32.to(DeviceType::CUDA), pool, 0, table, 1, 4, 2, 8); }, "F32 q");
    expect_throw([&] { cuda_paged_gqa_attention_decode(q_cuda, pool, 0, table_cpu, 1, 4, 2, 8); }, "CPU block table");
    expect_throw([&] { cuda_paged_gqa_attention_decode(q_cuda, pool, 0, table_f32.to(DeviceType::CUDA), 1, 4, 2, 8); }, "F32 block table");
    expect_throw([&] { cuda_paged_gqa_attention_decode(q_cuda, pool, 0, table_rank2, 1, 4, 2, 8); }, "rank-2 block table");
    expect_throw([&] { PagedSequenceKvCache empty(pool, 4); empty.make_device_block_table_i32(); }, "empty block table");
    expect_throw([&] { cuda_paged_gqa_attention_decode(q_cuda, pool, 0, table, 0, 4, 2, 8); }, "zero kv length");
    expect_throw([&] { cuda_paged_gqa_attention_decode(q_cuda, pool, 0, table, 5, 4, 2, 8); }, "kv length over capacity");
    expect_throw([&] { cuda_paged_gqa_attention_decode(q_cuda, pool, 2, table, 1, 4, 2, 8); }, "invalid layer");
    expect_throw([&] { cuda_paged_gqa_attention_decode(q_cuda, pool, 0, table, 1, 3, 2, 8); }, "invalid GQA ratio");
    Tensor bad_table(DType::I32, {1}, DeviceType::CUDA);
    const int32_t bad_id = -1;
    CUDA_CHECK(cudaMemcpy(bad_table.data(), &bad_id, sizeof(bad_id), cudaMemcpyHostToDevice));
    expect_throw([&] { cuda_paged_gqa_attention_decode(q_cuda, pool, 0, bad_table, 1, 4, 2, 8); }, "free/invalid block id");
    expect_throw([&] { cuda_paged_gqa_attention_decode(q_cuda, pool, 0, free_table, 1, 4, 2, 8); }, "free block id");

    Tensor batch_q_bad(DType::F16, {3, 4, 8}, DeviceType::CUDA);
    Tensor batch_tables_bad(DType::I32, {3, 2}, DeviceType::CUDA);
    Tensor batch_positions_bad(DType::I32, {3}, DeviceType::CUDA);
    expect_throw([&] { cuda_paged_gqa_attention_decode_batch(
        batch_q_bad, pool, 0, batch_tables_bad, batch_positions_bad, 4, 2, 8); },
        "unsupported batch size");
    Tensor batch_q_cpu(DType::F16, {1, 4, 8});
    Tensor batch_tables_cpu(DType::I32, {1, 2});
    Tensor batch_positions_cpu(DType::I32, {1});
    expect_throw([&] { cuda_paged_gqa_attention_decode_batch(
        batch_q_cpu, pool, 0, batch_tables_cpu, batch_positions_cpu, 4, 2, 8); },
        "batch CPU inputs");
    Tensor batch_positions_bad_value(DType::I32, {1}, DeviceType::CUDA);
    const int32_t bad_position = 8;
    CUDA_CHECK(cudaMemcpy(batch_positions_bad_value.data(), &bad_position,
                          sizeof(bad_position), cudaMemcpyHostToDevice));
    Tensor batch_q_one = q_cuda.reshape({1, 4, 8});
    Tensor batch_table_one = table.reshape({1, 1});
    expect_throw([&] { cuda_paged_gqa_attention_decode_batch(
        batch_q_one, pool, 0, batch_table_one, batch_positions_bad_value, 4, 2, 8); },
        "batch position out of range");
    Tensor batch_output_bad(DType::F16, {1, 4, 7}, DeviceType::CUDA);
    Tensor wrong_positions(DType::I32, {1});
    expect_throw([&] { cuda_paged_gqa_attention_decode_batch_out(
        batch_q_one, pool, 0, batch_table_one, wrong_positions, 4, 2, 8,
        batch_output_bad); }, "batch output/metadata shape");
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "paged attention validation and CUDA health checks passed\n";
    std::cout << "test_paged_gqa_attention_cuda passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
