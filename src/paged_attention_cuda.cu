#include "llm/ops_cuda.h"

#include "llm/cuda_check.h"
#include "llm/paged_kv_cache_pool.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <atomic>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CUDART_INF_F
#define CUDART_INF_F 3.402823466e+38F
#endif

namespace llm {
namespace {

std::atomic<std::uint64_t> g_paged_batch_attention_launches{0};

__global__ void paged_gqa_decode_kernel(
    const __half* q, const __half* storage, const int32_t* block_table,
    __half* output, std::size_t layer, std::size_t total_blocks,
    std::size_t block_size, std::size_t kv_heads, std::size_t head_dim,
    std::size_t kv_length, const int32_t* device_kv_length,
    std::size_t q_heads, float scale) {
  if (device_kv_length != nullptr)
    kv_length = static_cast<std::size_t>(*device_kv_length);
  const std::size_t q_head = blockIdx.x;
  const std::size_t dim = threadIdx.x;
  if (q_head >= q_heads || dim >= head_dim) return;
  extern __shared__ unsigned char shared_memory[];
  __half* query_shared = reinterpret_cast<__half*>(shared_memory);
  __half* key_shared = query_shared + head_dim;
  const std::size_t group = q_heads / kv_heads;
  const std::size_t kv_head = q_head / group;
  const std::size_t token_stride = kv_heads * head_dim;
  const std::size_t plane_elements = block_size * token_stride;
  const __half* query = q + q_head * head_dim;
  query_shared[dim] = query[dim];
  __syncthreads();
  float maximum = -CUDART_INF_F;
  for (std::size_t token = 0; token < kv_length; ++token) {
    const std::size_t block = static_cast<std::size_t>(block_table[token / block_size]);
    const std::size_t offset = token % block_size;
    const std::size_t layer_block = (layer * total_blocks + block) * 2;
    const __half* key = storage + layer_block * plane_elements +
                        offset * token_stride + kv_head * head_dim;
    key_shared[dim] = key[dim];
    __syncthreads();
    float score = 0.0f;
    for (std::size_t e = 0; e < head_dim; ++e)
      score += __half2float(query_shared[e]) * __half2float(key_shared[e]);
    maximum = fmaxf(maximum, score * scale);
    __syncthreads();
  }
  float denominator = 0.0f;
  float result = 0.0f;
  for (std::size_t token = 0; token < kv_length; ++token) {
    const std::size_t block = static_cast<std::size_t>(block_table[token / block_size]);
    const std::size_t offset = token % block_size;
    const std::size_t layer_block = (layer * total_blocks + block) * 2;
    const __half* key = storage + layer_block * plane_elements +
                        offset * token_stride + kv_head * head_dim;
    const __half* value = storage + (layer_block + 1) * plane_elements +
                          offset * token_stride + kv_head * head_dim;
    key_shared[dim] = key[dim];
    __syncthreads();
    float score = 0.0f;
    for (std::size_t e = 0; e < head_dim; ++e)
      score += __half2float(query_shared[e]) * __half2float(key_shared[e]);
    const float weight = expf(score * scale - maximum);
    denominator += weight;
    result += weight * __half2float(value[dim]);
    __syncthreads();
  }
  output[q_head * head_dim + dim] = __float2half(result / denominator);
}

// Single-query FlashAttention-style path: load K/V tiles once, compute each
// score once, and update the softmax/value state online.
__global__ void paged_gqa_decode_flash_kernel(
    const __half* q, const __half* storage, const int32_t* block_table,
    __half* output, std::size_t layer, std::size_t total_blocks,
    std::size_t block_size, std::size_t kv_heads, std::size_t head_dim,
    std::size_t kv_length, const int32_t* device_kv_length,
    std::size_t q_heads, float scale) {
  if (device_kv_length != nullptr)
    kv_length = static_cast<std::size_t>(*device_kv_length);
  constexpr std::size_t kTile = 32;
  const std::size_t q_head = blockIdx.x;
  const std::size_t dim = threadIdx.x;
  if (q_head >= q_heads || dim >= head_dim) return;
  extern __shared__ __half smem[];
  __half* query = smem;
  __half* key_tile = query + head_dim;
  __half* value_tile = key_tile + kTile * head_dim;
  float* scores = reinterpret_cast<float*>(value_tile + kTile * head_dim);
  __shared__ float partial[4];
  const std::size_t group = q_heads / kv_heads;
  const std::size_t kv_head = q_head / group;
  const std::size_t token_stride = kv_heads * head_dim;
  const std::size_t plane_elements = block_size * token_stride;
  query[dim] = q[q_head * head_dim + dim];
  __syncthreads();
  float maximum = -CUDART_INF_F;
  double denominator = 0.0;
  double result = 0.0;
  for (std::size_t tile_begin = 0; tile_begin < kv_length; tile_begin += kTile) {
    const std::size_t tile_count = min(kTile, kv_length - tile_begin);
    for (std::size_t i = dim; i < tile_count * head_dim; i += head_dim) {
      const std::size_t token = tile_begin + i / head_dim;
      const std::size_t block = static_cast<std::size_t>(block_table[token / block_size]);
      const std::size_t offset = token % block_size;
      const std::size_t layer_block = (layer * total_blocks + block) * 2;
      const __half* key = storage + layer_block * plane_elements +
                          offset * token_stride + kv_head * head_dim;
      const __half* value = storage + (layer_block + 1) * plane_elements +
                            offset * token_stride + kv_head * head_dim;
      key_tile[i] = key[i % head_dim];
      value_tile[i] = value[i % head_dim];
    }
    __syncthreads();
    for (std::size_t key = 0; key < tile_count; ++key) {
      float partial_score = 0.0f;
      for (std::size_t e = dim; e < head_dim; e += head_dim)
        partial_score += __half2float(query[e]) * __half2float(key_tile[key * head_dim + e]);
      for (int delta = 16; delta > 0; delta >>= 1)
        partial_score += __shfl_down_sync(0xffffffff, partial_score, delta);
      if ((dim % 32) == 0) partial[dim / 32] = partial_score;
      __syncthreads();
      if (dim == 0) scores[key] = (partial[0] + partial[1] + partial[2] + partial[3]) * scale;
      __syncthreads();
    }
    if (dim == 0) { float tile_max = -CUDART_INF_F; for (std::size_t key = 0; key < tile_count; ++key) tile_max = fmaxf(tile_max, scores[key]); scores[32] = tile_max; }
    __syncthreads();
    const float next_max = fmaxf(maximum, scores[32]);
    const float old_scale = expf(maximum - next_max);
    result *= double(old_scale); denominator *= double(old_scale);
    for (std::size_t key = 0; key < tile_count; ++key) {
      const float weight = expf(scores[key] - next_max);
      result += double(weight) * __half2float(value_tile[key * head_dim + dim]);
      denominator += double(weight);
    }
    maximum = next_max;
    __syncthreads();
  }
  output[q_head * head_dim + dim] = __float2half(result / denominator);
}

__global__ void paged_gqa_decode_flash_batch_kernel(
    const __half* q, const __half* storage, const int32_t* block_tables,
    const int32_t* positions, __half* output, std::size_t layer,
    std::size_t total_blocks, std::size_t block_size, std::size_t kv_heads,
    std::size_t head_dim, std::size_t q_heads, std::size_t max_blocks,
    float scale) {
  constexpr std::size_t kTile = 32;
  const std::size_t index = blockIdx.x, q_head = index % q_heads;
  const std::size_t row = index / q_heads, dim = threadIdx.x;
  if (dim >= head_dim) return;
  extern __shared__ __half smem[];
  __half* query = smem; __half* key_tile = query + head_dim;
  __half* value_tile = key_tile + kTile * head_dim;
  float* scores = reinterpret_cast<float*>(value_tile + kTile * head_dim);
  __shared__ float partial[4];
  const std::size_t group = q_heads / kv_heads, kv_head = q_head / group;
  const std::size_t token_stride = kv_heads * head_dim;
  const std::size_t plane_elements = block_size * token_stride;
  const std::size_t q_stride = q_heads * head_dim;
  const __half* q_row = q + row * q_stride + q_head * head_dim;
  const int32_t* table = block_tables + row * max_blocks;
  const std::size_t kv_length = static_cast<std::size_t>(positions[row]) + 1;
  query[dim] = q_row[dim]; __syncthreads();
  float maximum = -CUDART_INF_F; double denominator = 0.0, result = 0.0;
  for (std::size_t begin = 0; begin < kv_length; begin += kTile) {
    const std::size_t count = min(kTile, kv_length - begin);
    for (std::size_t i = dim; i < count * head_dim; i += head_dim) {
      const std::size_t token = begin + i / head_dim;
      const std::size_t block = static_cast<std::size_t>(table[token / block_size]);
      const std::size_t offset = token % block_size;
      const std::size_t layer_block = (layer * total_blocks + block) * 2;
      const __half* key = storage + layer_block * plane_elements + offset * token_stride + kv_head * head_dim;
      const __half* value = storage + (layer_block + 1) * plane_elements + offset * token_stride + kv_head * head_dim;
      key_tile[i] = key[i % head_dim]; value_tile[i] = value[i % head_dim];
    }
    __syncthreads();
    for (std::size_t key = 0; key < count; ++key) {
      float part = 0.0f;
      for (std::size_t e = dim; e < head_dim; e += head_dim)
        part += __half2float(query[e]) * __half2float(key_tile[key * head_dim + e]);
      for (int delta = 16; delta > 0; delta >>= 1)
        part += __shfl_down_sync(0xffffffff, part, delta);
      if ((dim % 32) == 0) partial[dim / 32] = part;
      __syncthreads();
      if (dim == 0) scores[key] = (partial[0] + partial[1] + partial[2] + partial[3]) * scale;
      __syncthreads();
    }
    if (dim == 0) { float tile_max = -CUDART_INF_F; for (std::size_t key = 0; key < count; ++key) tile_max = fmaxf(tile_max, scores[key]); scores[32] = tile_max; }
    __syncthreads();
    const float next_max = fmaxf(maximum, scores[32]);
    const float old_scale = expf(maximum - next_max);
    result *= double(old_scale); denominator *= double(old_scale);
    for (std::size_t key = 0; key < count; ++key) {
      const float weight = expf(scores[key] - next_max);
      result += double(weight) * __half2float(value_tile[key * head_dim + dim]);
      denominator += double(weight);
    }
    maximum = next_max;
    __syncthreads();
  }
  output[row * q_stride + q_head * head_dim + dim] = __float2half(result / denominator);
}

__global__ void paged_kv_write_decode_kernel(
    const __half* key, const __half* value, __half* storage,
    const int32_t* block_table, const int32_t* device_cache_length,
    std::size_t layer, std::size_t total_blocks, std::size_t block_size,
    std::size_t kv_heads, std::size_t head_dim) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
  const std::size_t elements = kv_heads * head_dim;
  if (element >= elements) return;
  const std::size_t cache_length = static_cast<std::size_t>(*device_cache_length);
  if (cache_length == 0) return;
  const std::size_t token = cache_length - 1;
  const std::size_t block = static_cast<std::size_t>(block_table[token / block_size]);
  const std::size_t offset = token % block_size;
  const std::size_t plane_elements = block_size * elements;
  const std::size_t layer_block = (layer * total_blocks + block) * 2;
  const std::size_t dst = layer_block * plane_elements + offset * elements + element;
  storage[dst] = key[element];
  storage[(layer_block + 1) * plane_elements + offset * elements + element] = value[element];
}

__global__ void paged_kv_write_decode_batch_kernel(
    const __half* key, const __half* value, __half* storage,
    const int32_t* block_tables, const int32_t* positions,
    std::size_t layer, std::size_t total_blocks, std::size_t block_size,
    std::size_t kv_heads, std::size_t head_dim, std::size_t max_blocks,
    std::size_t batch_size) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
  const std::size_t per_row = kv_heads * head_dim;
  const std::size_t batch = element / per_row;
  const std::size_t row_element = element % per_row;
  if (batch >= batch_size) return;
  const std::size_t cache_length = static_cast<std::size_t>(positions[batch]) + 1;
  const std::size_t token = cache_length - 1;
  const std::size_t block = static_cast<std::size_t>(
      block_tables[batch * max_blocks + token / block_size]);
  const std::size_t offset = token % block_size;
  const std::size_t plane_elements = block_size * per_row;
  const std::size_t layer_block = (layer * total_blocks + block) * 2;
  const std::size_t src = batch * per_row + row_element;
  storage[layer_block * plane_elements + offset * per_row + row_element] = key[src];
  storage[(layer_block + 1) * plane_elements + offset * per_row + row_element] = value[src];
}

__global__ void paged_gqa_decode_batch_kernel(
    const __half* q, const __half* storage, const int32_t* block_tables,
    const int32_t* positions, __half* output, std::size_t layer,
    std::size_t total_blocks, std::size_t block_size, std::size_t kv_heads,
    std::size_t head_dim, std::size_t q_heads, std::size_t max_blocks,
    float scale) {
  const std::size_t q_index = blockIdx.x;
  const std::size_t row = q_index / q_heads;
  const std::size_t q_head = q_index % q_heads;
  const std::size_t dim = threadIdx.x;
  if (dim >= head_dim) return;
  extern __shared__ unsigned char shared_memory[];
  __half* query_shared = reinterpret_cast<__half*>(shared_memory);
  __half* key_shared = query_shared + head_dim;
  const std::size_t group = q_heads / kv_heads;
  const std::size_t kv_head = q_head / group;
  const std::size_t token_stride = kv_heads * head_dim;
  const std::size_t plane_elements = block_size * token_stride;
  const std::size_t q_row_stride = q_heads * head_dim;
  const std::size_t out_row_stride = q_row_stride;
  const __half* q_row = q + row * q_row_stride;
  __half* out_row = output + row * out_row_stride;
  const int32_t* table = block_tables + row * max_blocks;
  const std::size_t kv_length = static_cast<std::size_t>(positions[row]) + 1;

  const __half* query = q_row + q_head * head_dim;
  query_shared[dim] = query[dim];
  __syncthreads();
  float maximum = -CUDART_INF_F;
  for (std::size_t token = 0; token < kv_length; ++token) {
    const std::size_t block = static_cast<std::size_t>(table[token / block_size]);
    const std::size_t offset = token % block_size;
    const std::size_t layer_block = (layer * total_blocks + block) * 2;
    const __half* key = storage + layer_block * plane_elements +
                        offset * token_stride + kv_head * head_dim;
    key_shared[dim] = key[dim];
    __syncthreads();
    float score = 0.0f;
    for (std::size_t e = 0; e < head_dim; ++e)
      score += __half2float(query_shared[e]) * __half2float(key_shared[e]);
    maximum = fmaxf(maximum, score * scale);
    __syncthreads();
  }
  float denominator = 0.0f;
  float result = 0.0f;
  for (std::size_t token = 0; token < kv_length; ++token) {
    const std::size_t block = static_cast<std::size_t>(table[token / block_size]);
    const std::size_t offset = token % block_size;
    const std::size_t layer_block = (layer * total_blocks + block) * 2;
    const __half* key = storage + layer_block * plane_elements +
                        offset * token_stride + kv_head * head_dim;
    const __half* value = storage + (layer_block + 1) * plane_elements +
                          offset * token_stride + kv_head * head_dim;
    key_shared[dim] = key[dim];
    __syncthreads();
    float score = 0.0f;
    for (std::size_t e = 0; e < head_dim; ++e)
      score += __half2float(query_shared[e]) * __half2float(key_shared[e]);
    const float weight = expf(score * scale - maximum);
    denominator += weight;
    result += weight * __half2float(value[dim]);
    __syncthreads();
  }
  out_row[q_head * head_dim + dim] = __float2half(result / denominator);
}


// Legacy candidate: retained as reference code, but excluded from the build.
// Its online-softmax accumulation is not numerically aligned with the
// contiguous decode reference, so production uses the generic two-pass kernel.
#if 0
[[maybe_unused]] __global__ void paged_gqa_decode_batch_group2_warp_kernel(
    const __half* q, const __half* storage, const int32_t* block_tables,
    const int32_t* positions, __half* output, std::size_t layer,
    std::size_t total_blocks, std::size_t block_size, std::size_t kv_heads,
    std::size_t head_dim, std::size_t max_blocks, float scale) {
  const std::size_t warp = threadIdx.x / 32;
  const std::size_t lane = threadIdx.x % 32;
  const std::size_t row = blockIdx.x / kv_heads;
  const std::size_t kv_head = blockIdx.x % kv_heads;
  const std::size_t q_head = kv_head * 2 + warp;
  const std::size_t dim_base = lane * 4;
  const std::size_t token_stride = kv_heads * head_dim;
  const std::size_t plane_elements = block_size * token_stride;
  const std::size_t q_row_stride = 2 * head_dim;
  const __half* q_row = q + row * q_row_stride;
  __half* out_row = output + row * q_row_stride;
  const int32_t* table = block_tables + row * max_blocks;
  const std::size_t kv_length = static_cast<std::size_t>(positions[row]) + 1;
  extern __shared__ unsigned char shared_memory[];
  __half* key_tile = reinterpret_cast<__half*>(shared_memory);
  __half* value_tile = key_tile + block_size * head_dim;

  const __half* query = q_row + q_head * head_dim;
  float query_reg[4];
  for (std::size_t j = 0; j < 4; ++j)
    query_reg[j] = __half2float(query[dim_base + j]);
  float running_max = -CUDART_INF_F;
  float denominator = 0.0f;
  float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  if (kv_length <= block_size) {
    for (std::size_t token = 0; token < kv_length; ++token) {
      const std::size_t block =
          static_cast<std::size_t>(table[token / block_size]);
      const std::size_t offset = token % block_size;
      const std::size_t layer_block = (layer * total_blocks + block) * 2;
      const __half* key = storage + layer_block * plane_elements +
                          offset * token_stride + kv_head * head_dim;
      const __half* value = storage + (layer_block + 1) * plane_elements +
                            offset * token_stride + kv_head * head_dim;
      float partial = 0.0f;
      for (std::size_t j = 0; j < 4; ++j)
        partial += query_reg[j] * __half2float(key[dim_base + j]);
      for (int delta = 16; delta > 0; delta >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, delta);
      const float score = __shfl_sync(0xffffffff, partial, 0) * scale;
      const float next_max = fmaxf(running_max, score);
      const float old_scale = expf(running_max - next_max);
      const float weight = expf(score - next_max);
      denominator = denominator * old_scale + weight;
      for (std::size_t j = 0; j < 4; ++j)
        result[j] = result[j] * old_scale +
                    weight * __half2float(value[dim_base + j]);
      running_max = next_max;
    }
    for (std::size_t j = 0; j < 4; ++j)
      out_row[q_head * head_dim + dim_base + j] =
          __float2half(result[j] / denominator);
    return;
  }

  for (std::size_t tile_start = 0; tile_start < kv_length;
       tile_start += block_size) {
    const std::size_t tile_tokens =
        min(block_size, kv_length - tile_start);
    if (tile_start != 0) __syncthreads();

    for (std::size_t tile_token = 0; tile_token < tile_tokens;
         ++tile_token) {
      const std::size_t token = tile_start + tile_token;
      const std::size_t block =
          static_cast<std::size_t>(table[token / block_size]);
      const std::size_t offset = token % block_size;
      const std::size_t layer_block = (layer * total_blocks + block) * 2;
      const __half* key = storage + layer_block * plane_elements +
                          offset * token_stride + kv_head * head_dim;
      const __half* value = storage + (layer_block + 1) * plane_elements +
                            offset * token_stride + kv_head * head_dim;
      __half2* key_tile_vec = reinterpret_cast<__half2*>(
          key_tile + tile_token * head_dim);
      __half2* value_tile_vec = reinterpret_cast<__half2*>(
          value_tile + tile_token * head_dim);
      const __half2* key_vec = reinterpret_cast<const __half2*>(key);
      const __half2* value_vec = reinterpret_cast<const __half2*>(value);
      const std::size_t vec = threadIdx.x;
      key_tile_vec[vec] = key_vec[vec];
      value_tile_vec[vec] = value_vec[vec];
    }
    __syncthreads();

    for (std::size_t tile_token = 0; tile_token < tile_tokens;
         ++tile_token) {
      const __half* key = key_tile + tile_token * head_dim;
      const __half* value = value_tile + tile_token * head_dim;
      float partial = 0.0f;
      for (std::size_t j = 0; j < 4; ++j)
        partial += query_reg[j] * __half2float(key[dim_base + j]);
      for (int delta = 16; delta > 0; delta >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, delta);
      const float score = __shfl_sync(0xffffffff, partial, 0) * scale;
      const float next_max = fmaxf(running_max, score);
      const float old_scale = expf(running_max - next_max);
      const float weight = expf(score - next_max);
      denominator = denominator * old_scale + weight;
      for (std::size_t j = 0; j < 4; ++j)
        result[j] = result[j] * old_scale +
                    weight * __half2float(value[dim_base + j]);
      running_max = next_max;
    }
  }

  for (std::size_t j = 0; j < 4; ++j)
    out_row[q_head * head_dim + dim_base + j] =
        __float2half(result[j] / denominator);
}

#endif

[[noreturn]] void paged_error(const std::string& message) {
  throw std::invalid_argument("cuda_paged_gqa_attention_decode: " + message);
}

[[noreturn]] void paged_batch_error(const std::string& message) {
  throw std::invalid_argument("cuda_paged_gqa_attention_decode_batch: " + message);
}

void validate_batch_shapes(const Tensor& q, const Tensor& tables,
                           const Tensor& positions, const Tensor& output,
                           const PagedKvCachePool& pool, std::size_t layer,
                           std::size_t q_heads, std::size_t kv_heads,
                           std::size_t head_dim) {
  if (q.device() != DeviceType::CUDA || q.dtype() != DType::F16 ||
      !q.is_contiguous() || q.shape().size() != 3)
    paged_batch_error("q must be CUDA/F16/contiguous rank-3 [B,q_heads,head_dim]");
  const auto& qs = q.shape();
  const std::size_t batch = static_cast<std::size_t>(qs[0]);
  if (batch != 1 && batch != 2 && batch != 4)
    paged_batch_error("batch must be 1, 2, or 4");
  if (qs[1] != static_cast<int64_t>(q_heads) ||
      qs[2] != static_cast<int64_t>(head_dim))
    paged_batch_error("q shape does not match head parameters");
  if (tables.device() != DeviceType::CUDA || tables.dtype() != DType::I32 ||
      !tables.is_contiguous() || tables.shape().size() != 2 ||
      tables.shape()[0] != static_cast<int64_t>(batch) || tables.shape()[1] <= 0)
    paged_batch_error("block tables must be CUDA/I32/contiguous [B,max_blocks]");
  if (positions.device() != DeviceType::CUDA || positions.dtype() != DType::I32 ||
      !positions.is_contiguous() || positions.shape() !=
          std::vector<int64_t>{static_cast<int64_t>(batch)})
    paged_batch_error("positions must be CUDA/I32/contiguous [B]");
  if (output.device() != DeviceType::CUDA || output.dtype() != DType::F16 ||
      !output.is_contiguous() || output.shape() != q.shape())
    paged_batch_error("output must be CUDA/F16/contiguous with q shape");

  const auto& config = pool.config();
  if (config.dtype != DType::F16 || layer >= config.num_layers)
    paged_batch_error("layer is out of range or pool is not F16");
  if (q_heads == 0 || kv_heads == 0 || head_dim == 0 ||
      kv_heads != config.num_kv_heads || head_dim != config.head_dim ||
      q_heads % kv_heads != 0)
    paged_batch_error("invalid head counts/dimension or GQA ratio");
  if (static_cast<std::size_t>(tables.shape()[1]) >
      std::numeric_limits<std::size_t>::max() / config.block_size)
    paged_batch_error("block table capacity overflows");
}

void validate_batch_metadata(const Tensor& tables, const Tensor& positions,
                             const PagedKvCachePool& pool) {
  const std::size_t batch = static_cast<std::size_t>(tables.shape()[0]);
  const std::size_t max_blocks = static_cast<std::size_t>(tables.shape()[1]);
  std::vector<int32_t> host_positions(batch);
  CUDA_CHECK(cudaMemcpy(host_positions.data(), positions.data(),
                        batch * sizeof(int32_t), cudaMemcpyDeviceToHost));
  std::vector<int32_t> host_tables(batch * max_blocks);
  CUDA_CHECK(cudaMemcpy(host_tables.data(), tables.data(),
                        host_tables.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  for (std::size_t b = 0; b < batch; ++b) {
    if (host_positions[b] < 0 ||
        static_cast<std::size_t>(host_positions[b]) + 1 >
            max_blocks * pool.config().block_size)
      paged_batch_error("position is outside block-table capacity at row=" +
                        std::to_string(b));
    const std::size_t required =
        (static_cast<std::size_t>(host_positions[b]) + 1 + pool.config().block_size - 1) /
        pool.config().block_size;
    for (std::size_t i = 0; i < required; ++i) {
      const int32_t id = host_tables[b * max_blocks + i];
      if (id < 0 || static_cast<std::size_t>(id) >= pool.config().total_blocks ||
          !pool.block_manager().is_allocated(static_cast<BlockId>(id)))
        paged_batch_error("block table contains invalid/free BlockId at row=" +
                          std::to_string(b) + ", index=" + std::to_string(i));
    }
  }
}

}  // namespace

static void paged_gqa_attention_decode_out_impl(
    const Tensor& q, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& device_block_table_i32, std::size_t kv_length,
    std::size_t num_q_heads, std::size_t num_kv_heads, std::size_t head_dim,
    Tensor& output, bool validate_metadata,
    const Tensor* device_cache_length_i32 = nullptr,
    cudaStream_t stream = 0) {
  if (q.device() != DeviceType::CUDA || q.dtype() != DType::F16 ||
      !q.is_contiguous() || q.shape() != std::vector<int64_t>{
          static_cast<int64_t>(num_q_heads), static_cast<int64_t>(head_dim)})
    paged_error("q must be CUDA/F16/contiguous [num_q_heads, head_dim]");
  if (device_block_table_i32.device() != DeviceType::CUDA ||
      device_block_table_i32.dtype() != DType::I32 ||
      !device_block_table_i32.is_contiguous() ||
      device_block_table_i32.shape().size() != 1 ||
      device_block_table_i32.shape()[0] <= 0)
    paged_error("device block table must be CUDA/I32/contiguous non-empty rank-1");
  const auto& config = pool.config();
  if (config.dtype != DType::F16 || layer >= config.num_layers)
    paged_error("layer is out of range or pool is not F16");
  if (num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0 ||
      num_kv_heads != config.num_kv_heads || head_dim != config.head_dim ||
      num_q_heads % num_kv_heads != 0)
    paged_error("invalid head counts/dimension or GQA ratio");
  if (kv_length == 0)
    if (device_cache_length_i32 == nullptr)
      paged_error("kv_length must be greater than zero");
  if (device_cache_length_i32 != nullptr &&
      (device_cache_length_i32->device() != DeviceType::CUDA ||
       device_cache_length_i32->dtype() != DType::I32 ||
       !device_cache_length_i32->is_contiguous() ||
       device_cache_length_i32->shape() != std::vector<int64_t>{1}))
    paged_error("device cache length must be CUDA/I32/contiguous [1]");
  const std::size_t table_blocks = static_cast<std::size_t>(device_block_table_i32.shape()[0]);
  if (device_cache_length_i32 == nullptr && kv_length > table_blocks * config.block_size)
    paged_error("kv_length exceeds device block table capacity");

  if (validate_metadata) {
    std::vector<int32_t> host_table(table_blocks);
    CUDA_CHECK(cudaMemcpy(host_table.data(), device_block_table_i32.data(),
                          table_blocks * sizeof(int32_t), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < table_blocks; ++i) {
      const int32_t id = host_table[i];
      if (id < 0 || static_cast<std::size_t>(id) >= config.total_blocks ||
          !pool.block_manager().is_allocated(static_cast<BlockId>(id)))
        paged_error("device block table contains an invalid or free BlockId at index " +
                    std::to_string(i));
    }
  }

  if (output.device() != DeviceType::CUDA || output.dtype() != DType::F16 ||
      !output.is_contiguous() || output.shape() !=
          std::vector<int64_t>{static_cast<int64_t>(num_q_heads),
                               static_cast<int64_t>(head_dim)})
    paged_error("output must be CUDA/F16/contiguous [num_q_heads, head_dim]");
  const std::size_t blocks = num_q_heads;
  const unsigned int shared_bytes = g_cuda_attention_mode == CudaAttentionMode::kFlashOnline
      ? static_cast<unsigned int>((head_dim + 2 * 32 * head_dim) * sizeof(__half) + 37 * sizeof(float))
      : static_cast<unsigned int>(2 * head_dim * sizeof(__half));
  if (g_cuda_attention_mode == CudaAttentionMode::kFlashOnline)
    paged_gqa_decode_flash_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(head_dim), shared_bytes, stream>>>(
        static_cast<const __half*>(q.data()), static_cast<const __half*>(pool.storage().data()),
        static_cast<const int32_t*>(device_block_table_i32.data()), static_cast<__half*>(output.data()), layer,
        config.total_blocks, config.block_size, num_kv_heads, head_dim, kv_length,
        device_cache_length_i32 == nullptr ? nullptr :
            static_cast<const int32_t*>(device_cache_length_i32->data()), num_q_heads,
        1.0f / sqrtf(static_cast<float>(head_dim)));
  else
    paged_gqa_decode_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(head_dim), shared_bytes, stream>>>(
        static_cast<const __half*>(q.data()), static_cast<const __half*>(pool.storage().data()),
        static_cast<const int32_t*>(device_block_table_i32.data()), static_cast<__half*>(output.data()), layer,
        config.total_blocks, config.block_size, num_kv_heads, head_dim, kv_length,
        device_cache_length_i32 == nullptr ? nullptr :
            static_cast<const int32_t*>(device_cache_length_i32->data()), num_q_heads,
        1.0f / sqrtf(static_cast<float>(head_dim)));
  CUDA_KERNEL_CHECK();
}

void cuda_paged_gqa_attention_decode_out(
    const Tensor& q, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& device_block_table_i32, std::size_t kv_length,
    std::size_t num_q_heads, std::size_t num_kv_heads, std::size_t head_dim,
    Tensor& output) {
  paged_gqa_attention_decode_out_impl(
      q, pool, layer, device_block_table_i32, kv_length, num_q_heads,
      num_kv_heads, head_dim, output, true);
}

void cuda_paged_gqa_attention_decode_out_unchecked(
    const Tensor& q, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& device_block_table_i32, std::size_t kv_length,
    std::size_t num_q_heads, std::size_t num_kv_heads, std::size_t head_dim,
    Tensor& output) {
  paged_gqa_attention_decode_out_impl(
      q, pool, layer, device_block_table_i32, kv_length, num_q_heads,
      num_kv_heads, head_dim, output, false);
}

void cuda_paged_gqa_attention_decode_out_device_length(
    const Tensor& q, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& device_block_table_i32, const Tensor& device_cache_length_i32,
    std::size_t num_q_heads, std::size_t num_kv_heads, std::size_t head_dim,
    Tensor& output, cudaStream_t stream) {
  paged_gqa_attention_decode_out_impl(
      q, pool, layer, device_block_table_i32, 1, num_q_heads, num_kv_heads,
      head_dim, output, false, &device_cache_length_i32, stream);
}

void cuda_paged_kv_write_decode(
    const Tensor& key, const Tensor& value, PagedKvCachePool& pool,
    std::size_t layer, const Tensor& device_block_table_i32,
    const Tensor& device_cache_length_i32, std::size_t num_kv_heads,
    std::size_t head_dim, cudaStream_t stream) {
  const auto& config = pool.config();
  if (key.device() != DeviceType::CUDA || value.device() != DeviceType::CUDA ||
      key.dtype() != DType::F16 || value.dtype() != DType::F16 ||
      !key.is_contiguous() || !value.is_contiguous() ||
      key.shape() != std::vector<int64_t>{static_cast<int64_t>(num_kv_heads),
                                          static_cast<int64_t>(head_dim)} ||
      value.shape() != key.shape())
    paged_error("key/value must be CUDA/F16/contiguous [kv_heads,head_dim]");
  if (device_block_table_i32.device() != DeviceType::CUDA ||
      device_block_table_i32.dtype() != DType::I32 ||
      !device_block_table_i32.is_contiguous() ||
      device_block_table_i32.shape().size() != 1 ||
      device_block_table_i32.shape()[0] <= 0)
    paged_error("device block table must be CUDA/I32/contiguous non-empty rank-1");
  if (device_cache_length_i32.device() != DeviceType::CUDA ||
      device_cache_length_i32.dtype() != DType::I32 ||
      !device_cache_length_i32.is_contiguous() ||
      device_cache_length_i32.shape() != std::vector<int64_t>{1})
    paged_error("device cache length must be CUDA/I32/contiguous [1]");
  if (config.dtype != DType::F16 || layer >= config.num_layers ||
      num_kv_heads != config.num_kv_heads || head_dim != config.head_dim)
    paged_error("invalid KV cache dimensions");
  const std::size_t elements = num_kv_heads * head_dim;
  paged_kv_write_decode_kernel<<<static_cast<unsigned int>((elements + 255) / 256),
                                 256, 0, stream>>>(
      static_cast<const __half*>(key.data()), static_cast<const __half*>(value.data()),
      static_cast<__half*>(pool.storage().data()),
      static_cast<const int32_t*>(device_block_table_i32.data()),
      static_cast<const int32_t*>(device_cache_length_i32.data()), layer,
      config.total_blocks, config.block_size, num_kv_heads, head_dim);
  CUDA_KERNEL_CHECK();
}

void cuda_paged_kv_write_decode_batch(
    const Tensor& key_bhd, const Tensor& value_bhd, PagedKvCachePool& pool,
    std::size_t layer, const Tensor& block_tables_bm_i32,
    const Tensor& positions_before_append_b_i32, std::size_t num_kv_heads,
    std::size_t head_dim) {
  const auto& config = pool.config();
  if (key_bhd.device() != DeviceType::CUDA || value_bhd.device() != DeviceType::CUDA ||
      key_bhd.dtype() != DType::F16 || value_bhd.dtype() != DType::F16 ||
      !key_bhd.is_contiguous() || !value_bhd.is_contiguous() ||
      key_bhd.shape() != value_bhd.shape() || key_bhd.shape().size() != 3 ||
      key_bhd.shape()[1] != static_cast<int64_t>(num_kv_heads) ||
      key_bhd.shape()[2] != static_cast<int64_t>(head_dim))
    paged_error("key/value must be CUDA/F16/contiguous [B,kv_heads,head_dim]");
  if (block_tables_bm_i32.device() != DeviceType::CUDA ||
      block_tables_bm_i32.dtype() != DType::I32 ||
      !block_tables_bm_i32.is_contiguous() || block_tables_bm_i32.shape().size() != 2 ||
      block_tables_bm_i32.shape()[0] != key_bhd.shape()[0] ||
      block_tables_bm_i32.shape()[1] <= 0)
    paged_error("block tables must be CUDA/I32/contiguous [B,max_blocks]");
  if (positions_before_append_b_i32.device() != DeviceType::CUDA ||
      positions_before_append_b_i32.dtype() != DType::I32 ||
      !positions_before_append_b_i32.is_contiguous() ||
      positions_before_append_b_i32.shape() !=
          std::vector<int64_t>{key_bhd.shape()[0]})
    paged_error("positions must be CUDA/I32/contiguous [B]");
  if (config.dtype != DType::F16 || layer >= config.num_layers ||
      num_kv_heads != config.num_kv_heads || head_dim != config.head_dim)
    paged_error("invalid KV cache dimensions");
  const std::size_t batch = static_cast<std::size_t>(key_bhd.shape()[0]);
  const std::size_t elements = batch * num_kv_heads * head_dim;
  paged_kv_write_decode_batch_kernel<<<static_cast<unsigned int>((elements + 255) / 256),
                                       256>>>(
      static_cast<const __half*>(key_bhd.data()),
      static_cast<const __half*>(value_bhd.data()),
      static_cast<__half*>(pool.storage().data()),
      static_cast<const int32_t*>(block_tables_bm_i32.data()),
      static_cast<const int32_t*>(positions_before_append_b_i32.data()), layer,
      config.total_blocks, config.block_size, num_kv_heads, head_dim,
      static_cast<std::size_t>(block_tables_bm_i32.shape()[1]), batch);
  CUDA_KERNEL_CHECK();
}

Tensor cuda_paged_gqa_attention_decode(
    const Tensor& q, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& device_block_table_i32, std::size_t kv_length,
    std::size_t num_q_heads, std::size_t num_kv_heads, std::size_t head_dim) {
  Tensor output(DType::F16,
                {static_cast<int64_t>(num_q_heads), static_cast<int64_t>(head_dim)},
                DeviceType::CUDA);
  cuda_paged_gqa_attention_decode_out(
      q, pool, layer, device_block_table_i32, kv_length, num_q_heads,
      num_kv_heads, head_dim, output);
  return output;
}

void cuda_paged_gqa_attention_decode_batch_out(
    const Tensor& q_bhd, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& block_tables_bm_i32,
    const Tensor& positions_before_append_b_i32,
    std::size_t num_q_heads, std::size_t num_kv_heads, std::size_t head_dim,
    Tensor& output_bhd) {
  validate_batch_shapes(q_bhd, block_tables_bm_i32,
                        positions_before_append_b_i32, output_bhd, pool, layer,
                        num_q_heads, num_kv_heads, head_dim);
  const auto& config = pool.config();
  const std::size_t batch = static_cast<std::size_t>(q_bhd.shape()[0]);
  const std::size_t max_blocks = static_cast<std::size_t>(block_tables_bm_i32.shape()[1]);
  g_paged_batch_attention_launches.fetch_add(1, std::memory_order_relaxed);
  const unsigned int shared_bytes = g_cuda_attention_mode == CudaAttentionMode::kFlashOnline
      ? static_cast<unsigned int>((head_dim + 2 * 32 * head_dim) * sizeof(__half) + 37 * sizeof(float))
      : static_cast<unsigned int>(2 * head_dim * sizeof(__half));
  if (g_cuda_attention_mode == CudaAttentionMode::kFlashOnline)
    paged_gqa_decode_flash_batch_kernel<<<static_cast<unsigned int>(batch * num_q_heads),
                                          static_cast<unsigned int>(head_dim), shared_bytes>>>(
      static_cast<const __half*>(q_bhd.data()), static_cast<const __half*>(pool.storage().data()),
      static_cast<const int32_t*>(block_tables_bm_i32.data()), static_cast<const int32_t*>(positions_before_append_b_i32.data()),
      static_cast<__half*>(output_bhd.data()), layer, config.total_blocks, config.block_size,
      num_kv_heads, head_dim, num_q_heads, max_blocks, 1.0f / sqrtf(static_cast<float>(head_dim)));
  else
    paged_gqa_decode_batch_kernel<<<static_cast<unsigned int>(batch * num_q_heads),
                                    static_cast<unsigned int>(head_dim), shared_bytes>>>(
      static_cast<const __half*>(q_bhd.data()),
      static_cast<const __half*>(pool.storage().data()),
      static_cast<const int32_t*>(block_tables_bm_i32.data()),
      static_cast<const int32_t*>(positions_before_append_b_i32.data()),
      static_cast<__half*>(output_bhd.data()), layer, config.total_blocks,
      config.block_size, num_kv_heads, head_dim, num_q_heads, max_blocks,
      1.0f / sqrtf(static_cast<float>(head_dim)));
  CUDA_KERNEL_CHECK();
}

Tensor cuda_paged_gqa_attention_decode(
    const Tensor& q, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& device_block_table_i32, std::size_t kv_length,
    const AttentionConfig& config) {
  if (!config.is_gqa() || config.head_dim == 0)
    paged_error("AttentionConfig has invalid head configuration");
  if (config.page_size != 0 && config.page_size != pool.config().block_size)
    paged_error("AttentionConfig page_size does not match KV pool block size");
  return cuda_paged_gqa_attention_decode(
      q, pool, layer, device_block_table_i32, kv_length,
      config.num_q_heads, config.num_kv_heads, config.head_dim);
}

void cuda_paged_gqa_attention_decode_batch_out(
    const Tensor& q_bhd, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& block_tables_bm_i32,
    const Tensor& positions_before_append_b_i32,
    const AttentionConfig& config, Tensor& output_bhd) {
  if (!config.is_gqa() || config.head_dim == 0)
    paged_error("AttentionConfig has invalid head configuration");
  if (config.page_size != 0 && config.page_size != pool.config().block_size)
    paged_error("AttentionConfig page_size does not match KV pool block size");
  cuda_paged_gqa_attention_decode_batch_out(
      q_bhd, pool, layer, block_tables_bm_i32, positions_before_append_b_i32,
      config.num_q_heads, config.num_kv_heads, config.head_dim, output_bhd);
}

Tensor cuda_paged_gqa_attention_decode_batch(
    const Tensor& q_bhd, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& block_tables_bm_i32,
    const Tensor& positions_before_append_b_i32,
    const AttentionConfig& config) {
  if (!config.is_gqa() || config.head_dim == 0)
    paged_error("AttentionConfig has invalid head configuration");
  if (config.page_size != 0 && config.page_size != pool.config().block_size)
    paged_error("AttentionConfig page_size does not match KV pool block size");
  return cuda_paged_gqa_attention_decode_batch(
      q_bhd, pool, layer, block_tables_bm_i32, positions_before_append_b_i32,
      config.num_q_heads, config.num_kv_heads, config.head_dim);
}

void reset_cuda_paged_gqa_attention_decode_batch_launch_count() {
  g_paged_batch_attention_launches.store(0, std::memory_order_relaxed);
}

std::uint64_t cuda_paged_gqa_attention_decode_batch_launch_count() {
  return g_paged_batch_attention_launches.load(std::memory_order_relaxed);
}

Tensor cuda_paged_gqa_attention_decode_batch(
    const Tensor& q_bhd, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& block_tables_bm_i32,
    const Tensor& positions_before_append_b_i32,
    std::size_t num_q_heads, std::size_t num_kv_heads, std::size_t head_dim) {
  Tensor output(DType::F16, q_bhd.shape(), DeviceType::CUDA);
  validate_batch_shapes(q_bhd, block_tables_bm_i32,
                        positions_before_append_b_i32, output, pool, layer,
                        num_q_heads, num_kv_heads, head_dim);
  validate_batch_metadata(block_tables_bm_i32, positions_before_append_b_i32, pool);
  cuda_paged_gqa_attention_decode_batch_out(
      q_bhd, pool, layer, block_tables_bm_i32, positions_before_append_b_i32,
      num_q_heads, num_kv_heads, head_dim, output);
  return output;
}

}  // namespace llm
