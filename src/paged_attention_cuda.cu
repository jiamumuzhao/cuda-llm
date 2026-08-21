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
    std::size_t kv_length, std::size_t q_heads, float scale) {
  const std::size_t q_head = blockIdx.x;
  const std::size_t dim = threadIdx.x;
  if (q_head >= q_heads || dim >= head_dim) return;
  __shared__ float score_workspace[32];

  const std::size_t group = q_heads / kv_heads;
  const std::size_t kv_head = q_head / group;
  const std::size_t token_stride = kv_heads * head_dim;
  const std::size_t plane_elements = block_size * token_stride;
  const float query = __half2float(q[q_head * head_dim + dim]);

  float maximum = -CUDART_INF_F;
  for (std::size_t token = 0; token < kv_length; ++token) {
    const std::size_t block = static_cast<std::size_t>(block_table[token / block_size]);
    const std::size_t offset = token % block_size;
    const std::size_t layer_block = (layer * total_blocks + block) * 2;
    const __half* key = storage + layer_block * plane_elements +
                        offset * token_stride + kv_head * head_dim;
    float partial = 0.0f;
    for (std::size_t e = dim; e < head_dim; e += blockDim.x)
      partial += __half2float(q[q_head * head_dim + e]) * __half2float(key[e]);
    for (int offset = 16; offset > 0; offset >>= 1)
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    const std::size_t warp = dim / 32;
    const int lane = static_cast<int>(dim % 32);
    if (lane == 0) score_workspace[warp] = partial;
    __syncthreads();
    if (dim == 0) {
      float total = score_workspace[0];
      for (std::size_t w = 1; w < (head_dim + 31) / 32; ++w)
        total += score_workspace[w];
      score_workspace[0] = total;
    }
    __syncthreads();
    maximum = fmaxf(maximum, score_workspace[0] * scale);
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
    float partial = 0.0f;
    for (std::size_t e = dim; e < head_dim; e += blockDim.x)
      partial += __half2float(q[q_head * head_dim + e]) * __half2float(key[e]);
    for (int offset = 16; offset > 0; offset >>= 1)
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    const std::size_t warp = dim / 32;
    const int lane = static_cast<int>(dim % 32);
    if (lane == 0) score_workspace[warp] = partial;
    __syncthreads();
    if (dim == 0) {
      float total = score_workspace[0];
      for (std::size_t w = 1; w < (head_dim + 31) / 32; ++w)
        total += score_workspace[w];
      score_workspace[0] = total;
    }
    __syncthreads();
    const float probability = expf(score_workspace[0] * scale - maximum);
    denominator += probability;
    result += probability * __half2float(value[dim]);
  }
  output[q_head * head_dim + dim] = __float2half(result / denominator);
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
  __shared__ float score_workspace[32];

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

  float maximum = -CUDART_INF_F;
  for (std::size_t token = 0; token < kv_length; ++token) {
    const std::size_t block = static_cast<std::size_t>(table[token / block_size]);
    const std::size_t offset = token % block_size;
    const std::size_t layer_block = (layer * total_blocks + block) * 2;
    const __half* key = storage + layer_block * plane_elements +
                        offset * token_stride + kv_head * head_dim;
    float partial = 0.0f;
    for (std::size_t e = dim; e < head_dim; e += blockDim.x)
      partial += __half2float(q_row[q_head * head_dim + e]) * __half2float(key[e]);
    for (int offset = 16; offset > 0; offset >>= 1)
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    const std::size_t warp = dim / 32;
    const int lane = static_cast<int>(dim % 32);
    if (lane == 0) score_workspace[warp] = partial;
    __syncthreads();
    if (dim == 0) {
      float total = score_workspace[0];
      for (std::size_t w = 1; w < (head_dim + 31) / 32; ++w)
        total += score_workspace[w];
      score_workspace[0] = total;
    }
    __syncthreads();
    maximum = fmaxf(maximum, score_workspace[0] * scale);
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
    float partial = 0.0f;
    for (std::size_t e = dim; e < head_dim; e += blockDim.x)
      partial += __half2float(q_row[q_head * head_dim + e]) * __half2float(key[e]);
    for (int offset = 16; offset > 0; offset >>= 1)
      partial += __shfl_down_sync(0xffffffff, partial, offset);
    const std::size_t warp = dim / 32;
    const int lane = static_cast<int>(dim % 32);
    if (lane == 0) score_workspace[warp] = partial;
    __syncthreads();
    if (dim == 0) {
      float total = score_workspace[0];
      for (std::size_t w = 1; w < (head_dim + 31) / 32; ++w)
        total += score_workspace[w];
      score_workspace[0] = total;
    }
    __syncthreads();
    const float probability = expf(score_workspace[0] * scale - maximum);
    denominator += probability;
    result += probability * __half2float(value[dim]);
  }
  out_row[q_head * head_dim + dim] = __float2half(result / denominator);
}

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

Tensor cuda_paged_gqa_attention_decode(
    const Tensor& q, const PagedKvCachePool& pool, std::size_t layer,
    const Tensor& device_block_table_i32, std::size_t kv_length,
    std::size_t num_q_heads, std::size_t num_kv_heads, std::size_t head_dim) {
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
    paged_error("kv_length must be greater than zero");
  const std::size_t table_blocks = static_cast<std::size_t>(device_block_table_i32.shape()[0]);
  if (kv_length > table_blocks * config.block_size)
    paged_error("kv_length exceeds device block table capacity");

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

  Tensor output(DType::F16,
                {static_cast<int64_t>(num_q_heads), static_cast<int64_t>(head_dim)},
                DeviceType::CUDA);
  const std::size_t blocks = num_q_heads;
  paged_gqa_decode_kernel<<<static_cast<unsigned int>(blocks),
                            static_cast<unsigned int>(head_dim)>>>(
      static_cast<const __half*>(q.data()),
      static_cast<const __half*>(pool.storage().data()),
      static_cast<const int32_t*>(device_block_table_i32.data()),
      static_cast<__half*>(output.data()), layer, config.total_blocks,
      config.block_size, num_kv_heads, head_dim, kv_length, num_q_heads,
      1.0f / sqrtf(static_cast<float>(head_dim)));
  CUDA_KERNEL_CHECK();
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
  paged_gqa_decode_batch_kernel<<<static_cast<unsigned int>(batch * num_q_heads),
                                  static_cast<unsigned int>(head_dim)>>>(
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
