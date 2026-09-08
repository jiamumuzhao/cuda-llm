#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_model_adapter.h"

#include "llm/model_package.h"
#include "llm/ops_cuda.h"
#include "llm/cuda_check.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace llm {

struct PagedDecodeGraphState {
  Qwen3PagedKvCache* cache = nullptr;
  PagedKvCachePool* pool = nullptr;
  std::size_t capacity = 0;
  CudaAttentionMode attention_mode = CudaAttentionMode::kReference;
  Tensor block_table;
  Tensor logits;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;

  ~PagedDecodeGraphState() noexcept {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
};

struct PagedBatchDecodeGraphState {
  std::vector<Qwen3PagedKvCache*> caches;
  PagedKvCachePool* pool = nullptr;
  std::vector<std::size_t> capacities;
  CudaAttentionMode attention_mode = CudaAttentionMode::kReference;
  Tensor block_tables;
  Tensor logits;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;

  ~PagedBatchDecodeGraphState() noexcept {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
  }
};

namespace {

std::size_t g_paged_prefill_fault_layer = static_cast<std::size_t>(-1);

[[noreturn]] void model_error(const std::string& message) {
  throw std::runtime_error("Qwen3CudaModel: " + message);
}

Tensor f32_to_f16_cpu(const Tensor& input, const std::string& name) {
  if (input.device() != DeviceType::CPU || input.dtype() != DType::F32 ||
      !input.is_contiguous()) {
    model_error(name + " must be CPU/F32/contiguous at load boundary");
  }
  Tensor output(DType::F16, input.shape(), DeviceType::CPU);
  for (size_t i = 0; i < input.numel(); ++i) output.set_f32(i, input.get_f32(i));
  return output;
}

Tensor load_f16_cuda(const ModelPackage& package, const std::string& name,
                     const std::vector<int64_t>& expected_shape) {
  Tensor cpu = package.load_tensor(name);
  if (cpu.shape() != expected_shape) {
    std::ostringstream message;
    message << name << " shape actual=[";
    for (size_t i = 0; i < cpu.shape().size(); ++i) {
      if (i) message << ",";
      message << cpu.shape()[i];
    }
    message << "] expected=[";
    for (size_t i = 0; i < expected_shape.size(); ++i) {
      if (i) message << ",";
      message << expected_shape[i];
    }
    message << "]";
    model_error(message.str());
  }
  return f32_to_f16_cpu(cpu, name).to(DeviceType::CUDA);
}

std::vector<DecoderKvCache*> make_decoder_caches(
    const std::vector<Qwen3KvCache*>& caches) {
  std::vector<DecoderKvCache*> result;
  result.reserve(caches.size());
  for (Qwen3KvCache* cache : caches) result.push_back(cache);
  return result;
}

std::vector<DecoderPagedKvCache*> make_decoder_paged_caches(
    const std::vector<Qwen3PagedKvCache*>& caches) {
  std::vector<DecoderPagedKvCache*> result;
  result.reserve(caches.size());
  for (Qwen3PagedKvCache* cache : caches) result.push_back(cache);
  return result;
}

void validate_contiguous_cache(const Qwen3KvCache& cache,
                               const DecoderModelSpec& spec,
                               const char* operation) {
  if (cache.num_layers() != spec.num_layers ||
      cache.num_kv_heads() != spec.attention.num_kv_heads ||
      cache.head_dim() != spec.attention.head_dim) {
    model_error(std::string(operation) + " cache configuration does not match model");
  }
}

void validate_paged_cache_pool(const Qwen3PagedKvCache& cache,
                               const DecoderModelSpec& spec,
                               const char* operation) {
  const auto& config = cache.pool().config();
  if (config.num_layers != spec.num_layers ||
      config.num_kv_heads != spec.attention.num_kv_heads ||
      config.head_dim != spec.attention.head_dim ||
      config.dtype != DType::F16 || config.block_size == 0) {
    model_error(std::string(operation) +
                " paged cache pool configuration does not match model");
  }
}

Qwen3CudaLayerWeights load_layer(
    const ModelPackage& package, size_t index, const DecoderModelSpec& spec) {
  const std::string prefix = "layers." + std::to_string(index) + ".";
  const auto hidden = static_cast<int64_t>(spec.hidden_size);
  const auto q_dim = static_cast<int64_t>(
      spec.attention.num_q_heads * spec.attention.head_dim);
  const auto kv_dim = static_cast<int64_t>(
      spec.attention.num_kv_heads * spec.attention.head_dim);
  const auto head_dim = static_cast<int64_t>(spec.attention.head_dim);
  const auto intermediate = static_cast<int64_t>(spec.intermediate_size);
  return {
      load_f16_cuda(package, prefix + "input_norm", {hidden}),
      load_f16_cuda(package, prefix + "q_proj", {q_dim, hidden}),
      load_f16_cuda(package, prefix + "k_proj", {kv_dim, hidden}),
      load_f16_cuda(package, prefix + "v_proj", {kv_dim, hidden}),
      load_f16_cuda(package, prefix + "q_norm", {head_dim}),
      load_f16_cuda(package, prefix + "k_norm", {head_dim}),
      load_f16_cuda(package, prefix + "o_proj", {hidden, q_dim}),
      load_f16_cuda(package, prefix + "post_attention_norm", {hidden}),
      load_f16_cuda(package, prefix + "gate_proj", {intermediate, hidden}),
      load_f16_cuda(package, prefix + "up_proj", {intermediate, hidden}),
      load_f16_cuda(package, prefix + "down_proj", {hidden, intermediate})};
}

void validate_runtime_inputs(const std::vector<int32_t>& token_ids,
                             const std::vector<int32_t>& position_ids,
                             size_t layer_count) {
  if (token_ids.empty()) model_error("prefill requires non-empty token_ids");
  if (token_ids.size() != position_ids.size()) {
    model_error("prefill position_ids length actual=" +
                std::to_string(position_ids.size()) + " expected=" +
                std::to_string(token_ids.size()));
  }
  if (token_ids.size() < 1 || token_ids.size() > 32) {
    model_error("prefill seq_len actual=" + std::to_string(token_ids.size()) +
                " expected in [1,32]");
  }
  if (layer_count == 0 || layer_count > 28) {
    model_error("layer_count actual=" + std::to_string(layer_count) +
                " expected in [1,28]");
  }
  for (size_t i = 0; i < token_ids.size(); ++i) {
    if (token_ids[i] < 0 || token_ids[i] >= 151936) {
      model_error("token_ids[" + std::to_string(i) + "] actual=" +
                  std::to_string(token_ids[i]) + " expected in [0,151936)");
    }
  }
}

PagedDecodeMetadataUploadStats metadata_upload_delta(
    const CudaAllocationStats& before, const CudaAllocationStats& after,
    size_t positions_uploads, size_t block_table_uploads, bool valid) {
  PagedDecodeMetadataUploadStats result;
  result.valid = valid && after.cuda_malloc_calls >= before.cuda_malloc_calls &&
                 after.cuda_free_calls >= before.cuda_free_calls &&
                 after.cuda_allocated_bytes_total >= before.cuda_allocated_bytes_total &&
                 after.cuda_freed_bytes_total >= before.cuda_freed_bytes_total;
  if (!result.valid) return result;
  result.cuda_malloc_calls = after.cuda_malloc_calls - before.cuda_malloc_calls;
  result.cuda_free_calls = after.cuda_free_calls - before.cuda_free_calls;
  result.cuda_allocated_bytes =
      after.cuda_allocated_bytes_total - before.cuda_allocated_bytes_total;
  result.cuda_freed_bytes =
      after.cuda_freed_bytes_total - before.cuda_freed_bytes_total;
  result.positions_uploads = positions_uploads;
  result.block_table_uploads = block_table_uploads;
  return result;
}

}  // namespace

DecodeWorkspace::DecodeWorkspace(const DecoderModelSpec& spec)
    : hidden_a(DType::F16, {4, static_cast<int64_t>(spec.hidden_size)}, DeviceType::CUDA),
      hidden_b(DType::F16, {4, static_cast<int64_t>(spec.hidden_size)}, DeviceType::CUDA),
      input_norm(DType::F16, {4, static_cast<int64_t>(spec.hidden_size)}, DeviceType::CUDA),
      q_linear(DType::F16, {4, static_cast<int64_t>(spec.attention.num_q_heads * spec.attention.head_dim)}, DeviceType::CUDA),
      k_linear(DType::F16, {4, static_cast<int64_t>(spec.attention.num_kv_heads * spec.attention.head_dim)}, DeviceType::CUDA),
      v_linear(DType::F16, {4, static_cast<int64_t>(spec.attention.num_kv_heads * spec.attention.head_dim)}, DeviceType::CUDA),
      q_norm(DType::F16, {4, static_cast<int64_t>(spec.attention.num_q_heads * spec.attention.head_dim)}, DeviceType::CUDA),
      k_norm(DType::F16, {4, static_cast<int64_t>(spec.attention.num_kv_heads * spec.attention.head_dim)}, DeviceType::CUDA),
      attention(DType::F16, {4, static_cast<int64_t>(spec.attention.num_q_heads * spec.attention.head_dim)}, DeviceType::CUDA),
      o_proj(DType::F16, {4, static_cast<int64_t>(spec.hidden_size)}, DeviceType::CUDA),
      attention_residual(DType::F16, {4, static_cast<int64_t>(spec.hidden_size)}, DeviceType::CUDA),
      post_attention_norm(DType::F16, {4, static_cast<int64_t>(spec.hidden_size)}, DeviceType::CUDA),
      gate(DType::F16, {4, static_cast<int64_t>(spec.intermediate_size)}, DeviceType::CUDA),
      up(DType::F16, {4, static_cast<int64_t>(spec.intermediate_size)}, DeviceType::CUDA),
      down(DType::F16, {4, static_cast<int64_t>(spec.hidden_size)}, DeviceType::CUDA),
      final_norm(DType::F16, {4, static_cast<int64_t>(spec.hidden_size)}, DeviceType::CUDA),
      position_ids(DType::F32, {1}, DeviceType::CUDA),
      variable_position_ids(DType::I32, {4}, DeviceType::CUDA),
      variable_lengths(DType::I32, {4}, DeviceType::CUDA),
      token_ids(DType::F32, {4}, DeviceType::CUDA) {}

DecodeWorkspace::DecodeWorkspace()
    : DecodeWorkspace(DecoderModelSpec{
          AttentionConfig{16, 8, 128, 16, 512, true},
          1024, 3072, 151936, 28, 0.0f, 0.0f, true}) {}

Qwen3CudaModel::Qwen3CudaModel(const std::filesystem::path& package_root)
    : DecoderRuntime(make_qwen3_model_adapter(package_root)) {
  rms_norm_eps_ = model_spec_.rms_norm_eps;
  rope_theta_ = model_spec_.rope_theta;
}

Qwen3CudaModel::~Qwen3CudaModel() = default;

Tensor Qwen3CudaModel::prefill_hidden_layers(
    const std::vector<int32_t>& token_ids,
    const std::vector<int32_t>& position_ids, size_t layer_count) const {
  validate_runtime_inputs(token_ids, position_ids, layer_count);
  Tensor hidden = cuda_embedding_lookup(token_embedding_, token_ids);
  for (size_t i = 0; i < layer_count; ++i) {
    hidden = layer_executor().prefill(
        hidden, position_ids, layers_[i], rms_norm_eps_, rope_theta_);
  }
  return hidden;
}

Tensor Qwen3CudaModel::prefill_final_hidden(
    const std::vector<int32_t>& token_ids,
    const std::vector<int32_t>& position_ids) const {
  Tensor hidden = prefill_hidden_layers(token_ids, position_ids, layers_.size());
  return cuda_rms_norm(hidden, final_norm_, rms_norm_eps_);
}

Tensor Qwen3CudaModel::prefill_logits(
    const std::vector<int32_t>& token_ids,
    const std::vector<int32_t>& position_ids) const {
  Tensor final_hidden = prefill_final_hidden(token_ids, position_ids);
  return cuda_lm_head(final_hidden, token_embedding_);
}

Tensor Qwen3CudaModel::prefill_logits_with_cache(
    const std::vector<int32_t>& prompt_ids, Qwen3KvCache& cache) const {
  if (cache.length() != 0)
    model_error("prefill_logits_with_cache requires empty cache, actual length=" +
                std::to_string(cache.length()));
  validate_contiguous_cache(cache, model_spec_, "prefill");
  if (prompt_ids.empty() || prompt_ids.size() > 32 ||
      prompt_ids.size() > cache.capacity())
    model_error("prompt length actual=" + std::to_string(prompt_ids.size()) +
                " expected in [1,min(32,cache.capacity)]");
  for (size_t i = 0; i < prompt_ids.size(); ++i)
    if (prompt_ids[i] < 0 || prompt_ids[i] >= 151936)
      model_error("prompt_ids[" + std::to_string(i) + "] out of range");
  std::vector<int32_t> positions(prompt_ids.size());
  for (size_t i = 0; i < positions.size(); ++i) positions[i] = static_cast<int32_t>(i);
  Tensor hidden = cuda_embedding_lookup(token_embedding_, prompt_ids);
  for (size_t i = 0; i < layers_.size(); ++i) {
    DecoderLayerOutput trace = layer_executor().prefill_with_trace(
        hidden, positions, layers_[i], rms_norm_eps_, rope_theta_);
    Tensor k = trace.k_rope;
    Tensor v = trace.v_linear.reshape({static_cast<int64_t>(prompt_ids.size()), 8, 128});
    cache.write_prefill_layer(i, k, v, prompt_ids.size());
    hidden = std::move(trace.layer_output);
  }
  cache.commit_prefill(prompt_ids.size());
  return cuda_lm_head(cuda_rms_norm(hidden, final_norm_, rms_norm_eps_),
                      token_embedding_);
}

void qwen3_set_paged_prefill_fault_for_testing(std::size_t layer) {
  g_paged_prefill_fault_layer = layer;
}

void qwen3_clear_paged_prefill_fault_for_testing() {
  g_paged_prefill_fault_layer = static_cast<std::size_t>(-1);
}

Tensor Qwen3CudaModel::prefill_logits_paged(
    const std::vector<int32_t>& prompt_ids, Qwen3PagedKvCache& cache) const {
  // Preserve the established single-request API shape [S, vocab].  Use
  // packed prefill for long contexts because the legacy equal-length path is
  // intentionally capped at 32 tokens.
  CudaLinearModeGuard fast_linear(
      prompt_ids.size() > 32 ? CudaLinearMode::kFastGemm
                             : CudaLinearMode::kReference);
  Tensor batched = prompt_ids.size() > 32
                        ? prefill_logits_paged_packed_batch({prompt_ids}, {&cache})
                        : prefill_logits_paged_batch({prompt_ids}, {&cache});
  return batched.reshape({static_cast<int64_t>(prompt_ids.size()),
                          token_embedding_.shape().at(0)});
}

Tensor Qwen3CudaModel::prefill_logits_paged_batch(
    const std::vector<std::vector<int32_t>>& prompt_ids_batch,
    const std::vector<Qwen3PagedKvCache*>& caches) const {
  const size_t batch = prompt_ids_batch.size();
  if (batch != 1 && batch != 2 && batch != 4)
    model_error("paged batched prefill batch size must be 1, 2, or 4");
  if (caches.size() != batch)
    model_error("paged batched prefill prompt/cache count mismatch");
  const size_t seq = prompt_ids_batch.front().size();
  if (seq == 0 || seq > 32)
    model_error("paged batched prefill seq_len must be in [1,32]");
  PagedKvCachePool* common_pool = nullptr;
  std::vector<Qwen3PagedKvCache*> unique;
  unique.reserve(batch);
  for (size_t b = 0; b < batch; ++b) {
    if (prompt_ids_batch[b].size() != seq)
      model_error("paged batched prefill requires equal prompt lengths");
    for (size_t i = 0; i < seq; ++i)
      if (prompt_ids_batch[b][i] < 0 || prompt_ids_batch[b][i] >= 151936)
        model_error("paged batched prefill token out of range at row=" + std::to_string(b));
    if (!caches[b]) model_error("paged batched prefill null cache at row=" + std::to_string(b));
    if (std::find(unique.begin(), unique.end(), caches[b]) != unique.end())
      model_error("paged batched prefill cache pointers must be unique");
    unique.push_back(caches[b]);
    if (!common_pool) common_pool = &caches[b]->pool();
    if (&caches[b]->pool() != common_pool)
      model_error("paged batched prefill caches must share one PagedKvCachePool");
    validate_paged_cache_pool(*caches[b], model_spec_, "paged batched prefill");
    if (caches[b]->length() != 0 || caches[b]->block_table().size() != 0 ||
        caches[b]->in_decode_transaction() || caches[b]->in_prefill_transaction() ||
        caches[b]->capacity() < seq)
      model_error("paged batched prefill cache must be empty and have capacity >= seq_len");
  }
  std::vector<bool> begun(batch, false);
  try {
    for (size_t b = 0; b < batch; ++b) {
      caches[b]->begin_prefill(seq);
      begun[b] = true;
    }
    std::vector<int32_t> flat_ids;
    flat_ids.reserve(batch * seq);
    for (const auto& ids : prompt_ids_batch)
      flat_ids.insert(flat_ids.end(), ids.begin(), ids.end());
    std::vector<int32_t> positions(seq);
    for (size_t i = 0; i < seq; ++i) positions[i] = static_cast<int32_t>(i);
    Tensor hidden = cuda_embedding_lookup(token_embedding_, flat_ids)
                        .reshape({static_cast<int64_t>(batch), static_cast<int64_t>(seq), 1024});
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      DecoderLayerOutput trace = layer_executor().prefill_batch(
          hidden, positions, layers_[layer], rms_norm_eps_, rope_theta_);
      for (size_t b = 0; b < batch; ++b)
        caches[b]->append_prefill_layer_kv_batched_slice(
            layer, trace.k_rope, trace.v_linear, b, batch, seq);
      if (g_paged_prefill_fault_layer == layer)
        throw std::runtime_error("Qwen3CudaModel: injected direct paged prefill fault after layer=" +
                                 std::to_string(layer));
      hidden = std::move(trace.layer_output);
    }
    Tensor final_hidden = cuda_rms_norm(hidden, final_norm_, rms_norm_eps_);
    Tensor logits = cuda_lm_head(
        final_hidden.reshape({static_cast<int64_t>(batch * seq), 1024}),
        token_embedding_)
        .reshape({static_cast<int64_t>(batch), static_cast<int64_t>(seq), 151936});
    CUDA_CHECK(cudaDeviceSynchronize());
    for (auto* cache : caches) cache->commit_prefill();
    return logits;
  } catch (...) {
    for (size_t b = 0; b < batch; ++b)
      if (begun[b]) caches[b]->abort_prefill();
    throw;
  }
}

Tensor Qwen3CudaModel::prefill_logits_paged_packed_batch(
    const std::vector<std::vector<int32_t>>& prompt_ids_batch,
    const std::vector<Qwen3PagedKvCache*>& caches) const {
  const size_t batch = prompt_ids_batch.size();
  if (batch != 1 && batch != 2 && batch != 4)
    model_error("packed paged prefill batch size must be 1, 2, or 4");
  if (caches.size() != batch)
    model_error("packed paged prefill prompt/cache count mismatch");
  size_t total_tokens = 0;
  std::vector<size_t> offsets(batch + 1, 0);
  for (size_t b = 0; b < batch; ++b) {
    const size_t length = prompt_ids_batch[b].size();
    if (length == 0 || length > 512)
      model_error("packed paged prefill prompt length must be in [1,512]");
    offsets[b + 1] = offsets[b] + length;
    total_tokens += length;
  }
  if (total_tokens > 512)
    model_error("packed paged prefill total token count must be <= 512");
  PagedKvCachePool* common_pool = nullptr;
  std::vector<Qwen3PagedKvCache*> unique;
  unique.reserve(batch);
  for (size_t b = 0; b < batch; ++b) {
    if (!caches[b] || std::find(unique.begin(), unique.end(), caches[b]) != unique.end())
      model_error("packed paged prefill cache pointers must be non-null and unique");
    unique.push_back(caches[b]);
    if (!common_pool) common_pool = &caches[b]->pool();
    if (&caches[b]->pool() != common_pool)
      model_error("packed paged prefill caches must share one PagedKvCachePool");
    validate_paged_cache_pool(*caches[b], model_spec_, "packed paged prefill");
    if (caches[b]->length() != 0 || !caches[b]->block_table().empty() ||
        caches[b]->in_decode_transaction() || caches[b]->in_prefill_transaction() ||
        caches[b]->capacity() < prompt_ids_batch[b].size())
      model_error("packed paged prefill cache must be empty and have capacity >= prompt length");
    for (int32_t id : prompt_ids_batch[b])
      if (id < 0 || id >= 151936)
        model_error("packed paged prefill token out of range");
  }
  std::vector<bool> begun(batch, false);
  std::vector<Tensor> block_tables;
  try {
    for (size_t b = 0; b < batch; ++b) {
      caches[b]->begin_prefill(prompt_ids_batch[b].size());
      begun[b] = true;
    }
    block_tables.reserve(batch);
    for (auto* cache : caches)
      block_tables.push_back(cache->make_device_block_table_i32());
    std::vector<int32_t> flat_ids;
    std::vector<int32_t> positions;
    flat_ids.reserve(total_tokens);
    positions.reserve(total_tokens);
    for (const auto& prompt : prompt_ids_batch)
      for (size_t i = 0; i < prompt.size(); ++i) {
        flat_ids.push_back(prompt[i]);
        positions.push_back(static_cast<int32_t>(i));
      }
    Tensor positions_cuda(DType::I32, {static_cast<int64_t>(total_tokens)},
                          DeviceType::CUDA);
    Tensor offsets_cuda(DType::I32, {static_cast<int64_t>(batch + 1)},
                        DeviceType::CUDA);
    CUDA_CHECK(cudaMemcpy(positions_cuda.data(), positions.data(),
                          positions.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    std::vector<int32_t> offsets_i32(offsets.begin(), offsets.end());
    CUDA_CHECK(cudaMemcpy(offsets_cuda.data(), offsets_i32.data(),
                          offsets_i32.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    Tensor hidden = cuda_embedding_lookup(token_embedding_, flat_ids);
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      DecoderLayerOutput trace = layer_executor().prefill_packed(
          hidden, positions_cuda, offsets_cuda, layers_[layer], rms_norm_eps_,
          rope_theta_);
      for (size_t b = 0; b < batch; ++b)
        caches[b]->append_prefill_layer_kv_packed_slice(
            layer, trace.k_rope, trace.v_linear, offsets[b],
            prompt_ids_batch[b].size(), block_tables[b]);
      if (g_paged_prefill_fault_layer == layer)
        throw std::runtime_error("Qwen3CudaModel: injected packed prefill fault after layer=" +
                                 std::to_string(layer));
      hidden = std::move(trace.layer_output);
    }
    Tensor logits = cuda_lm_head(
        cuda_rms_norm(hidden, final_norm_, rms_norm_eps_), token_embedding_);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (auto* cache : caches) cache->commit_prefill();
    return logits;
  } catch (...) {
    for (size_t b = 0; b < batch; ++b)
      if (begun[b]) caches[b]->abort_prefill();
    throw;
  }
}

Tensor Qwen3CudaModel::prefill_logits_paged_padded_batch(
    const PagedPaddedPrefillBatch& batch) const {
  if (batch.batch_size != 1 && batch.batch_size != 2 && batch.batch_size != 4)
    model_error("paged padded prefill batch_size must be 1, 2, or 4");
  if (batch.caches.size() != batch.batch_size ||
      batch.valid_lengths.size() != batch.batch_size || batch.max_seq_len == 0 ||
      batch.max_seq_len > 32 ||
      batch.padded_token_ids.size() != batch.batch_size * batch.max_seq_len)
    model_error("paged padded prefill metadata shape/count mismatch");

  PagedKvCachePool* common_pool = nullptr;
  std::vector<Qwen3PagedKvCache*> unique;
  unique.reserve(batch.batch_size);
  for (size_t b = 0; b < batch.batch_size; ++b) {
    const size_t valid = batch.valid_lengths[b];
    if (valid == 0 || valid > batch.max_seq_len)
      model_error("paged padded prefill valid length must be in [1,max_seq_len]");
    if (batch.caches[b] == nullptr ||
        std::find(unique.begin(), unique.end(), batch.caches[b]) != unique.end())
      model_error("paged padded prefill cache pointers must be non-null and unique");
    unique.push_back(batch.caches[b]);
    if (!common_pool) common_pool = &batch.caches[b]->pool();
    if (&batch.caches[b]->pool() != common_pool)
      model_error("paged padded prefill caches must share one PagedKvCachePool");
    validate_paged_cache_pool(*batch.caches[b], model_spec_,
                              "paged padded prefill");
    if (batch.caches[b]->length() != 0 || !batch.caches[b]->block_table().empty() ||
        batch.caches[b]->in_decode_transaction() ||
        batch.caches[b]->in_prefill_transaction() || batch.caches[b]->capacity() < valid)
      model_error("paged padded prefill cache must be empty and have capacity >= valid length");
  }
  for (size_t i = 0; i < batch.padded_token_ids.size(); ++i)
    if (batch.padded_token_ids[i] < 0 || batch.padded_token_ids[i] >= 151936)
      model_error("paged padded token id out of range at flat index=" + std::to_string(i));
  for (size_t b = 0; b < batch.batch_size; ++b)
    for (size_t s = batch.valid_lengths[b]; s < batch.max_seq_len; ++s)
      if (batch.padded_token_ids[b * batch.max_seq_len + s] != 0)
        model_error("paged padded right padding token must be 0: row=" +
                    std::to_string(b) + " position=" + std::to_string(s));

  std::vector<bool> begun(batch.batch_size, false);
  try {
    for (size_t b = 0; b < batch.batch_size; ++b) {
      batch.caches[b]->begin_prefill(batch.valid_lengths[b]);
      begun[b] = true;
    }
    Tensor valid_lengths_cuda(DType::I32,
                              {static_cast<int64_t>(batch.batch_size)},
                              DeviceType::CUDA);
    std::vector<int32_t> valid_lengths_host(batch.batch_size);
    for (size_t b = 0; b < batch.batch_size; ++b)
      valid_lengths_host[b] = static_cast<int32_t>(batch.valid_lengths[b]);
    CUDA_CHECK(cudaMemcpy(valid_lengths_cuda.data(), valid_lengths_host.data(),
                          valid_lengths_host.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    record_valid_length_h2d_upload();
    std::vector<int32_t> positions(batch.max_seq_len);
    for (size_t i = 0; i < positions.size(); ++i) positions[i] = static_cast<int32_t>(i);
    Tensor hidden = cuda_embedding_lookup(token_embedding_, batch.padded_token_ids)
                        .reshape({static_cast<int64_t>(batch.batch_size),
                                  static_cast<int64_t>(batch.max_seq_len), 1024});
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      DecoderLayerOutput trace = layer_executor().prefill_batch_valid_lengths(
          hidden, positions, valid_lengths_cuda, layers_[layer], rms_norm_eps_, rope_theta_);
      for (size_t b = 0; b < batch.batch_size; ++b)
        batch.caches[b]->append_prefill_layer_kv_batched_valid_slice(
            layer, trace.k_rope, trace.v_linear, b, batch.batch_size,
            batch.max_seq_len, batch.valid_lengths[b]);
      if (g_paged_prefill_fault_layer == layer)
        throw std::runtime_error("Qwen3CudaModel: injected direct paged prefill fault after layer=" +
                                 std::to_string(layer));
      hidden = std::move(trace.layer_output);
    }
    Tensor logits = cuda_lm_head(
        cuda_rms_norm(hidden, final_norm_, rms_norm_eps_)
            .reshape({static_cast<int64_t>(batch.batch_size * batch.max_seq_len), 1024}),
        token_embedding_)
        .reshape({static_cast<int64_t>(batch.batch_size),
                  static_cast<int64_t>(batch.max_seq_len), 151936});
    CUDA_CHECK(cudaDeviceSynchronize());
    for (auto* cache : batch.caches) cache->commit_prefill();
    return logits;
  } catch (...) {
    for (size_t b = 0; b < batch.batch_size; ++b)
      if (begun[b]) batch.caches[b]->abort_prefill();
    throw;
  }
}

Tensor Qwen3CudaModel::prefill_logits_batch_with_caches(
    const std::vector<std::vector<int32_t>>& prompt_ids_batch,
    const std::vector<Qwen3KvCache*>& caches) const {
  const size_t batch = prompt_ids_batch.size();
  if (batch != 1 && batch != 2 && batch != 4)
    model_error("batched prefill batch size must be 1, 2, or 4; actual=" + std::to_string(batch));
  if (caches.size() != batch)
    model_error("batched prefill prompt/cache count mismatch");
  const size_t seq = prompt_ids_batch.front().size();
  if (seq == 0 || seq > 32) model_error("batched prefill seq_len must be in [1,32]");
  std::vector<Qwen3KvCache*> unique_caches;
  for (size_t b = 0; b < batch; ++b) {
    if (prompt_ids_batch[b].size() != seq) model_error("batched prefill requires equal prompt lengths");
    for (const auto id : prompt_ids_batch[b])
      if (id < 0 || id >= 151936) model_error("batched prompt token out of range");
    if (caches[b] == nullptr) model_error("batched prefill cache pointer must be non-null");
    if (std::find(unique_caches.begin(), unique_caches.end(), caches[b]) != unique_caches.end())
      model_error("batched prefill cache pointers must be unique");
    unique_caches.push_back(caches[b]);
    validate_contiguous_cache(*caches[b], model_spec_, "batched prefill");
    if (caches[b]->length() != 0 || caches[b]->capacity() < seq)
      model_error("batched prefill cache must be empty and have capacity >= seq_len");
  }
  std::vector<int32_t> flat_ids;
  flat_ids.reserve(batch * seq);
  for (const auto& prompt : prompt_ids_batch) flat_ids.insert(flat_ids.end(), prompt.begin(), prompt.end());
  std::vector<int32_t> positions(seq);
  for (size_t i = 0; i < seq; ++i) positions[i] = static_cast<int32_t>(i);
  Tensor hidden = cuda_embedding_lookup(token_embedding_, flat_ids).reshape({static_cast<int64_t>(batch), static_cast<int64_t>(seq), 1024});
  for (size_t layer = 0; layer < layers_.size(); ++layer) {
    DecoderLayerOutput trace = layer_executor().prefill_batch(
        hidden, positions, layers_[layer], rms_norm_eps_, rope_theta_);
    for (size_t b = 0; b < batch; ++b)
      caches[b]->write_prefill_layer_batched_slice(layer, trace.k_rope, trace.v_linear, b, seq);
    hidden = std::move(trace.layer_output);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  for (auto* cache : caches) {
    CUDA_CHECK(cudaDeviceSynchronize());
    cache->commit_prefill(seq);
  }
  Tensor final_hidden = cuda_rms_norm(hidden, final_norm_, rms_norm_eps_).reshape({static_cast<int64_t>(batch * seq), 1024});
  Tensor logits = cuda_lm_head(final_hidden, token_embedding_).reshape({static_cast<int64_t>(batch), static_cast<int64_t>(seq), 151936});
  CUDA_CHECK(cudaDeviceSynchronize());
  return logits;
}

Tensor Qwen3CudaModel::prefill_logits_padded_batch_with_caches(
    const PaddedPrefillBatch& batch) const {
  if (batch.batch_size != 1 && batch.batch_size != 2 && batch.batch_size != 4)
    model_error("padded prefill batch_size must be 1, 2, or 4");
  if (batch.caches.size() != batch.batch_size ||
      batch.valid_lengths.size() != batch.batch_size || batch.max_seq_len == 0 ||
      batch.max_seq_len > 32 || batch.padded_token_ids.size() !=
          batch.batch_size * batch.max_seq_len)
    model_error("padded prefill metadata shape/count mismatch");
  std::vector<Qwen3KvCache*> unique;
  unique.reserve(batch.batch_size);
  for (size_t b = 0; b < batch.batch_size; ++b) {
    const size_t valid = batch.valid_lengths[b];
    if (valid == 0 || valid > batch.max_seq_len || valid > 32)
      model_error("padded prefill valid length must be in [1,max_seq_len<=32]");
    if (batch.caches[b] == nullptr ||
        std::find(unique.begin(), unique.end(), batch.caches[b]) != unique.end())
      model_error("padded prefill cache pointers must be non-null and unique");
    unique.push_back(batch.caches[b]);
    validate_contiguous_cache(*batch.caches[b], model_spec_, "padded prefill");
    if (batch.caches[b]->length() != 0 || batch.caches[b]->capacity() < valid)
      model_error("padded prefill cache must be empty and have capacity >= valid length");
  }
  for (size_t i = 0; i < batch.padded_token_ids.size(); ++i)
    if (batch.padded_token_ids[i] < 0 || batch.padded_token_ids[i] >= 151936)
      model_error("padded token id out of range at flat index=" + std::to_string(i));
  for (size_t b = 0; b < batch.batch_size; ++b) {
    for (size_t s = batch.valid_lengths[b]; s < batch.max_seq_len; ++s) {
      const int32_t token = batch.padded_token_ids[b * batch.max_seq_len + s];
      if (token != 0)
        model_error("right padding token must be 0: row=" + std::to_string(b) +
                    " position=" + std::to_string(s) +
                    " actual token=" + std::to_string(token));
    }
  }

  std::vector<int32_t> positions(batch.max_seq_len);
  for (size_t i = 0; i < positions.size(); ++i) positions[i] = static_cast<int32_t>(i);
  try {
    // Right-padding-only path: positions remain the shared 0..S_max-1 range;
    // arbitrary position IDs and left padding are intentionally unsupported.
    Tensor valid_lengths_cuda(DType::I32,
                              {static_cast<int64_t>(batch.batch_size)},
                              DeviceType::CUDA);
    std::vector<int32_t> valid_lengths_host(batch.batch_size);
    for (size_t b = 0; b < batch.batch_size; ++b)
      valid_lengths_host[b] = static_cast<int32_t>(batch.valid_lengths[b]);
    CUDA_CHECK(cudaMemcpy(valid_lengths_cuda.data(), valid_lengths_host.data(),
                          valid_lengths_host.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    record_valid_length_h2d_upload();
    Tensor hidden = cuda_embedding_lookup(token_embedding_, batch.padded_token_ids)
                        .reshape({static_cast<int64_t>(batch.batch_size),
                                  static_cast<int64_t>(batch.max_seq_len), 1024});
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      DecoderLayerOutput trace = layer_executor().prefill_batch_valid_lengths(
          hidden, positions, valid_lengths_cuda, layers_[layer], rms_norm_eps_, rope_theta_);
      for (size_t b = 0; b < batch.batch_size; ++b) {
        batch.caches[b]->write_prefill_layer_batched_valid_slice(
            layer, trace.k_rope, trace.v_linear, b, batch.valid_lengths[b]);
      }
      hidden = std::move(trace.layer_output);
    }
    Tensor final_hidden = cuda_rms_norm(hidden, final_norm_, rms_norm_eps_)
                              .reshape({static_cast<int64_t>(batch.batch_size * batch.max_seq_len), 1024});
    Tensor logits = cuda_lm_head(final_hidden, token_embedding_)
                        .reshape({static_cast<int64_t>(batch.batch_size),
                                  static_cast<int64_t>(batch.max_seq_len), 151936});
    CUDA_CHECK(cudaDeviceSynchronize());
    for (size_t b = 0; b < batch.batch_size; ++b)
      batch.caches[b]->commit_prefill(batch.valid_lengths[b]);
    return logits;
  } catch (...) {
    for (auto* cache : batch.caches) {
      if (cache) {
        try { cache->abort_prefill(); } catch (...) {}
      }
    }
    throw;
  }
}

Tensor Qwen3CudaModel::decode_logits(int32_t next_input_id,
                                     Qwen3KvCache& cache) const {
  return decode_logits_variable_length_batch_with_caches({next_input_id}, {&cache});
}

Tensor Qwen3CudaModel::decode_logits_paged(int32_t next_input_id,
                                           Qwen3PagedKvCache& cache) const {
  Tensor logits(DType::F16,
                {1, static_cast<int64_t>(model_spec_.vocab_size)},
                DeviceType::CUDA);
  decode_logits_paged_into(next_input_id, cache, logits);
  return logits;
}

Tensor Qwen3CudaModel::decode_logits_paged_graph(
    int32_t next_input_id, Qwen3PagedKvCache& cache) const {
  return decode_logits_paged_graph_with_mode(
      next_input_id, cache, CudaAttentionMode::kReference);
}

Tensor Qwen3CudaModel::decode_logits_paged_flash_graph(
    int32_t next_input_id, Qwen3PagedKvCache& cache) const {
  return decode_logits_paged_graph_with_mode(
      next_input_id, cache, CudaAttentionMode::kFlashOnline);
}

Tensor Qwen3CudaModel::decode_logits_paged_batch_graph(
    const std::vector<int32_t>& next_input_ids,
    const std::vector<Qwen3PagedKvCache*>& caches) const {
  return decode_logits_paged_batch_graph_with_mode(
      next_input_ids, caches, CudaAttentionMode::kReference);
}

Tensor Qwen3CudaModel::decode_logits_paged_flash_batch_graph(
    const std::vector<int32_t>& next_input_ids,
    const std::vector<Qwen3PagedKvCache*>& caches) const {
  return decode_logits_paged_batch_graph_with_mode(
      next_input_ids, caches, CudaAttentionMode::kFlashOnline);
}

Tensor Qwen3CudaModel::decode_logits_paged_graph_with_mode(
    int32_t next_input_id, Qwen3PagedKvCache& cache,
    CudaAttentionMode mode) const {
  CudaAttentionModeGuard attention_mode(mode);
  if (next_input_id < 0 || next_input_id >= 151936)
    model_error("paged graph decode token out of range");
  validate_paged_cache_pool(cache, model_spec_, "paged graph decode");
  if (cache.in_decode_transaction() || cache.length() == 0 ||
      cache.length() >= cache.capacity() ||
      cache.length() >= static_cast<std::size_t>(INT32_MAX))
    model_error("paged graph decode cache has invalid length/transaction state");
  if (!decode_workspace_)
    model_error("paged graph decode workspace is not initialized");

  const bool compatible =
      paged_decode_graph_ && paged_decode_graph_->cache == &cache &&
      paged_decode_graph_->pool == &cache.pool() &&
      paged_decode_graph_->capacity == cache.capacity() &&
      paged_decode_graph_->attention_mode == mode;
  if (!compatible) paged_decode_graph_.reset();

  DecodeWorkspace& workspace = *decode_workspace_;
  const int32_t position = static_cast<int32_t>(cache.length());
  bool begun = false;
  bool capture_started = false;
  try {
    cache.begin_decode();
    begun = true;
    if (!paged_decode_metadata_workspace_)
      paged_decode_metadata_workspace_ =
          std::make_unique<PagedDecodeMetadataWorkspace>(
              cache.pool().config().block_size);
    else if (paged_decode_metadata_workspace_->block_size() !=
             cache.pool().config().block_size)
      model_error("paged graph decode metadata workspace block_size mismatch");
    Tensor positions = paged_decode_metadata_workspace_->positions_for_batch(1);
    Tensor block_table = paged_decode_metadata_workspace_->block_table_row(
        0, cache.block_table().size());
    CUDA_CHECK(cudaMemcpy(positions.data(), &position, sizeof(position),
                          cudaMemcpyHostToDevice));
    cache.copy_block_table_to_cuda(block_table);
    CUDA_CHECK(cudaMemcpy(workspace.token_ids.data(), &next_input_id,
                          sizeof(next_input_id), cudaMemcpyHostToDevice));

    if (!paged_decode_graph_) {
      auto candidate = std::make_unique<PagedDecodeGraphState>();
      candidate->cache = &cache;
      candidate->pool = &cache.pool();
      candidate->capacity = cache.capacity();
      candidate->attention_mode = mode;
      candidate->block_table = block_table;
      candidate->logits = Tensor(
          DType::F16, {1, static_cast<int64_t>(model_spec_.vocab_size)},
          DeviceType::CUDA);
      cuda_prepare_graph_capture();
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread,
                                        cudaStreamCaptureModeThreadLocal));
      capture_started = true;
      Tensor hidden_a = workspace.hidden_a.prefix_first_dim(1);
      Tensor hidden_b = workspace.hidden_b.prefix_first_dim(1);
      Tensor token_ids = workspace.token_ids.prefix_first_dim(1);
      cuda_embedding_lookup_device_ids_out(token_embedding_, token_ids, hidden_a);
      Tensor* current = &hidden_a;
      Tensor* next = &hidden_b;
      for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
        layer_executor().decode_paged_into(
            *current, position, layers_[layer], cache, layer,
            candidate->block_table, workspace, *next, rms_norm_eps_,
            rope_theta_, cudaStreamPerThread);
        std::swap(current, next);
      }
      Tensor final_hidden = workspace.final_norm.prefix_first_dim(1);
      cuda_rms_norm_out(*current, final_norm_, rms_norm_eps_, final_hidden);
      cuda_linear_out(final_hidden, token_embedding_, candidate->logits);
      CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &candidate->graph));
      capture_started = false;
      CUDA_CHECK(cudaGraphInstantiate(&candidate->executable, candidate->graph,
                                     nullptr, nullptr, 0));
      CUDA_CHECK(cudaGraphLaunch(candidate->executable, cudaStreamPerThread));
      CUDA_CHECK(cudaDeviceSynchronize());
      cache.commit_decode();
      paged_decode_graph_ = std::move(candidate);
      return paged_decode_graph_->logits;
    }

    CUDA_CHECK(cudaGraphLaunch(paged_decode_graph_->executable, cudaStreamPerThread));
    CUDA_CHECK(cudaDeviceSynchronize());
    // Graph replay executes the device-side KV writes but cannot execute the
    // host transaction bookkeeping performed by the layer wrapper.
    for (std::size_t layer = 0; layer < layers_.size(); ++layer)
      cache.mark_layer_kv_device_written(layer);
    cache.commit_decode();
    return paged_decode_graph_->logits;
  } catch (...) {
    if (capture_started) {
      cudaGraph_t discarded = nullptr;
      cudaStreamEndCapture(cudaStreamPerThread, &discarded);
      if (discarded) cudaGraphDestroy(discarded);
      capture_started = false;
    }
    if (begun) {
      try { cache.abort_decode(); } catch (...) {}
    }
    throw;
  }
}

Tensor Qwen3CudaModel::decode_logits_paged_batch_graph_with_mode(
    const std::vector<int32_t>& next_input_ids,
    const std::vector<Qwen3PagedKvCache*>& caches,
    CudaAttentionMode mode) const {
  CudaAttentionModeGuard attention_mode(mode);
  const std::size_t batch = next_input_ids.size();
  if (batch != 2 && batch != 4)
    model_error("paged batch graph decode batch size must be exactly 2 or 4");
  if (caches.size() != batch)
    model_error("paged batch graph decode token/cache count mismatch");
  if (!decode_workspace_)
    model_error("paged batch graph decode workspace is not initialized");

  PagedKvCachePool* common_pool = nullptr;
  std::vector<std::size_t> capacities;
  capacities.reserve(batch);
  std::vector<int32_t> positions(batch);
  for (std::size_t b = 0; b < batch; ++b) {
    if (next_input_ids[b] < 0 || next_input_ids[b] >= 151936)
      model_error("paged batch graph decode token out of range at row=" +
                  std::to_string(b));
    if (!caches[b])
      model_error("paged batch graph decode null cache at row=" +
                  std::to_string(b));
    if (std::find(caches.begin(), caches.begin() + b, caches[b]) !=
        caches.begin() + b)
      model_error("paged batch graph decode cache pointers must be unique");
    if (!common_pool) common_pool = &caches[b]->pool();
    if (&caches[b]->pool() != common_pool)
      model_error("paged batch graph decode caches must share one pool");
    validate_paged_cache_pool(*caches[b], model_spec_,
                              "paged batch graph decode");
    if (caches[b]->in_decode_transaction() || caches[b]->length() == 0 ||
        caches[b]->length() >= caches[b]->capacity() ||
        caches[b]->length() >= static_cast<std::size_t>(INT32_MAX))
      model_error("paged batch graph decode cache row has invalid length");
    positions[b] = static_cast<int32_t>(caches[b]->length());
    capacities.push_back(caches[b]->capacity());
  }

  bool compatible = paged_batch_decode_graph_ &&
      paged_batch_decode_graph_->pool == common_pool &&
      paged_batch_decode_graph_->attention_mode == mode &&
      paged_batch_decode_graph_->caches == caches &&
      paged_batch_decode_graph_->capacities == capacities;
  if (!compatible) paged_batch_decode_graph_.reset();

  if (!paged_decode_metadata_workspace_)
    paged_decode_metadata_workspace_ =
        std::make_unique<PagedDecodeMetadataWorkspace>(
            common_pool->config().block_size);
  else if (paged_decode_metadata_workspace_->block_size() !=
           common_pool->config().block_size)
    model_error("paged batch graph decode metadata block_size mismatch");

  std::vector<bool> begun(batch, false);
  bool capture_started = false;
  try {
    for (std::size_t b = 0; b < batch; ++b) {
      caches[b]->begin_decode();
      begun[b] = true;
    }
    Tensor positions_cuda =
        paged_decode_metadata_workspace_->positions_for_batch(batch);
    Tensor block_tables_cuda =
        paged_decode_metadata_workspace_->block_tables.prefix_first_dim(batch);
    CUDA_CHECK(cudaMemcpy(positions_cuda.data(), positions.data(),
                          batch * sizeof(int32_t), cudaMemcpyHostToDevice));
    for (std::size_t b = 0; b < batch; ++b) {
      Tensor row = paged_decode_metadata_workspace_->block_table_row(
          b, caches[b]->block_table().size());
      caches[b]->copy_block_table_to_cuda(row);
    }
    DecodeWorkspace& workspace = *decode_workspace_;
    CUDA_CHECK(cudaMemcpy(workspace.token_ids.data(), next_input_ids.data(),
                          batch * sizeof(int32_t), cudaMemcpyHostToDevice));

    if (!paged_batch_decode_graph_) {
      auto candidate = std::make_unique<PagedBatchDecodeGraphState>();
      candidate->caches = caches;
      candidate->pool = common_pool;
      candidate->capacities = capacities;
      candidate->attention_mode = mode;
      candidate->block_tables = block_tables_cuda;
      candidate->logits = Tensor(
          DType::F16,
          {static_cast<int64_t>(batch),
           static_cast<int64_t>(model_spec_.vocab_size)},
          DeviceType::CUDA);
      cuda_prepare_graph_capture();
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread,
                                        cudaStreamCaptureModeThreadLocal));
      capture_started = true;
      Tensor hidden_a = workspace.hidden_a.prefix_first_dim(batch);
      Tensor hidden_b = workspace.hidden_b.prefix_first_dim(batch);
      Tensor token_ids = workspace.token_ids.prefix_first_dim(batch);
      cuda_embedding_lookup_device_ids_out(token_embedding_, token_ids, hidden_a);
      Tensor* current = &hidden_a;
      Tensor* next = &hidden_b;
      const std::vector<DecoderPagedKvCache*> generic_caches =
          make_decoder_paged_caches(caches);
      for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
        layer_executor().decode_paged_batch_into(
            *current, positions_cuda, positions, layers_[layer],
            generic_caches, layer, candidate->block_tables, workspace, *next,
            rms_norm_eps_, rope_theta_);
        std::swap(current, next);
      }
      Tensor final_hidden = workspace.final_norm.prefix_first_dim(batch);
      cuda_rms_norm_out(*current, final_norm_, rms_norm_eps_, final_hidden);
      cuda_linear_out(final_hidden, token_embedding_, candidate->logits);
      CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &candidate->graph));
      capture_started = false;
      CUDA_CHECK(cudaGraphInstantiate(&candidate->executable, candidate->graph,
                                      nullptr, nullptr, 0));
      CUDA_CHECK(cudaGraphLaunch(candidate->executable, cudaStreamPerThread));
      CUDA_CHECK(cudaDeviceSynchronize());
      for (Qwen3PagedKvCache* cache : caches) cache->commit_decode();
      paged_batch_decode_graph_ = std::move(candidate);
      return paged_batch_decode_graph_->logits;
    }

    CUDA_CHECK(cudaGraphLaunch(paged_batch_decode_graph_->executable,
                               cudaStreamPerThread));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (Qwen3PagedKvCache* cache : caches) {
      for (std::size_t layer = 0; layer < layers_.size(); ++layer)
        cache->mark_layer_kv_device_written(layer);
      cache->commit_decode();
    }
    return paged_batch_decode_graph_->logits;
  } catch (...) {
    if (capture_started) {
      cudaGraph_t discarded = nullptr;
      cudaStreamEndCapture(cudaStreamPerThread, &discarded);
      if (discarded) cudaGraphDestroy(discarded);
    }
    for (std::size_t b = 0; b < batch; ++b) {
      if (begun[b]) {
        try { caches[b]->abort_decode(); } catch (...) {}
      }
    }
    throw;
  }
}

void Qwen3CudaModel::decode_logits_paged_flash_into(
    int32_t next_input_id, Qwen3PagedKvCache& cache, Tensor& logits) const {
  CudaAttentionModeGuard flash_attention(CudaAttentionMode::kFlashOnline);
  decode_logits_paged_into(next_input_id, cache, logits);
}

Tensor Qwen3CudaModel::decode_logits_paged_flash_batch(
    const std::vector<int32_t>& next_input_ids,
    const std::vector<Qwen3PagedKvCache*>& caches) const {
  CudaAttentionModeGuard flash_attention(CudaAttentionMode::kFlashOnline);
  return decode_logits_paged_batch(next_input_ids, caches);
}

void Qwen3CudaModel::decode_logits_paged_into(int32_t next_input_id,
                                              Qwen3PagedKvCache& cache,
                                              Tensor& logits) const {
  if (next_input_id < 0 || next_input_id >= 151936)
    model_error("paged decode token out of range");
  validate_paged_cache_pool(cache, model_spec_, "paged decode");
  if (cache.in_decode_transaction() || cache.length() == 0 ||
      cache.length() >= cache.capacity() ||
      cache.length() >= static_cast<size_t>(INT32_MAX))
    model_error("paged decode cache has invalid length/transaction state");
  if (!decode_workspace_)
    model_error("paged decode workspace is not initialized");

  DecodeWorkspace& workspace = *decode_workspace_;
  const int32_t position = static_cast<int32_t>(cache.length());
  bool begun = false;
  try {
    cache.begin_decode();
    begun = true;
    if (!paged_decode_metadata_workspace_)
      paged_decode_metadata_workspace_ =
          std::make_unique<PagedDecodeMetadataWorkspace>(cache.pool().config().block_size);
    else if (paged_decode_metadata_workspace_->block_size() !=
             cache.pool().config().block_size)
      model_error("paged decode metadata workspace block_size mismatch");
    const CudaAllocationStats metadata_before = cuda_allocation_stats();
    size_t positions_uploads = 0;
    size_t block_table_uploads = 0;
    Tensor block_table_cuda;
    try {
      Tensor positions_cuda = paged_decode_metadata_workspace_->positions_for_batch(1);
      CUDA_CHECK(cudaMemcpy(positions_cuda.data(), &position, sizeof(position),
                            cudaMemcpyHostToDevice));
      positions_uploads = 1;
      block_table_cuda = paged_decode_metadata_workspace_->block_table_row(
          0, cache.block_table().size());
      cache.copy_block_table_to_cuda(block_table_cuda);
      block_table_uploads = 1;
      last_paged_decode_metadata_upload_stats_ = metadata_upload_delta(
          metadata_before, cuda_allocation_stats(), positions_uploads,
          block_table_uploads, true);
    } catch (...) {
      last_paged_decode_metadata_upload_stats_ = metadata_upload_delta(
          metadata_before, cuda_allocation_stats(), positions_uploads,
          block_table_uploads, false);
      throw;
    }
    Tensor hidden_a = workspace.hidden_a.prefix_first_dim(1);
    Tensor hidden_b = workspace.hidden_b.prefix_first_dim(1);
    cuda_embedding_lookup_out(token_embedding_, {next_input_id},
                              workspace.token_ids, hidden_a);
    Tensor* current = &hidden_a;
    Tensor* next = &hidden_b;
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      layer_executor().decode_paged_into(
          *current, position, layers_[layer], cache, layer, block_table_cuda,
          workspace, *next, rms_norm_eps_, rope_theta_);
      std::swap(current, next);
    }
    Tensor final_hidden = workspace.final_norm.prefix_first_dim(1);
    cuda_rms_norm_out(*current, final_norm_, rms_norm_eps_, final_hidden);
    cuda_linear_out(final_hidden, token_embedding_, logits);
    CUDA_CHECK(cudaDeviceSynchronize());
    cache.commit_decode();
    return;
  } catch (...) {
    if (begun) {
      try { cache.abort_decode(); } catch (...) {}
    }
    throw;
  }
}

Tensor Qwen3CudaModel::decode_logits_paged_batch(
    const std::vector<int32_t>& next_input_ids,
    const std::vector<Qwen3PagedKvCache*>& caches) const {
  const size_t batch = next_input_ids.size();
  if (batch == 1) {
    if (caches.size() != 1 || !caches[0])
      model_error("paged batched decode token/cache count mismatch");
    return decode_logits_paged(next_input_ids[0], *caches[0]);
  }
  if (batch != 1 && batch != 2 && batch != 4)
    model_error("paged batched decode batch size must be 1, 2, or 4");
  if (caches.size() != batch)
    model_error("paged batched decode token/cache count mismatch");
  std::vector<Qwen3PagedKvCache*> unique;
  unique.reserve(batch);
  std::vector<int32_t> positions(batch);
  PagedKvCachePool* common_pool = nullptr;
  for (size_t b = 0; b < batch; ++b) {
    if (next_input_ids[b] < 0 || next_input_ids[b] >= 151936)
      model_error("paged batched decode token out of range at row=" + std::to_string(b));
    if (!caches[b]) model_error("paged batched decode null cache at row=" + std::to_string(b));
    if (std::find(unique.begin(), unique.end(), caches[b]) != unique.end())
      model_error("paged batched decode cache pointers must be unique");
    unique.push_back(caches[b]);
    if (!common_pool) common_pool = &caches[b]->pool();
    if (&caches[b]->pool() != common_pool)
      model_error("paged batched decode caches must share one PagedKvCachePool");
    validate_paged_cache_pool(*caches[b], model_spec_, "paged batched decode");
    if (caches[b]->in_decode_transaction() || caches[b]->length() == 0 ||
        caches[b]->length() >= caches[b]->capacity() ||
        caches[b]->length() >= static_cast<size_t>(INT32_MAX))
      model_error("paged batched decode cache row has invalid length/transaction state");
    positions[b] = static_cast<int32_t>(caches[b]->length());
  }
  std::vector<bool> begun(batch, false);
  if (!paged_decode_metadata_workspace_)
    paged_decode_metadata_workspace_ =
        std::make_unique<PagedDecodeMetadataWorkspace>(common_pool->config().block_size);
  else if (paged_decode_metadata_workspace_->block_size() !=
           common_pool->config().block_size)
    model_error("paged decode metadata workspace block_size mismatch");
  try {
    for (size_t b = 0; b < batch; ++b) {
      caches[b]->begin_decode();
      begun[b] = true;
    }
    Tensor block_tables_cuda =
        paged_decode_metadata_workspace_->block_tables.prefix_first_dim(batch);
    // This interval intentionally contains only metadata views and H2D
    // copies; it excludes embedding, all 28 layers and logits.
    const CudaAllocationStats metadata_before = cuda_allocation_stats();
    size_t positions_uploads = 0;
    size_t block_table_uploads = 0;
    std::optional<Tensor> positions_cuda;
    try {
      positions_cuda.emplace(
          paged_decode_metadata_workspace_->positions_for_batch(batch));
      CUDA_CHECK(cudaMemcpy(positions_cuda->data(), positions.data(),
                            batch * sizeof(int32_t), cudaMemcpyHostToDevice));
      positions_uploads = 1;
      for (size_t b = 0; b < batch; ++b) {
        Tensor block_table_row = paged_decode_metadata_workspace_->block_table_row(
            b, caches[b]->block_table().size());
        caches[b]->copy_block_table_to_cuda(block_table_row);
        ++block_table_uploads;
      }
      last_paged_decode_metadata_upload_stats_ = metadata_upload_delta(
          metadata_before, cuda_allocation_stats(), positions_uploads,
          block_table_uploads, true);
    } catch (...) {
      last_paged_decode_metadata_upload_stats_ = metadata_upload_delta(
          metadata_before, cuda_allocation_stats(), positions_uploads,
          block_table_uploads, false);
      throw;
    }
    if (!decode_workspace_)
      model_error("paged batched decode workspace is not initialized");
    DecodeWorkspace& workspace = *decode_workspace_;
    Tensor hidden_a = workspace.hidden_a.prefix_first_dim(batch);
    Tensor hidden_b = workspace.hidden_b.prefix_first_dim(batch);
    cuda_embedding_lookup_out(token_embedding_, next_input_ids, workspace.token_ids, hidden_a);
    Tensor* current = &hidden_a;
    Tensor* next = &hidden_b;
    const std::vector<DecoderPagedKvCache*> generic_caches =
        make_decoder_paged_caches(caches);
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      layer_executor().decode_paged_batch_into(
          *current, *positions_cuda, positions, layers_[layer], generic_caches,
          layer, block_tables_cuda, workspace, *next, rms_norm_eps_,
          rope_theta_);
      std::swap(current, next);
    }
    Tensor final_hidden = workspace.final_norm.prefix_first_dim(batch);
    cuda_rms_norm_out(*current, final_norm_, rms_norm_eps_, final_hidden);
    Tensor logits = cuda_lm_head(final_hidden, token_embedding_);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (size_t b = 0; b < batch; ++b) caches[b]->commit_decode();
    return logits;
  } catch (...) {
    for (size_t b = 0; b < batch; ++b)
      if (begun[b]) { try { caches[b]->abort_decode(); } catch (...) {} }
    throw;
  }
}

Tensor Qwen3CudaModel::decode_logits_batch_with_caches(
    const std::vector<int32_t>& next_input_ids,
    const std::vector<Qwen3KvCache*>& caches) const {
  const size_t batch = next_input_ids.size();
  if (batch != 1 && batch != 2 && batch != 4)
    model_error("batched decode batch size must be 1, 2, or 4; actual=" + std::to_string(batch));
  if (caches.size() != batch)
    model_error("batched decode token/cache count mismatch");
  std::vector<Qwen3KvCache*> unique_caches;
  unique_caches.reserve(batch);
  size_t previous_length = 0;
  for (size_t b = 0; b < batch; ++b) {
    if (next_input_ids[b] < 0 || next_input_ids[b] >= 151936)
      model_error("batched decode token out of range");
    if (caches[b] == nullptr) model_error("batched decode cache pointer must be non-null");
    if (std::find(unique_caches.begin(), unique_caches.end(), caches[b]) != unique_caches.end())
      model_error("batched decode cache pointers must be unique");
    unique_caches.push_back(caches[b]);
    if (b == 0) previous_length = caches[b]->length();
    validate_contiguous_cache(*caches[b], model_spec_, "batched decode");
    if (caches[b]->length() != previous_length || previous_length == 0 ||
        previous_length >= caches[b]->capacity() ||
        previous_length >= static_cast<size_t>(INT32_MAX))
      model_error("batched decode caches must have equal non-empty Qwen3-compatible length and capacity >= length+1");
  }
  std::vector<int32_t> positions{static_cast<int32_t>(previous_length)};
  if (!decode_workspace_) model_error("decode workspace is not initialized");
  DecodeWorkspace& workspace = *decode_workspace_;
  Tensor hidden_a = workspace.hidden_a.prefix_first_dim(batch);
  Tensor hidden_b = workspace.hidden_b.prefix_first_dim(batch);
  try {
    cuda_embedding_lookup_out(token_embedding_, next_input_ids,
                              workspace.token_ids, hidden_a);
    Tensor* current = &hidden_a;
    Tensor* next = &hidden_b;
    const std::vector<DecoderKvCache*> generic_caches =
        make_decoder_caches(caches);
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      layer_executor().decode_batch_into(
          *current, positions[0], layers_[layer], generic_caches, layer,
          workspace, *next, rms_norm_eps_, rope_theta_);
      std::swap(current, next);
    }
    Tensor final_hidden = workspace.final_norm.prefix_first_dim(batch);
    cuda_rms_norm_out(*current, final_norm_, rms_norm_eps_, final_hidden);
    Tensor logits = cuda_lm_head(final_hidden, token_embedding_);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (auto* cache : caches) cache->commit_decode(previous_length);
    CUDA_CHECK(cudaDeviceSynchronize());
    return logits;
  } catch (...) {
    for (auto* cache : caches) {
      try { cache->abort_decode(); } catch (...) {}
    }
    throw;
  }
}

Tensor Qwen3CudaModel::decode_logits_variable_length_batch_with_caches(
    const std::vector<int32_t>& next_input_ids,
    const std::vector<Qwen3KvCache*>& caches) const {
  const char* function = "variable-length decode";
  const size_t batch = next_input_ids.size();
  if (batch != 1 && batch != 2 && batch != 4)
    model_error(std::string(function) + " batch size must be 1, 2, or 4; actual=" + std::to_string(batch));
  if (caches.size() != batch)
    model_error(std::string(function) + " token/cache count mismatch");
  std::vector<Qwen3KvCache*> unique;
  unique.reserve(batch);
  std::vector<int32_t> positions(batch), lengths_after(batch);
  for (size_t b = 0; b < batch; ++b) {
    if (next_input_ids[b] < 0 || next_input_ids[b] >= 151936)
      model_error(std::string(function) + " token out of range at row=" + std::to_string(b));
    if (!caches[b]) model_error(std::string(function) + " cache pointer must be non-null at row=" + std::to_string(b));
    if (std::find(unique.begin(), unique.end(), caches[b]) != unique.end())
      model_error(std::string(function) + " cache pointers must be unique");
    unique.push_back(caches[b]);
    const size_t length = caches[b]->length();
    validate_contiguous_cache(*caches[b], model_spec_, function);
    if (length == 0 || length >= caches[b]->capacity() ||
        length > static_cast<size_t>(INT32_MAX))
      model_error(std::string(function) + " row=" + std::to_string(b) +
                  " cache length/capacity/config invalid; length=" + std::to_string(length));
    positions[b] = static_cast<int32_t>(length);
    lengths_after[b] = static_cast<int32_t>(length + 1);
  }
  if (!decode_workspace_) model_error(std::string(function) + " workspace is not initialized");
  DecodeWorkspace& workspace = *decode_workspace_;
  Tensor hidden_a = workspace.hidden_a.prefix_first_dim(batch);
  Tensor hidden_b = workspace.hidden_b.prefix_first_dim(batch);
  try {
    cuda_embedding_lookup_out(token_embedding_, next_input_ids, workspace.token_ids, hidden_a);
    CUDA_CHECK(cudaMemcpy(workspace.variable_position_ids.data(), positions.data(),
                          batch * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(workspace.variable_lengths.data(), lengths_after.data(),
                          batch * sizeof(int32_t), cudaMemcpyHostToDevice));
    record_variable_decode_h2d_upload();
    Tensor* current = &hidden_a;
    Tensor* next = &hidden_b;
    const std::vector<DecoderKvCache*> generic_caches =
        make_decoder_caches(caches);
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      layer_executor().decode_batch_variable_into(
          *current, workspace.variable_position_ids.prefix_first_dim(batch),
          workspace.variable_lengths.prefix_first_dim(batch), layers_[layer],
          generic_caches, layer, workspace, *next, rms_norm_eps_, rope_theta_);
      std::swap(current, next);
    }
    Tensor final_hidden = workspace.final_norm.prefix_first_dim(batch);
    cuda_rms_norm_out(*current, final_norm_, rms_norm_eps_, final_hidden);
    Tensor logits = cuda_lm_head(final_hidden, token_embedding_);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (size_t b = 0; b < batch; ++b) caches[b]->commit_decode(caches[b]->length());
    CUDA_CHECK(cudaDeviceSynchronize());
    return logits;
  } catch (...) {
    for (auto* cache : caches) {
      try { if (cache) cache->abort_decode(); } catch (...) {}
    }
    throw;
  }
}

Tensor Qwen3CudaModel::decode_logits_variable_length_flash_batch_with_caches(
    const std::vector<int32_t>& next_input_ids,
    const std::vector<Qwen3KvCache*>& caches) const {
  CudaAttentionModeGuard flash_attention(CudaAttentionMode::kFlashOnline);
  return decode_logits_variable_length_batch_with_caches(next_input_ids, caches);
}

GreedyGenerationResult Qwen3CudaModel::generate_greedy(
    const std::vector<int32_t>& prompt_ids, size_t max_new_tokens,
    std::optional<int32_t> eos_token_id, size_t max_seq_len) const {
  if (prompt_ids.empty()) model_error("generate_greedy requires non-empty prompt");
  if (max_seq_len == 0 || max_seq_len < prompt_ids.size())
    model_error("max_seq_len actual=" + std::to_string(max_seq_len) +
                " expected >= prompt length and > 0");
  if (eos_token_id && (*eos_token_id < 0 || *eos_token_id >= 151936))
    model_error("eos_token_id actual=" + std::to_string(*eos_token_id) +
                " expected in [0,151936)");
  for (size_t i = 0; i < prompt_ids.size(); ++i)
    if (prompt_ids[i] < 0 || prompt_ids[i] >= 151936)
      model_error("prompt_ids[" + std::to_string(i) + "] out of range");
  GreedyGenerationResult result;
  if (max_new_tokens == 0) {
    result.stop_reason = "max_new_tokens";
    return result;
  }
  Qwen3KvCache cache(max_seq_len);
  Tensor logits = prefill_logits_with_cache(prompt_ids, cache);
  for (size_t step = 0; step < max_new_tokens; ++step) {
    const int32_t token = cuda_argmax_last_row(logits);
    result.generated_ids.push_back(token);
    if (eos_token_id && token == *eos_token_id) {
      result.stop_reason = "eos";
      result.final_cache_length = cache.length();
      return result;
    }
    if (step + 1 == max_new_tokens) {
      result.stop_reason = "max_new_tokens";
      result.final_cache_length = cache.length();
      return result;
    }
    if (cache.length() >= cache.capacity()) {
      result.stop_reason = "cache_capacity";
      result.final_cache_length = cache.length();
      return result;
    }
    logits = decode_logits(token, cache);
  }
  result.stop_reason = "max_new_tokens";
  result.final_cache_length = cache.length();
  return result;
}

GreedyGenerationResult Qwen3CudaModel::generate_greedy_paged(
    PagedKvCachePool& pool, const std::vector<int32_t>& prompt_ids,
    size_t max_new_tokens, std::optional<int32_t> eos_token_id,
    size_t max_seq_len) const {
  if (prompt_ids.empty()) model_error("generate_greedy_paged requires non-empty prompt");
  if (max_seq_len == 0 || max_seq_len < prompt_ids.size())
    model_error("max_seq_len actual=" + std::to_string(max_seq_len) +
                " expected >= prompt length and > 0");
  if (eos_token_id && (*eos_token_id < 0 || *eos_token_id >= 151936))
    model_error("eos_token_id actual=" + std::to_string(*eos_token_id) +
                " expected in [0,151936)");
  for (size_t i = 0; i < prompt_ids.size(); ++i)
    if (prompt_ids[i] < 0 || prompt_ids[i] >= 151936)
      model_error("prompt_ids[" + std::to_string(i) + "] out of range");

  GreedyGenerationResult result;
  if (max_new_tokens == 0) {
    result.stop_reason = "max_new_tokens";
    return result;
  }

  Qwen3PagedKvCache cache(pool, max_seq_len);
  struct CacheReleaseGuard {
    Qwen3PagedKvCache& cache;
    ~CacheReleaseGuard() { cache.release_all(); }
  } cache_guard{cache};
  Tensor logits = prefill_logits_paged(prompt_ids, cache);
  for (size_t step = 0; step < max_new_tokens; ++step) {
    const int32_t token = cuda_argmax_last_row(logits);
    result.generated_ids.push_back(token);
    if (eos_token_id && token == *eos_token_id) {
      result.stop_reason = "eos";
      result.final_cache_length = cache.length();
      return result;
    }
    if (step + 1 == max_new_tokens) {
      result.stop_reason = "max_new_tokens";
      result.final_cache_length = cache.length();
      return result;
    }
    if (cache.length() >= cache.capacity()) {
      result.stop_reason = "cache_capacity";
      result.final_cache_length = cache.length();
      return result;
    }
    logits = decode_logits_paged(token, cache);
  }
  result.stop_reason = "max_new_tokens";
  result.final_cache_length = cache.length();
  return result;
}

GreedyGenerationResult Qwen3CudaModel::generate_sampled_paged(
    PagedKvCachePool& pool, const std::vector<int32_t>& prompt_ids,
    size_t max_new_tokens, std::optional<int32_t> eos_token_id,
    size_t max_seq_len, const SamplingConfig& sampling) const {
  if (prompt_ids.empty()) model_error("generate_sampled_paged requires non-empty prompt");
  if (max_seq_len == 0 || max_seq_len < prompt_ids.size())
    model_error("max_seq_len actual=" + std::to_string(max_seq_len) +
                " expected >= prompt length and > 0");
  if (eos_token_id && (*eos_token_id < 0 || *eos_token_id >= 151936))
    model_error("eos_token_id actual=" + std::to_string(*eos_token_id) +
                " expected in [0,151936)");
  for (size_t i = 0; i < prompt_ids.size(); ++i)
    if (prompt_ids[i] < 0 || prompt_ids[i] >= 151936)
      model_error("prompt_ids[" + std::to_string(i) + "] out of range");

  GreedyGenerationResult result;
  if (max_new_tokens == 0) {
    result.stop_reason = "max_new_tokens";
    return result;
  }

  Qwen3PagedKvCache cache(pool, max_seq_len);
  struct CacheReleaseGuard {
    Qwen3PagedKvCache& cache;
    ~CacheReleaseGuard() { cache.release_all(); }
  } cache_guard{cache};
  Tensor logits = prefill_logits_paged(prompt_ids, cache);
  CudaSampler sampler(151936);
  for (size_t step = 0; step < max_new_tokens; ++step) {
    const int32_t token = sampler.sample_last_row(logits, sampling, step);
    result.generated_ids.push_back(token);
    if (eos_token_id && token == *eos_token_id) {
      result.stop_reason = "eos";
      result.final_cache_length = cache.length();
      return result;
    }
    if (step + 1 == max_new_tokens) {
      result.stop_reason = "max_new_tokens";
      result.final_cache_length = cache.length();
      return result;
    }
    if (cache.length() >= cache.capacity()) {
      result.stop_reason = "cache_capacity";
      result.final_cache_length = cache.length();
      return result;
    }
    logits = decode_logits_paged(token, cache);
  }
  result.stop_reason = "max_new_tokens";
  result.final_cache_length = cache.length();
  return result;
}

GreedyGenerationResult Qwen3CudaModel::generate_sampled(
    const std::vector<int32_t>& prompt_ids, size_t max_new_tokens,
    std::optional<int32_t> eos_token_id, size_t max_seq_len,
    const SamplingConfig& sampling) const {
  if (prompt_ids.empty()) model_error("generate_sampled requires non-empty prompt");
  if (max_seq_len == 0 || max_seq_len < prompt_ids.size())
    model_error("max_seq_len actual=" + std::to_string(max_seq_len) +
                " expected >= prompt length and > 0");
  if (eos_token_id && (*eos_token_id < 0 || *eos_token_id >= 151936))
    model_error("eos_token_id actual=" + std::to_string(*eos_token_id) +
                " expected in [0,151936)");
  for (size_t i = 0; i < prompt_ids.size(); ++i)
    if (prompt_ids[i] < 0 || prompt_ids[i] >= 151936)
      model_error("prompt_ids[" + std::to_string(i) + "] out of range");
  GreedyGenerationResult result;
  if (max_new_tokens == 0) {
    result.stop_reason = "max_new_tokens";
    return result;
  }
  Qwen3KvCache cache(max_seq_len);
  Tensor logits = prefill_logits_with_cache(prompt_ids, cache);
  CudaSampler sampler(151936);
  for (size_t step = 0; step < max_new_tokens; ++step) {
    const int32_t token = sampler.sample_last_row(logits, sampling, step);
    result.generated_ids.push_back(token);
    if (eos_token_id && token == *eos_token_id) {
      result.stop_reason = "eos";
      result.final_cache_length = cache.length();
      return result;
    }
    if (step + 1 == max_new_tokens) {
      result.stop_reason = "max_new_tokens";
      result.final_cache_length = cache.length();
      return result;
    }
    if (cache.length() >= cache.capacity()) {
      result.stop_reason = "cache_capacity";
      result.final_cache_length = cache.length();
      return result;
    }
    logits = decode_logits(token, cache);
  }
  result.stop_reason = "max_new_tokens";
  result.final_cache_length = cache.length();
  return result;
}

}  // namespace llm
