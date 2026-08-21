#include "llm/qwen3_cuda_model.h"

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

Qwen3CudaLayerWeights load_layer(const ModelPackage& package, size_t index) {
  const std::string prefix = "layers." + std::to_string(index) + ".";
  return {
      load_f16_cuda(package, prefix + "input_norm", {1024}),
      load_f16_cuda(package, prefix + "q_proj", {2048, 1024}),
      load_f16_cuda(package, prefix + "k_proj", {1024, 1024}),
      load_f16_cuda(package, prefix + "v_proj", {1024, 1024}),
      load_f16_cuda(package, prefix + "q_norm", {128}),
      load_f16_cuda(package, prefix + "k_norm", {128}),
      load_f16_cuda(package, prefix + "o_proj", {1024, 2048}),
      load_f16_cuda(package, prefix + "post_attention_norm", {1024}),
      load_f16_cuda(package, prefix + "gate_proj", {3072, 1024}),
      load_f16_cuda(package, prefix + "up_proj", {3072, 1024}),
      load_f16_cuda(package, prefix + "down_proj", {1024, 3072})};
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

Qwen3CudaModel::Qwen3CudaModel(const std::filesystem::path& package_root) {
  ModelPackage package(package_root);
  if (package.config("export_dtype") != "f32")
    model_error("config export_dtype actual=" + package.config("export_dtype") +
                " expected=f32");
  rms_norm_eps_ = std::stof(package.config("rms_norm_eps"));
  rope_theta_ = std::stof(package.config("rope_theta"));
  if (!std::isfinite(rms_norm_eps_) || rms_norm_eps_ <= 0.0f)
    model_error("rms_norm_eps actual=" + package.config("rms_norm_eps") +
                " expected finite > 0");
  if (!std::isfinite(rope_theta_) || rope_theta_ <= 0.0f)
    model_error("rope_theta actual=" + package.config("rope_theta") +
                " expected finite > 0");
  if (package.config("tie_word_embeddings") != "true")
    model_error("config tie_word_embeddings actual=" +
                package.config("tie_word_embeddings") + " expected=true");

  model_spec_.attention = AttentionConfig{16, 8, 128, 16, 512, true};
  model_spec_.hidden_size = 1024;
  model_spec_.intermediate_size = 3072;
  model_spec_.vocab_size = 151936;
  model_spec_.num_layers = 28;
  model_spec_.rms_norm_eps = rms_norm_eps_;
  model_spec_.rope_theta = rope_theta_;
  model_spec_.tie_word_embeddings = true;

  token_embedding_ = load_f16_cuda(package, "token_embedding", {151936, 1024});
  final_norm_ = load_f16_cuda(package, "final_norm", {1024});
  layers_.reserve(model_spec_.num_layers);
  for (size_t i = 0; i < model_spec_.num_layers; ++i)
    layers_.push_back(load_layer(package, i));

  resident_weight_bytes_ = token_embedding_.nbytes() + final_norm_.nbytes();
  for (const auto& layer : layers_) {
    resident_weight_bytes_ += layer.input_norm.nbytes();
    resident_weight_bytes_ += layer.q_proj.nbytes() + layer.k_proj.nbytes();
    resident_weight_bytes_ += layer.v_proj.nbytes() + layer.q_norm.nbytes();
    resident_weight_bytes_ += layer.k_norm.nbytes() + layer.o_proj.nbytes();
    resident_weight_bytes_ += layer.post_attention_norm.nbytes();
    resident_weight_bytes_ += layer.gate_proj.nbytes() + layer.up_proj.nbytes();
    resident_weight_bytes_ += layer.down_proj.nbytes();
  }
  // DecodeWorkspace is resident before any benchmark or steady-state decode
  // measurement; the model remains single-threaded for decode calls.
  decode_workspace_ = std::make_unique<DecodeWorkspace>();
}

Tensor Qwen3CudaModel::prefill_hidden_layers(
    const std::vector<int32_t>& token_ids,
    const std::vector<int32_t>& position_ids, size_t layer_count) const {
  validate_runtime_inputs(token_ids, position_ids, layer_count);
  Tensor hidden = cuda_embedding_lookup(token_embedding_, token_ids);
  for (size_t i = 0; i < layer_count; ++i) {
    Qwen3LayerTrace trace = qwen3_decoder_layer_cuda_trace_fp16(
        hidden, position_ids, layers_[i], rms_norm_eps_, rope_theta_);
    hidden = std::move(trace.layer_output);
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
  if (cache.num_layers() != layers_.size() || cache.num_kv_heads() != 8 ||
      cache.head_dim() != 128)
    model_error("cache config does not match Qwen3 model");
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
    Qwen3LayerTrace trace = qwen3_decoder_layer_cuda_trace_fp16(
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
    const auto& c = caches[b]->pool().config();
    if (c.num_layers != layers_.size() || c.num_kv_heads != 8 || c.head_dim != 128 ||
        c.dtype != DType::F16)
      model_error("paged batched prefill pool is not Qwen3-compatible F16");
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
      Qwen3BatchLayerTrace trace = qwen3_decoder_layer_cuda_trace_fp16_batch(
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
    const auto& c = caches[b]->pool().config();
    if (c.num_layers != layers_.size() || c.num_kv_heads != 8 ||
        c.head_dim != 128 || c.dtype != DType::F16)
      model_error("packed paged prefill pool is not Qwen3-compatible F16");
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
      Qwen3PackedLayerTrace trace =
          qwen3_decoder_layer_cuda_trace_fp16_packed(
              hidden, positions_cuda, offsets_cuda, layers_[layer],
              rms_norm_eps_, rope_theta_);
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
    const auto& config = common_pool->config();
    if (config.num_layers != layers_.size() || config.num_kv_heads != 8 ||
        config.head_dim != 128 || config.dtype != DType::F16)
      model_error("paged padded prefill pool is not Qwen3-compatible F16");
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
      Qwen3BatchLayerTrace trace = qwen3_decoder_layer_cuda_trace_fp16_batch_valid_lengths(
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
    if (caches[b]->length() != 0 || caches[b]->capacity() < seq ||
        caches[b]->num_layers() != layers_.size() || caches[b]->num_kv_heads() != 8 ||
        caches[b]->head_dim() != 128)
      model_error("batched prefill cache must be empty, Qwen3-configured, and have capacity >= seq_len");
  }
  std::vector<int32_t> flat_ids;
  flat_ids.reserve(batch * seq);
  for (const auto& prompt : prompt_ids_batch) flat_ids.insert(flat_ids.end(), prompt.begin(), prompt.end());
  std::vector<int32_t> positions(seq);
  for (size_t i = 0; i < seq; ++i) positions[i] = static_cast<int32_t>(i);
  Tensor hidden = cuda_embedding_lookup(token_embedding_, flat_ids).reshape({static_cast<int64_t>(batch), static_cast<int64_t>(seq), 1024});
  for (size_t layer = 0; layer < layers_.size(); ++layer) {
    Qwen3BatchLayerTrace trace = qwen3_decoder_layer_cuda_trace_fp16_batch(
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
    if (batch.caches[b]->length() != 0 || batch.caches[b]->capacity() < valid ||
        batch.caches[b]->num_layers() != layers_.size() ||
        batch.caches[b]->num_kv_heads() != 8 || batch.caches[b]->head_dim() != 128)
      model_error("padded prefill cache must be empty, Qwen3-configured, and have capacity >= valid length");
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
      Qwen3BatchLayerTrace trace = qwen3_decoder_layer_cuda_trace_fp16_batch_valid_lengths(
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
  // Keep the single-request API, but route it through the same genuine batch
  // kernel path so B=1 cannot silently retain the old row-by-row attention.
  return decode_logits_paged_batch({next_input_id}, {&cache});
}

Tensor Qwen3CudaModel::decode_logits_paged_batch(
    const std::vector<int32_t>& next_input_ids,
    const std::vector<Qwen3PagedKvCache*>& caches) const {
  const size_t batch = next_input_ids.size();
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
    const auto& c = caches[b]->pool().config();
    if (c.dtype != DType::F16 || c.num_layers != layers_.size() ||
        c.num_kv_heads != 8 || c.block_size == 0 || c.head_dim != 128)
      model_error("paged batched decode cache pool is not Qwen3-compatible F16");
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
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      qwen3_decoder_layer_cuda_decode_fp16_paged_batch_into(
          *current, *positions_cuda, positions, layers_[layer], caches, layer,
          block_tables_cuda, workspace, *next, rms_norm_eps_, rope_theta_);
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
    if (caches[b]->length() != previous_length || previous_length == 0 ||
        previous_length >= caches[b]->capacity() ||
        caches[b]->num_layers() != layers_.size() || caches[b]->num_kv_heads() != 8 ||
        caches[b]->head_dim() != 128 || previous_length >= static_cast<size_t>(INT32_MAX))
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
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      qwen3_decoder_layer_cuda_decode_fp16_batch_into(
          *current, positions[0], layers_[layer], caches, layer,
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
    if (length == 0 || length >= caches[b]->capacity() ||
        length > static_cast<size_t>(INT32_MAX) ||
        caches[b]->num_layers() != layers_.size() ||
        caches[b]->num_kv_heads() != 8 || caches[b]->head_dim() != 128)
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
    for (size_t layer = 0; layer < layers_.size(); ++layer) {
      qwen3_decoder_layer_cuda_decode_fp16_batch_variable_into(
          *current, workspace.variable_position_ids.prefix_first_dim(batch),
          workspace.variable_lengths.prefix_first_dim(batch), layers_[layer], caches, layer,
          workspace, *next, rms_norm_eps_, rope_theta_);
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
