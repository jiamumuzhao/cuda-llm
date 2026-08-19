#include "llm/cuda_check.h"
#include "llm/ops.h"
#include "llm/ops_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static void fail(const std::string& s) { throw std::runtime_error("test_ops_cuda: " + s); }

static Tensor quantize_f16(const Tensor& x) {
  Tensor h(DType::F16, x.shape());
  for (size_t i = 0; i < x.numel(); ++i) h.set_f32(i, x.get_f32(i));
  return h;
}

static Tensor dequantize_f16(const Tensor& x) {
  Tensor f(DType::F32, x.shape());
  for (size_t i = 0; i < x.numel(); ++i) f.set_f32(i, x.get_f32(i));
  return f;
}

static void metric(const std::string& name, const Tensor& actual, const Tensor& expected,
                   float tolerance) {
  if (actual.shape() != expected.shape()) {
    fail(name + " shape actual/expected mismatch");
  }
  float max_error = 0.0f;
  double sum_error = 0.0, dot = 0.0, aa = 0.0, bb = 0.0;
  for (size_t i = 0; i < actual.numel(); ++i) {
    const double a = actual.get_f32(i), b = expected.get_f32(i);
    const double d = std::fabs(a - b);
    max_error = std::max(max_error, static_cast<float>(d));
    sum_error += d;
    dot += a * b;
    aa += a * a;
    bb += b * b;
  }
  const double cosine = dot / (std::sqrt(aa * bb) + 1e-30);
  std::cout << name << " max_abs_error=" << max_error
            << " mean_abs_error=" << (actual.numel() ? sum_error / actual.numel() : 0.0)
            << " cosine_similarity=" << cosine << " tolerance=" << tolerance << "\n";
  if (max_error > tolerance) {
    fail(name + " actual=" + std::to_string(max_error) +
         " expected<=tolerance actual_shape=" + std::to_string(actual.numel()) +
         " tolerance=" + std::to_string(tolerance));
  }
}

static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); }
  catch (const std::exception&) { return; }
  fail(name + " expected exception");
}

static Tensor gqa_input(const std::vector<size_t>& shape, float base) {
  std::vector<int64_t> dims(shape.begin(), shape.end());
  Tensor x(DType::F32, dims);
  const size_t heads = shape[1], head_dim = shape[2];
  for (size_t i = 0; i < x.numel(); ++i) {
    const size_t d = i % head_dim;
    const size_t h = (i / head_dim) % heads;
    const size_t t = i / (heads * head_dim);
    x.set_f32(i, base + 0.013f * static_cast<float>(t) +
                    0.007f * static_cast<float>(h) + 0.001f * static_cast<float>(d));
  }
  return x;
}

static void test_gqa(int seq, bool f16_case) {
  constexpr size_t q_heads = 16, kv_heads = 8, head_dim = 128;
  Tensor q = gqa_input({static_cast<size_t>(seq), q_heads, head_dim}, 0.03f);
  Tensor k = gqa_input({static_cast<size_t>(seq), kv_heads, head_dim}, 0.07f);
  Tensor v = gqa_input({static_cast<size_t>(seq), kv_heads, head_dim}, 0.0f);
  for (size_t t = 0; t < static_cast<size_t>(seq); ++t) {
    for (size_t h = 0; h < kv_heads; ++h) {
      for (size_t d = 0; d < head_dim; ++d) {
        v.set_f32((t * kv_heads + h) * head_dim + d,
                  0.25f + 0.01f * static_cast<float>(t) +
                  0.02f * static_cast<float>(h) + 0.001f * static_cast<float>(d));
      }
    }
  }
  if (f16_case) {
    Tensor q16 = quantize_f16(q), k16 = quantize_f16(k), v16 = quantize_f16(v);
    metric("cuda_gqa_attention dtype=F16 seq_len=" + std::to_string(seq),
           cuda_gqa_attention(q16.to(DeviceType::CUDA), k16.to(DeviceType::CUDA),
                              v16.to(DeviceType::CUDA)).to(DeviceType::CPU),
           gqa_attention(dequantize_f16(q16), dequantize_f16(k16), dequantize_f16(v16)), 5e-3f);
  } else {
    metric("cuda_gqa_attention dtype=F32 seq_len=" + std::to_string(seq),
           cuda_gqa_attention(q.to(DeviceType::CUDA), k.to(DeviceType::CUDA),
                              v.to(DeviceType::CUDA)).to(DeviceType::CPU),
           gqa_attention(q, k, v), 1e-4f);
  }
}

int main() {
  try {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA device=" << prop.name << " compute_capability=" << prop.major
              << "." << prop.minor << "\n";

    Tensor x = Tensor::from_f32({3, 4}, {1, -2, 3, 0, .5f, 2, -1, 4, -3, 1, 2, -.5f});
    Tensor w = Tensor::from_f32({5, 4}, {1, 2, 3, 4, -1, 0, 2, 1, .5f, -2, 1, 3,
                                          2, 1, 0, -1, 3, -1, 2, 2});
    Tensor dx = x.to(DeviceType::CUDA), dw = w.to(DeviceType::CUDA);
    metric("cuda_linear_f32", cuda_linear(dx, dw).to(DeviceType::CPU), linear(x, w), 1e-4f);
    Tensor x16 = quantize_f16(x), w16 = quantize_f16(w);
    metric("cuda_linear_f16", cuda_linear(x16.to(DeviceType::CUDA), w16.to(DeviceType::CUDA)).to(DeviceType::CPU),
           linear(dequantize_f16(x16), dequantize_f16(w16)), 5e-3f);

    Tensor norm_w = Tensor::from_f32({4}, {1, .5f, 2, -1});
    metric("cuda_rms_norm_f32", cuda_rms_norm(dx, norm_w.to(DeviceType::CUDA), 1e-5f).to(DeviceType::CPU),
           rms_norm(x, norm_w, 1e-5f), 1e-4f);
    Tensor norm_w16 = quantize_f16(norm_w);
    metric("cuda_rms_norm_f16", cuda_rms_norm(x16.to(DeviceType::CUDA), norm_w16.to(DeviceType::CUDA), 1e-5f).to(DeviceType::CPU),
           rms_norm(dequantize_f16(x16), dequantize_f16(norm_w16), 1e-5f), 5e-3f);

    metric("cuda_add_f32", cuda_add(dx, dx).to(DeviceType::CPU), add(x, x), 1e-4f);
    metric("cuda_add_f16", cuda_add(x16.to(DeviceType::CUDA), x16.to(DeviceType::CUDA)).to(DeviceType::CPU),
           add(dequantize_f16(x16), dequantize_f16(x16)), 5e-3f);
    metric("cuda_swiglu_f32", cuda_swiglu(dx, dx).to(DeviceType::CPU), swiglu(x, x), 1e-4f);
    metric("cuda_swiglu_f16", cuda_swiglu(x16.to(DeviceType::CUDA), x16.to(DeviceType::CUDA)).to(DeviceType::CPU),
           swiglu(dequantize_f16(x16), dequantize_f16(x16)), 5e-3f);

    Tensor rope_x = Tensor::from_f32({2, 1, 4}, {1, 2, 3, 4, 2, -1, .5f, 3});
    std::vector<int32_t> positions{0, 1};
    metric("cuda_rope_f32", cuda_rope(rope_x.to(DeviceType::CUDA), positions, 10000).to(DeviceType::CPU),
           rope(rope_x, positions, 10000), 1e-4f);
    Tensor rope16 = quantize_f16(rope_x);
    metric("cuda_rope_f16", cuda_rope(rope16.to(DeviceType::CUDA), positions, 10000).to(DeviceType::CPU),
           rope(dequantize_f16(rope16), positions, 10000), 5e-3f);

    Tensor sm = Tensor::from_f32({2, 4}, {1000, 1001, -1000, -999, 1, 2, 3, 4});
    metric("cuda_softmax_last_dim_f32", cuda_softmax_last_dim(sm.to(DeviceType::CUDA)).to(DeviceType::CPU),
           softmax_last_dim(sm), 1e-4f);
    Tensor sm16 = quantize_f16(sm), sm16_out = cuda_softmax_last_dim(sm16.to(DeviceType::CUDA)).to(DeviceType::CPU);
    metric("cuda_softmax_last_dim_f16", sm16_out, softmax_last_dim(dequantize_f16(sm16)), 5e-3f);
    for (int row = 0; row < 2; ++row) {
      float sum = 0;
      for (int col = 0; col < 4; ++col) sum += sm16_out.get_f32(static_cast<size_t>(row * 4 + col));
      if (std::fabs(sum - 1.0f) > 5e-3f) fail("cuda_softmax_last_dim_f16 row sum actual=" + std::to_string(sum) + " expected=1 tolerance=0.005");
    }

    for (int seq : {4, 5, 8, 16, 32}) test_gqa(seq, false);
    for (int seq : {4, 32}) test_gqa(seq, true);

    Tensor huge_q = Tensor::from_f32({4, 1, 1}, std::vector<float>(4, 1e10f));
    Tensor huge_k = Tensor::from_f32({4, 1, 1}, std::vector<float>(4, 1e10f));
    Tensor huge_v = Tensor::from_f32({4, 1, 1}, std::vector<float>(4, 2.0f));
    Tensor huge_out = cuda_gqa_attention(huge_q.to(DeviceType::CUDA), huge_k.to(DeviceType::CUDA), huge_v.to(DeviceType::CUDA)).to(DeviceType::CPU);
    for (size_t i = 0; i < huge_out.numel(); ++i) if (!std::isfinite(huge_out.get_f32(i))) fail("cuda_gqa_attention large finite input produced non-finite output");

    expect_throw("cuda_linear dtype mismatch", [&] { cuda_linear(dx, w16.to(DeviceType::CUDA)); });
    expect_throw("cuda_rms_norm dtype mismatch", [&] { cuda_rms_norm(dx, norm_w16.to(DeviceType::CUDA), 1e-5f); });
    Tensor q4 = gqa_input({4, 4, 2}, 1), k4 = gqa_input({4, 2, 2}, 1), v4 = gqa_input({4, 2, 2}, 1);
    expect_throw("cuda_gqa_attention dtype mismatch", [&] { cuda_gqa_attention(q4.to(DeviceType::CUDA), quantize_f16(k4).to(DeviceType::CUDA), quantize_f16(v4).to(DeviceType::CUDA)); });
    expect_throw("cuda_gqa_attention nondivisible heads", [&] { cuda_gqa_attention(gqa_input({4, 3, 2}, 1).to(DeviceType::CUDA), k4.to(DeviceType::CUDA), v4.to(DeviceType::CUDA)); });
    expect_throw("cuda_gqa_attention KV shape mismatch", [&] { cuda_gqa_attention(q4.to(DeviceType::CUDA), gqa_input({4, 2, 1}, 1).to(DeviceType::CUDA), v4.to(DeviceType::CUDA)); });
    expect_throw("cuda_gqa_attention CPU input", [&] { cuda_gqa_attention(q4, k4, v4); });
    expect_throw("cuda_gqa_attention unsupported sequence", [&] { cuda_gqa_attention(gqa_input({33, 4, 2}, 1).to(DeviceType::CUDA), gqa_input({33, 2, 2}, 1).to(DeviceType::CUDA), gqa_input({33, 2, 2}, 1).to(DeviceType::CUDA)); });
    expect_throw("cuda_rope invalid theta", [&] { cuda_rope(rope_x.to(DeviceType::CUDA), positions, 0); });
    expect_throw("cuda_rope positions mismatch", [&] { cuda_rope(rope_x.to(DeviceType::CUDA), {0}, 10000); });
    for (int i = 0; i < 100; ++i) { auto tmp = cuda_rope(rope_x.to(DeviceType::CUDA), positions, 10000); CUDA_CHECK(cudaDeviceSynchronize()); }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "test_ops_cuda passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
