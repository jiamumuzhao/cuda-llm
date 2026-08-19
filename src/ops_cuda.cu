#include "llm/ops_cuda.h"
#include "llm/cuda_check.h"
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <array>
#include <cmath>
#include <cublas_v2.h>
#include <exception>
#include <limits>
#include <memory>
#include <atomic>
#include <stdexcept>
#include <string>
#define __global__void __global__ void
#ifndef CUDART_INF_F
#define CUDART_INF_F 3.402823466e+38F
#endif
namespace llm {
namespace {
std::atomic<uint64_t> g_valid_lengths_h2d{0};
std::atomic<uint64_t> g_valid_lengths_d2h{0};
std::atomic<uint64_t> g_variable_decode_h2d{0};
std::atomic<uint64_t> g_variable_decode_d2h{0};
}
ValidLengthTransferStats valid_length_transfer_stats() {
  return {g_valid_lengths_h2d.load(std::memory_order_relaxed),
          g_valid_lengths_d2h.load(std::memory_order_relaxed)};
}
void reset_valid_length_transfer_stats() {
  g_valid_lengths_h2d.store(0, std::memory_order_relaxed);
  g_valid_lengths_d2h.store(0, std::memory_order_relaxed);
}
void record_valid_length_h2d_upload() {
  g_valid_lengths_h2d.fetch_add(1, std::memory_order_relaxed);
}
VariableDecodeTransferStats variable_decode_transfer_stats() {
  return {g_variable_decode_h2d.load(std::memory_order_relaxed),
          g_variable_decode_d2h.load(std::memory_order_relaxed)};
}
void reset_variable_decode_transfer_stats() {
  g_variable_decode_h2d.store(0, std::memory_order_relaxed);
  g_variable_decode_d2h.store(0, std::memory_order_relaxed);
}
void record_variable_decode_h2d_upload() {
  g_variable_decode_h2d.fetch_add(1, std::memory_order_relaxed);
}
static void req(const Tensor&t,const char*n){if(t.device()!=DeviceType::CUDA||!t.is_contiguous())throw std::invalid_argument(std::string(n)+": requires contiguous CUDA tensor");if(t.dtype()!=DType::F32&&t.dtype()!=DType::F16)throw std::invalid_argument(std::string(n)+": supports only F32/F16");}
static void rank(const Tensor&t,size_t r,const char*n){if(t.shape().size()!=r)throw std::invalid_argument(std::string(n)+": unexpected rank");}
static void same_dtype(const Tensor&a,const Tensor&b,const char*n){if(a.dtype()!=b.dtype())throw std::invalid_argument(std::string(n)+": all input dtypes must match; actual "+dtype_name(a.dtype())+" and "+dtype_name(b.dtype()));}
static void same_dtype(const Tensor&a,const Tensor&b,const Tensor&c,const char*n){if(a.dtype()!=b.dtype()||a.dtype()!=c.dtype())throw std::invalid_argument(std::string(n)+": all input dtypes must match; actual "+dtype_name(a.dtype())+", "+dtype_name(b.dtype())+", "+dtype_name(c.dtype()));}
static void output_req(const Tensor& output, const std::vector<int64_t>& shape,
                       DType dtype, const char* name) {
  if (output.device() != DeviceType::CUDA || output.dtype() != dtype ||
      !output.is_contiguous() || output.shape() != shape)
    throw std::invalid_argument(std::string(name) + ": output must be CUDA/" +
                                dtype_name(dtype) + "/contiguous shape " +
                                "with the expected dimensions");
}
template<class T> __device__ float rf(T x);template<> __device__ float rf<float>(float x){return x;}template<> __device__ float rf<__half>(__half x){return __half2float(x);}
template<class T> __device__ T wf(float x);template<> __device__ float wf<float>(float x){return x;}template<> __device__ __half wf<__half>(float x){return __float2half(x);}
template<class T>__global__void addk(const T*a,const T*b,T*o,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)o[i]=wf<T>(rf(a[i])+rf(b[i]));}
template<class T>__global__void siluk(const T*g,const T*u,T*o,size_t n){size_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float x=rf(g[i]);o[i]=wf<T>((x/(1.f+expf(-x)))*rf(u[i]));}}
__global__ void linear_f16_reference(const __half* x, const __half* w, __half* o,
                                     size_t tokens, size_t in, size_t out) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < tokens * out) {
    size_t token = index / out, column = index % out;
    float sum = 0.0f;
    for (size_t k = 0; k < in; ++k)
      sum += __half2float(x[token * in + k]) * __half2float(w[column * in + k]);
    o[index] = __float2half(sum);
  }
}
template<class T>__global__void rmsk(const T*x,const T*w,T*o,size_t rows,size_t h,float eps){size_t r=blockIdx.x*blockDim.x+threadIdx.x;if(r<rows){double s=0;for(size_t j=0;j<h;++j){double z=double(rf(x[r*h+j]));s+=z*z;}float inv=float(1.0/sqrt(s/double(h)+double(eps)));for(size_t j=0;j<h;++j){float z=__fmul_rn(rf(x[r*h+j]),inv);z=__fmul_rn(z,rf(w[j]));o[r*h+j]=wf<T>(z);}}}
template<class T>__global__void softk(const T*x,T*o,size_t rows,size_t d){size_t r=blockIdx.x*blockDim.x+threadIdx.x;if(r<rows){float m=-CUDART_INF_F;for(size_t j=0;j<d;++j)m=fmaxf(m,rf(x[r*d+j]));float sum=0;for(size_t j=0;j<d;++j){float e=expf(rf(x[r*d+j])-m);o[r*d+j]=wf<T>(e);sum+=e;}for(size_t j=0;j<d;++j)o[r*d+j]=wf<T>(rf(o[r*d+j])/sum);}}
template<class T>__global__void ropek(const T*x,T*o,const int32_t*p,size_t seq,size_t heads,size_t h,float theta){size_t i=blockIdx.x*blockDim.x+threadIdx.x,half=h/2,total=seq*heads*half;if(i<total){size_t j=i%half, z=i/half, t=z/heads, head=z%heads, base=(t*heads+head)*h;float a=rf(x[base+j]),b=rf(x[base+half+j]),ang=float(p[t])*powf(theta,-2.f*float(j)/float(h)),c=cosf(ang),s=sinf(ang);o[base+j]=wf<T>(a*c-b*s);o[base+half+j]=wf<T>(b*c+a*s);}}
template<class T> __global__ void rope_batched_kernel(const T* x, T* o, const int32_t* p, size_t batch, size_t seq, size_t heads, size_t h, float theta) { size_t i=blockIdx.x*blockDim.x+threadIdx.x, half=h/2, total=batch*seq*heads*half; if(i<total){ size_t j=i%half, z=i/half, t=z/heads, b=t/seq, token=t%seq, head=z%heads, base=((b*seq+token)*heads+head)*h; float a=rf(x[base+j]), c2=rf(x[base+half+j]), ang=float(p[token])*powf(theta,-2.f*float(j)/float(h)), c=cosf(ang), s=sinf(ang); o[base+j]=wf<T>(a*c-c2*s); o[base+half+j]=wf<T>(c2*c+a*s); } }
template<class T> __global__ void rope_decode_batched_kernel(const T* x, T* o, const int32_t* p, size_t batch, size_t heads, size_t h, float theta) { size_t i=blockIdx.x*blockDim.x+threadIdx.x, half=h/2, total=batch*heads*half; if(i<total){ size_t j=i%half, z=i/half, b=z/heads, head=z%heads, base=(b*heads+head)*h; float a=rf(x[base+j]), c2=rf(x[base+half+j]), ang=float(p[b])*powf(theta,-2.f*float(j)/float(h)), c=cosf(ang), s=sinf(ang); o[base+j]=wf<T>(a*c-c2*s); o[base+half+j]=wf<T>(c2*c+a*s); } }
template<class T>__global__void gqak(const T*q,const T*k,const T*v,T*o,size_t seq,size_t qh,size_t kh,size_t d,float scale){size_t idx=blockIdx.x*blockDim.x+threadIdx.x,total=seq*qh*d;if(idx<total){size_t dim=idx%d, qhead=(idx/d)%qh,t=idx/(qh*d),group=qh/kh,kvhead=qhead/group;float mx=-CUDART_INF_F;for(size_t u=0;u<=t;++u){float score=0;for(size_t e=0;e<d;++e)score+=rf(q[(t*qh+qhead)*d+e])*rf(k[(u*kh+kvhead)*d+e]);mx=fmaxf(mx,score*scale);}float den=0,z=0;for(size_t u=0;u<=t;++u){float score=0;for(size_t e=0;e<d;++e)score+=rf(q[(t*qh+qhead)*d+e])*rf(k[(u*kh+kvhead)*d+e]);float w=expf(score*scale-mx);den+=w;z+=w*rf(v[(u*kh+kvhead)*d+dim]);}o[idx]=wf<T>(z/den);}}
template<class T> __global__ void gqa_batched_kernel(const T* q, const T* k, const T* v, T* o, size_t batch, size_t seq, size_t qh, size_t kh, size_t d, float scale) { size_t idx=blockIdx.x*blockDim.x+threadIdx.x,total=batch*seq*qh*d; if(idx<total){ size_t dim=idx%d, z=idx/d, qhead=z%qh, token=(z/qh)%seq, b=z/(qh*seq), group=qh/kh, kvhead=qhead/group; float mx=-CUDART_INF_F; for(size_t u=0;u<=token;++u){float score=0;for(size_t e=0;e<d;++e)score+=rf(q[((b*seq+token)*qh+qhead)*d+e])*rf(k[((b*seq+u)*kh+kvhead)*d+e]);mx=fmaxf(mx,score*scale);} float den=0,out=0;for(size_t u=0;u<=token;++u){float score=0;for(size_t e=0;e<d;++e)score+=rf(q[((b*seq+token)*qh+qhead)*d+e])*rf(k[((b*seq+u)*kh+kvhead)*d+e]);float w=expf(score*scale-mx);den+=w;out+=w*rf(v[((b*seq+u)*kh+kvhead)*d+dim]);}o[idx]=wf<T>(out/den); } }
template<class T> __global__ void gqa_batched_valid_lengths_kernel(
    const T* q, const T* k, const T* v, const int32_t* valid_lengths, T* o,
    size_t batch, size_t seq, size_t qh, size_t kh, size_t d, float scale) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = batch * seq * qh * d;
  if (idx >= total) return;
  const size_t dim = idx % d;
  const size_t z = idx / d;
  const size_t qhead = z % qh;
  const size_t token = (z / qh) % seq;
  const size_t b = z / (qh * seq);
  const size_t valid = static_cast<size_t>(valid_lengths[b]);
  if (token >= valid) { o[idx] = wf<T>(0.0f); return; }
  const size_t kvhead = qhead / (qh / kh);
  float maximum = -CUDART_INF_F;
  for (size_t u = 0; u < valid && u <= token; ++u) {
    float score = 0.0f;
    for (size_t e = 0; e < d; ++e)
      score += rf(q[((b * seq + token) * qh + qhead) * d + e]) *
               rf(k[((b * seq + u) * kh + kvhead) * d + e]);
    maximum = fmaxf(maximum, score * scale);
  }
  float denominator = 0.0f;
  float result = 0.0f;
  for (size_t u = 0; u < valid && u <= token; ++u) {
    float score = 0.0f;
    for (size_t e = 0; e < d; ++e)
      score += rf(q[((b * seq + token) * qh + qhead) * d + e]) *
               rf(k[((b * seq + u) * kh + kvhead) * d + e]);
    const float weight = expf(score * scale - maximum);
    denominator += weight;
    result += weight * rf(v[((b * seq + u) * kh + kvhead) * d + dim]);
  }
  o[idx] = wf<T>(result / denominator);
}
template<class T>__global__void gqadk(const T*q,const T*k,const T*v,T*o,size_t cache_len,size_t qh,size_t kh,size_t d,float scale){size_t idx=blockIdx.x*blockDim.x+threadIdx.x,total=qh*d;if(idx<total){size_t dim=idx%d,qhead=idx/d,kvhead=qhead/(qh/kh);float mx=-CUDART_INF_F;for(size_t u=0;u<cache_len;++u){float score=0;for(size_t e=0;e<d;++e)score+=rf(q[qhead*d+e])*rf(k[(u*kh+kvhead)*d+e]);mx=fmaxf(mx,score*scale);}float den=0,z=0;for(size_t u=0;u<cache_len;++u){float score=0;for(size_t e=0;e<d;++e)score+=rf(q[qhead*d+e])*rf(k[(u*kh+kvhead)*d+e]);float w=expf(score*scale-mx);den+=w;z+=w*rf(v[(u*kh+kvhead)*d+dim]);}o[idx]=wf<T>(z/den);}}
__global__ void gqadk_batched(const __half* q, const __half* const* keys,
                               const __half* const* values, __half* o,
                               size_t batch, size_t cache_len, size_t qh,
                               size_t kh, size_t d, float scale) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = batch * qh * d;
  if (idx >= total) return;
  const size_t dim = idx % d;
  const size_t z = idx / d;
  const size_t qhead = z % qh;
  const size_t b = z / qh;
  const size_t kvhead = qhead / (qh / kh);
  const __half* k = keys[b];
  const __half* v = values[b];
  float maximum = -CUDART_INF_F;
  for (size_t u = 0; u < cache_len; ++u) {
    float score = 0.0f;
    for (size_t e = 0; e < d; ++e)
      score += __half2float(q[(b * qh + qhead) * d + e]) *
               __half2float(k[(u * kh + kvhead) * d + e]);
    maximum = fmaxf(maximum, score * scale);
  }
  float denominator = 0.0f;
  float result = 0.0f;
  for (size_t u = 0; u < cache_len; ++u) {
    float score = 0.0f;
    for (size_t e = 0; e < d; ++e)
      score += __half2float(q[(b * qh + qhead) * d + e]) *
               __half2float(k[(u * kh + kvhead) * d + e]);
    const float weight = expf(score * scale - maximum);
    denominator += weight;
    result += weight * __half2float(v[(u * kh + kvhead) * d + dim]);
  }
  o[idx] = __float2half(result / denominator);
}
__global__ void gqadk_batched_variable(const __half* q,
                                        const __half* const* keys,
                                        const __half* const* values,
                                        const int32_t* lengths, __half* o,
                                        size_t batch, size_t qh, size_t kh,
                                        size_t d, float scale) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = batch * qh * d;
  if (idx >= total) return;
  const size_t dim = idx % d, z = idx / d, qhead = z % qh, b = z / qh;
  const size_t length = static_cast<size_t>(lengths[b]);
  const size_t kvhead = qhead / (qh / kh);
  const __half* k = keys[b];
  const __half* v = values[b];
  float maximum = -CUDART_INF_F;
  for (size_t u = 0; u < length; ++u) {
    float score = 0.0f;
    for (size_t e = 0; e < d; ++e)
      score += __half2float(q[(b * qh + qhead) * d + e]) *
               __half2float(k[(u * kh + kvhead) * d + e]);
    maximum = fmaxf(maximum, score * scale);
  }
  float denominator = 0.0f, result = 0.0f;
  for (size_t u = 0; u < length; ++u) {
    float score = 0.0f;
    for (size_t e = 0; e < d; ++e)
      score += __half2float(q[(b * qh + qhead) * d + e]) *
               __half2float(k[(u * kh + kvhead) * d + e]);
    const float weight = expf(score * scale - maximum);
    denominator += weight;
    result += weight * __half2float(v[(u * kh + kvhead) * d + dim]);
  }
  o[idx] = __float2half(result / denominator);
}
template<class T>__global__void argmax_partial(const T*x,float*values,int32_t*ids,size_t vocab){__shared__ float sv[256];__shared__ int32_t si[256];size_t tid=threadIdx.x;float best=-CUDART_INF_F;int32_t best_id=0;for(size_t i=blockIdx.x*blockDim.x+tid;i<vocab;i+=gridDim.x*blockDim.x){float value=rf(x[i]);if(value>best||(value==best&&int32_t(i)<best_id)){best=value;best_id=int32_t(i);}}sv[tid]=best;si[tid]=best_id;__syncthreads();for(size_t stride=128;stride>0;stride>>=1){if(tid<stride){float value=sv[tid+stride];int32_t id=si[tid+stride];if(value>sv[tid]||(value==sv[tid]&&id<si[tid])){sv[tid]=value;si[tid]=id;}}__syncthreads();}if(tid==0){values[blockIdx.x]=sv[0];ids[blockIdx.x]=si[0];}}
__global__void argmax_final(const float*values,const int32_t*ids,int32_t*result,size_t count){__shared__ float sv[256];__shared__ int32_t si[256];size_t tid=threadIdx.x;float best=-CUDART_INF_F;int32_t best_id=0;for(size_t i=tid;i<count;i+=blockDim.x){float value=values[i];int32_t id=ids[i];if(value>best||(value==best&&id<best_id)){best=value;best_id=id;}}sv[tid]=best;si[tid]=best_id;__syncthreads();for(size_t stride=128;stride>0;stride>>=1){if(tid<stride){float value=sv[tid+stride];int32_t id=si[tid+stride];if(value>sv[tid]||(value==sv[tid]&&id<si[tid])){sv[tid]=value;si[tid]=id;}}__syncthreads();}if(tid==0)*result=si[0];}
template<class T>__global__void embeddingk(const T*w,const int32_t*ids,T*o,size_t tokens,size_t hidden,size_t vocab){size_t idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx<tokens*hidden){size_t t=idx/hidden,h=idx%hidden;int32_t id=ids[t];o[idx]=w[size_t(id)*hidden+h];}}
class DeviceBuffer { void* p_{}; public: explicit DeviceBuffer(size_t bytes){if(bytes==0)throw std::invalid_argument("cuda_rope: positions buffer must be non-empty");CUDA_CHECK(cudaMalloc(&p_,bytes));} ~DeviceBuffer() noexcept {if(p_ && cudaFree(p_)!=cudaSuccess)std::terminate();} void* get() const{return p_;} DeviceBuffer(const DeviceBuffer&)=delete; DeviceBuffer& operator=(const DeviceBuffer&)=delete;};
class DecodePointerWorkspace {
 public:
  DeviceBuffer keys{4 * sizeof(const __half*)};
  DeviceBuffer values{4 * sizeof(const __half*)};
};
static DecodePointerWorkspace& decode_pointer_workspace() {
  static DecodePointerWorkspace workspace;
  return workspace;
}
template<class T> __global__ void sampler_fill(const T* row, float* scores,
                                                int32_t* ids, size_t n,
                                                float temperature) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { scores[i] = rf(row[i]) / temperature; ids[i] = static_cast<int32_t>(i); }
}
__device__ unsigned long long sampler_mix(unsigned long long x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
__global__ void sampler_select(const float* scores, const int32_t* ids,
                               size_t n, size_t top_k, float top_p,
                               uint64_t seed, uint64_t offset,
                               int32_t* result) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const size_t limit = top_k == 0 ? n : min(top_k, n);
  const float maximum = scores[0];
  float total = 0.0f;
  for (size_t i = 0; i < limit; ++i) total += expf(scores[i] - maximum);
  size_t prefix = limit;
  if (top_p < 1.0f) {
    float cumulative = 0.0f;
    for (size_t i = 0; i < limit; ++i) {
      cumulative += expf(scores[i] - maximum) / total;
      if (cumulative >= top_p) { prefix = i + 1; break; }
    }
  }
  float prefix_total = 0.0f;
  for (size_t i = 0; i < prefix; ++i) prefix_total += expf(scores[i] - maximum);
  const unsigned long long bits = sampler_mix(seed ^ (offset * 0xd1342543de82ef95ULL));
  const float u = (static_cast<float>((bits >> 11) & 0x1fffffffffffffULL) + 0.5f) /
                  9007199254740992.0f;
  const float target = u * prefix_total;
  float cumulative = 0.0f;
  int32_t selected = ids[prefix - 1];
  for (size_t i = 0; i < prefix; ++i) {
    cumulative += expf(scores[i] - maximum);
    if (target < cumulative) { selected = ids[i]; break; }
  }
  *result = selected;
}
struct SamplerWorkspace {
  std::unique_ptr<DeviceBuffer> keys_in, keys_out, ids_in, ids_out, cub_temp, result;
  size_t n = 0;
  size_t cub_bytes = 0;
  explicit SamplerWorkspace(size_t vocab) : n(vocab) {
    keys_in = std::make_unique<DeviceBuffer>(sizeof(float) * n);
    keys_out = std::make_unique<DeviceBuffer>(sizeof(float) * n);
    ids_in = std::make_unique<DeviceBuffer>(sizeof(int32_t) * n);
    ids_out = std::make_unique<DeviceBuffer>(sizeof(int32_t) * n);
    result = std::make_unique<DeviceBuffer>(sizeof(int32_t));
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
        nullptr, bytes, static_cast<float*>(keys_in->get()),
        static_cast<float*>(keys_out->get()), static_cast<int32_t*>(ids_in->get()),
        static_cast<int32_t*>(ids_out->get()), static_cast<int>(n)));
    cub_bytes = bytes;
    cub_temp = std::make_unique<DeviceBuffer>(cub_bytes);
  }
};
class Blas{public:cublasHandle_t h{};Blas(){CUDA_CHECK(cublasCreate(&h));}~Blas(){if(cublasDestroy(h)!=CUBLAS_STATUS_SUCCESS)std::terminate();}};
void cuda_linear_out(const Tensor&x,const Tensor&w,Tensor&o){req(x,"cuda_linear");req(w,"cuda_linear");same_dtype(x,w,"cuda_linear");if(w.shape().size()!=2||x.shape().size()<2||x.shape().back()!=w.shape()[1])throw std::invalid_argument("cuda_linear: incompatible shapes");auto out_shape=x.shape();out_shape.back()=w.shape()[0];output_req(o,out_shape,x.dtype(),"cuda_linear");size_t tokens=x.numel()/x.shape().back(),in=w.shape()[1],out=w.shape()[0];if(x.dtype()==DType::F32){Blas b;float alpha=1,beta=0;CUDA_CHECK(cublasSgemm(b.h,CUBLAS_OP_T,CUBLAS_OP_N,int(out),int(tokens),int(in),&alpha,(const float*)w.data(),int(in),(const float*)x.data(),int(in),&beta,(float*)o.data(),int(out)));}else{size_t n=tokens*out;linear_f16_reference<<<(n+255)/256,256>>>((const __half*)x.data(),(const __half*)w.data(),(__half*)o.data(),tokens,in,out);CUDA_KERNEL_CHECK();}}
Tensor cuda_linear(const Tensor&x,const Tensor&w){req(x,"cuda_linear");req(w,"cuda_linear");same_dtype(x,w,"cuda_linear");if(w.shape().size()!=2||x.shape().size()<2||x.shape().back()!=w.shape()[1])throw std::invalid_argument("cuda_linear: incompatible shapes");auto out_shape=x.shape();out_shape.back()=w.shape()[0];Tensor o(x.dtype(),out_shape,DeviceType::CUDA);cuda_linear_out(x,w,o);return o;}
void cuda_rms_norm_out(const Tensor&x,const Tensor&w,float eps,Tensor&o){req(x,"cuda_rms_norm");req(w,"cuda_rms_norm");same_dtype(x,w,"cuda_rms_norm");rank(w,1,"cuda_rms_norm");if(x.shape().back()!=w.shape()[0]||eps<=0)throw std::invalid_argument("cuda_rms_norm: incompatible shape/epsilon");output_req(o,x.shape(),x.dtype(),"cuda_rms_norm");size_t rows=x.numel()/x.shape().back();if(x.dtype()==DType::F32)rmsk<<<(rows+127)/128,128>>>((float*)x.data(),(float*)w.data(),(float*)o.data(),rows,w.shape()[0],eps);else rmsk<<<(rows+127)/128,128>>>((__half*)x.data(),( __half*)w.data(),(__half*)o.data(),rows,w.shape()[0],eps);CUDA_KERNEL_CHECK();}
Tensor cuda_rms_norm(const Tensor&x,const Tensor&w,float eps){req(x,"cuda_rms_norm");req(w,"cuda_rms_norm");same_dtype(x,w,"cuda_rms_norm");rank(w,1,"cuda_rms_norm");if(x.shape().back()!=w.shape()[0]||eps<=0)throw std::invalid_argument("cuda_rms_norm: incompatible shape/epsilon");Tensor o(x.dtype(),x.shape(),DeviceType::CUDA);cuda_rms_norm_out(x,w,eps,o);return o;}
void cuda_add_out(const Tensor&a,const Tensor&b,Tensor&o){req(a,"cuda_add");req(b,"cuda_add");same_dtype(a,b,"cuda_add");if(a.shape()!=b.shape())throw std::invalid_argument("cuda_add: shapes must match");output_req(o,a.shape(),a.dtype(),"cuda_add");size_t n=a.numel();if(a.dtype()==DType::F32)addk<<<(n+255)/256,256>>>((float*)a.data(),(float*)b.data(),(float*)o.data(),n);else addk<<<(n+255)/256,256>>>((__half*)a.data(),(__half*)b.data(),(__half*)o.data(),n);CUDA_KERNEL_CHECK();}
Tensor cuda_add(const Tensor&a,const Tensor&b){req(a,"cuda_add");req(b,"cuda_add");same_dtype(a,b,"cuda_add");if(a.shape()!=b.shape())throw std::invalid_argument("cuda_add: shapes must match");Tensor o(a.dtype(),a.shape(),DeviceType::CUDA);cuda_add_out(a,b,o);return o;}
void cuda_swiglu_out(const Tensor&a,const Tensor&b,Tensor&o){req(a,"cuda_swiglu");req(b,"cuda_swiglu");same_dtype(a,b,"cuda_swiglu");if(a.shape()!=b.shape())throw std::invalid_argument("cuda_swiglu: shapes must match");output_req(o,a.shape(),a.dtype(),"cuda_swiglu");size_t n=a.numel();if(a.dtype()==DType::F32)siluk<<<(n+255)/256,256>>>((float*)a.data(),(float*)b.data(),(float*)o.data(),n);else siluk<<<(n+255)/256,256>>>((__half*)a.data(),(__half*)b.data(),(__half*)o.data(),n);CUDA_KERNEL_CHECK();}
Tensor cuda_swiglu(const Tensor&a,const Tensor&b){req(a,"cuda_swiglu");req(b,"cuda_swiglu");same_dtype(a,b,"cuda_swiglu");if(a.shape()!=b.shape())throw std::invalid_argument("cuda_swiglu: shapes must match");Tensor o(a.dtype(),a.shape(),DeviceType::CUDA);cuda_swiglu_out(a,b,o);return o;}
Tensor cuda_rope(const Tensor&x,const std::vector<int32_t>&p,float theta){req(x,"cuda_rope");rank(x,3,"cuda_rope");auto s=x.shape();if(s[0]==0||s[1]==0||s[2]==0||s[2]%2||p.empty()||p.size()!=size_t(s[0])||theta<=0)throw std::invalid_argument("cuda_rope: invalid or empty shape/positions/theta");Tensor o(x.dtype(),s,DeviceType::CUDA);DeviceBuffer dpos(p.size()*sizeof(int32_t));CUDA_CHECK(cudaMemcpy(dpos.get(),p.data(),p.size()*sizeof(int32_t),cudaMemcpyHostToDevice));size_t n=size_t(s[0])*size_t(s[1])*(size_t(s[2])/2);if(x.dtype()==DType::F32)ropek<<<(n+255)/256,256>>>((float*)x.data(),(float*)o.data(),static_cast<const int32_t*>(dpos.get()),s[0],s[1],s[2],theta);else ropek<<<(n+255)/256,256>>>((__half*)x.data(),(__half*)o.data(),static_cast<const int32_t*>(dpos.get()),s[0],s[1],s[2],theta);CUDA_KERNEL_CHECK();return o;}
Tensor cuda_rope_batched(const Tensor&x,const std::vector<int32_t>&p,float theta){req(x,"cuda_rope_batched");rank(x,4,"cuda_rope_batched");auto s=x.shape();if(s[0]==0||s[1]==0||s[2]==0||s[3]==0||s[3]%2||p.size()!=size_t(s[1])||theta<=0)throw std::invalid_argument("cuda_rope_batched: invalid shape/positions/theta");Tensor o(x.dtype(),s,DeviceType::CUDA);DeviceBuffer dpos(p.size()*sizeof(int32_t));CUDA_CHECK(cudaMemcpy(dpos.get(),p.data(),p.size()*sizeof(int32_t),cudaMemcpyHostToDevice));size_t n=size_t(s[0])*size_t(s[1])*size_t(s[2])*(size_t(s[3])/2);if(x.dtype()==DType::F32)rope_batched_kernel<<<(n+255)/256,256>>>((float*)x.data(),(float*)o.data(),(const int32_t*)dpos.get(),s[0],s[1],s[2],s[3],theta);else rope_batched_kernel<<<(n+255)/256,256>>>((__half*)x.data(),(__half*)o.data(),(const int32_t*)dpos.get(),s[0],s[1],s[2],s[3],theta);CUDA_KERNEL_CHECK();return o;}
void cuda_rope_batched_out(const Tensor&x,const Tensor&positions_device,float theta,Tensor&o){req(x,"cuda_rope_batched");rank(x,4,"cuda_rope_batched");auto s=x.shape();if(s[0]==0||s[1]==0||s[2]==0||s[3]==0||s[3]%2||theta<=0)throw std::invalid_argument("cuda_rope_batched: invalid shape/theta");if(positions_device.device()!=DeviceType::CUDA||positions_device.dtype()!=DType::F32||!positions_device.is_contiguous()||positions_device.numel()<static_cast<size_t>(s[1]))throw std::invalid_argument("cuda_rope_batched: positions buffer must be CUDA/F32/contiguous and large enough");output_req(o,s,x.dtype(),"cuda_rope_batched");size_t n=size_t(s[0])*size_t(s[1])*size_t(s[2])*(size_t(s[3])/2);if(x.dtype()==DType::F32)rope_batched_kernel<<<(n+255)/256,256>>>((float*)x.data(),(float*)o.data(),(const int32_t*)positions_device.data(),s[0],s[1],s[2],s[3],theta);else rope_batched_kernel<<<(n+255)/256,256>>>((__half*)x.data(),(__half*)o.data(),(const int32_t*)positions_device.data(),s[0],s[1],s[2],s[3],theta);CUDA_KERNEL_CHECK();}
Tensor cuda_softmax_last_dim(const Tensor&x){req(x,"cuda_softmax_last_dim");if(x.shape().empty()||x.shape().back()<=0)throw std::invalid_argument("cuda_softmax_last_dim: invalid last dimension");Tensor o(x.dtype(),x.shape(),DeviceType::CUDA);size_t d=x.shape().back(),rows=x.numel()/d;if(x.dtype()==DType::F32)softk<<<(rows+127)/128,128>>>((float*)x.data(),(float*)o.data(),rows,d);else softk<<<(rows+127)/128,128>>>((__half*)x.data(),(__half*)o.data(),rows,d);CUDA_KERNEL_CHECK();return o;}
static int32_t argmax_row_impl(const Tensor& logits, size_t row_offset, size_t vocab) { size_t blocks=std::min<size_t>(256,(vocab+255)/256); DeviceBuffer partial(sizeof(float)*blocks+sizeof(int32_t)*blocks), result(sizeof(int32_t)); auto* values=static_cast<float*>(partial.get()); auto* ids=reinterpret_cast<int32_t*>(static_cast<char*>(partial.get())+sizeof(float)*blocks); if(logits.dtype()==DType::F32)argmax_partial<<<blocks,256>>>((const float*)logits.data()+row_offset,values,ids,vocab); else argmax_partial<<<blocks,256>>>((const __half*)logits.data()+row_offset,values,ids,vocab); CUDA_KERNEL_CHECK(); argmax_final<<<1,256>>>(values,ids,static_cast<int32_t*>(result.get()),blocks); CUDA_KERNEL_CHECK(); int32_t host=0; CUDA_CHECK(cudaMemcpy(&host,result.get(),sizeof(host),cudaMemcpyDeviceToHost)); return host; }
Tensor cuda_gqa_attention(const Tensor&q,const Tensor&k,const Tensor&v){req(q,"cuda_gqa_attention");req(k,"cuda_gqa_attention");req(v,"cuda_gqa_attention");same_dtype(q,k,v,"cuda_gqa_attention");rank(q,3,"cuda_gqa_attention");rank(k,3,"cuda_gqa_attention");rank(v,3,"cuda_gqa_attention");auto a=q.shape(),b=k.shape(),c=v.shape();if(a[0]<1||a[0]>32)throw std::invalid_argument("cuda_gqa_attention: supported sequence lengths are 1..32");if(a[0]!=b[0]||a[0]!=c[0]||a[2]!=b[2]||a[2]!=c[2]||b[1]!=c[1]||b[1]==0||a[1]==0||a[2]==0||a[1]%b[1])throw std::invalid_argument("cuda_gqa_attention: incompatible shapes");Tensor o(q.dtype(),a,DeviceType::CUDA);size_t n=size_t(a[0])*size_t(a[1])*size_t(a[2]);float scale=1.f/sqrtf(float(a[2]));if(q.dtype()==DType::F32)gqak<<<(n+255)/256,256>>>((float*)q.data(),(float*)k.data(),(float*)v.data(),(float*)o.data(),a[0],a[1],b[1],a[2],scale);else gqak<<<(n+255)/256,256>>>((__half*)q.data(),(__half*)k.data(),(__half*)v.data(),(__half*)o.data(),a[0],a[1],b[1],a[2],scale);CUDA_KERNEL_CHECK();return o;}
Tensor cuda_gqa_attention_batched(const Tensor&q,const Tensor&k,const Tensor&v){req(q,"cuda_gqa_attention_batched");req(k,"cuda_gqa_attention_batched");req(v,"cuda_gqa_attention_batched");same_dtype(q,k,v,"cuda_gqa_attention_batched");rank(q,4,"cuda_gqa_attention_batched");rank(k,4,"cuda_gqa_attention_batched");rank(v,4,"cuda_gqa_attention_batched");auto a=q.shape(),b=k.shape(),c=v.shape();if(a[0]==0||a[0]>4||a[1]==0||a[1]>32||a[2]!=16||a[3]!=128||b[0]!=a[0]||b[1]!=a[1]||b[2]!=8||b[3]!=128||c!=b)throw std::invalid_argument("cuda_gqa_attention_batched: expected q=[B,S,16,128], k/v=[B,S,8,128], B<=4 and S<=32");Tensor o(q.dtype(),a,DeviceType::CUDA);size_t n=size_t(a[0])*size_t(a[1])*size_t(a[2])*size_t(a[3]);float scale=1.f/sqrtf(128.f);if(q.dtype()==DType::F32)gqa_batched_kernel<<<(n+255)/256,256>>>((float*)q.data(),(float*)k.data(),(float*)v.data(),(float*)o.data(),a[0],a[1],16,8,128,scale);else gqa_batched_kernel<<<(n+255)/256,256>>>((__half*)q.data(),(__half*)k.data(),(__half*)v.data(),(__half*)o.data(),a[0],a[1],16,8,128,scale);CUDA_KERNEL_CHECK();return o;}
Tensor cuda_gqa_attention_batched_valid_lengths_checked(
    const Tensor&q,const Tensor&k,const Tensor&v,const Tensor&valid_lengths) {
  req(q, "cuda_gqa_attention_batched_valid_lengths");
  req(k, "cuda_gqa_attention_batched_valid_lengths");
  req(v, "cuda_gqa_attention_batched_valid_lengths");
  same_dtype(q, k, v, "cuda_gqa_attention_batched_valid_lengths");
  rank(q, 4, "cuda_gqa_attention_batched_valid_lengths");
  rank(k, 4, "cuda_gqa_attention_batched_valid_lengths");
  rank(v, 4, "cuda_gqa_attention_batched_valid_lengths");
  const auto a = q.shape(), b = k.shape(), c = v.shape();
  if (a[0] == 0 || a[0] > 4 || a[1] == 0 || a[1] > 32 || a[2] != 16 || a[3] != 128 ||
      b[0] != a[0] || b[1] != a[1] || b[2] != 8 || b[3] != 128 || c != b)
    throw std::invalid_argument("cuda_gqa_attention_batched_valid_lengths: expected q=[B,S,16,128], k/v=[B,S,8,128], B<=4 and S<=32");
  if (valid_lengths.device() != DeviceType::CUDA || valid_lengths.dtype() != DType::I32 ||
      !valid_lengths.is_contiguous() || valid_lengths.shape() != std::vector<int64_t>{a[0]})
    throw std::invalid_argument("cuda_gqa_attention_batched_valid_lengths: valid_lengths must be CUDA/I32/contiguous shape [B]");
  Tensor o(q.dtype(), a, DeviceType::CUDA);
  const size_t n = static_cast<size_t>(a[0]) * static_cast<size_t>(a[1]) * 16 * 128;
  const float scale = 1.0f / sqrtf(128.0f);
  if (q.dtype() == DType::F32)
    gqa_batched_valid_lengths_kernel<<<(n + 255) / 256, 256>>>((const float*)q.data(), (const float*)k.data(), (const float*)v.data(), (const int32_t*)valid_lengths.data(), (float*)o.data(), a[0], a[1], 16, 8, 128, scale);
  else
    gqa_batched_valid_lengths_kernel<<<(n + 255) / 256, 256>>>((const __half*)q.data(), (const __half*)k.data(), (const __half*)v.data(), (const int32_t*)valid_lengths.data(), (__half*)o.data(), a[0], a[1], 16, 8, 128, scale);
  CUDA_KERNEL_CHECK();
  return o;
}
Tensor cuda_gqa_attention_batched_valid_lengths(
    const Tensor&q,const Tensor&k,const Tensor&v,const Tensor&valid_lengths) {
  if (valid_lengths.device() != DeviceType::CUDA || valid_lengths.dtype() != DType::I32 ||
      !valid_lengths.is_contiguous() || valid_lengths.shape().size() != 1 ||
      valid_lengths.shape()[0] <= 0)
    throw std::invalid_argument("cuda_gqa_attention_batched_valid_lengths: valid_lengths must be CUDA/I32/contiguous rank-1");
  std::vector<int32_t> host_lengths(static_cast<size_t>(valid_lengths.shape()[0]));
  CUDA_CHECK(cudaMemcpy(host_lengths.data(), valid_lengths.data(), host_lengths.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  g_valid_lengths_d2h.fetch_add(1, std::memory_order_relaxed);
  const size_t seq = q.shape().size() > 1 ? static_cast<size_t>(q.shape()[1]) : 0;
  for (size_t i = 0; i < host_lengths.size(); ++i)
    if (host_lengths[i] <= 0 || static_cast<size_t>(host_lengths[i]) > seq)
      throw std::invalid_argument("cuda_gqa_attention_batched_valid_lengths: valid_lengths[" + std::to_string(i) + "] must be in [1,S]");
  return cuda_gqa_attention_batched_valid_lengths_checked(q, k, v, valid_lengths);
}
Tensor cuda_gqa_decode_attention(const Tensor&q,const Tensor&k,const Tensor&v,size_t cache_length){req(q,"cuda_gqa_decode_attention");req(k,"cuda_gqa_decode_attention");req(v,"cuda_gqa_decode_attention");same_dtype(q,k,v,"cuda_gqa_decode_attention");rank(q,3,"cuda_gqa_decode_attention");rank(k,3,"cuda_gqa_decode_attention");rank(v,3,"cuda_gqa_decode_attention");auto a=q.shape(),b=k.shape(),c=v.shape();if(a[0]!=1||a[1]!=16||a[2]!=128||b[1]!=8||b[2]!=128||c!=b||cache_length==0||cache_length>size_t(b[0]))throw std::invalid_argument("cuda_gqa_decode_attention: expected q=[1,16,128], k/v=[capacity,8,128], and 0<cache_length<=capacity");Tensor o(q.dtype(),a,DeviceType::CUDA);size_t n=16*128;float scale=1.f/sqrtf(128.f);if(q.dtype()==DType::F32)gqadk<<<(n+255)/256,256>>>((float*)q.data(),(float*)k.data(),(float*)v.data(),(float*)o.data(),cache_length,16,8,128,scale);else gqadk<<<(n+255)/256,256>>>((__half*)q.data(),(__half*)k.data(),(__half*)v.data(),(__half*)o.data(),cache_length,16,8,128,scale);CUDA_KERNEL_CHECK();return o;}
Tensor cuda_gqa_decode_attention_batched(
    const Tensor& q, const std::vector<const Tensor*>& keys,
    const std::vector<const Tensor*>& values, size_t cache_length) {
  req(q, "cuda_gqa_decode_attention_batched");
  if (q.dtype() != DType::F16 || q.shape() != std::vector<int64_t>{static_cast<int64_t>(keys.size()), 16, 128})
    throw std::invalid_argument("cuda_gqa_decode_attention_batched: q must be CUDA/F16/contiguous [B,16,128]");
  const size_t batch = keys.size();
  if ((batch != 1 && batch != 2 && batch != 4) || values.size() != batch)
    throw std::invalid_argument("cuda_gqa_decode_attention_batched: batch must be 1, 2, or 4 and key/value counts must match");
  if (cache_length == 0) throw std::invalid_argument("cuda_gqa_decode_attention_batched: cache_length must be > 0");
  std::array<const __half*, 4> key_ptrs{};
  std::array<const __half*, 4> value_ptrs{};
  for (size_t b = 0; b < batch; ++b) {
    if (keys[b] == nullptr || values[b] == nullptr)
      throw std::invalid_argument("cuda_gqa_decode_attention_batched: cache pointer must be non-null");
    const Tensor& k = *keys[b];
    const Tensor& v = *values[b];
    if (k.device() != DeviceType::CUDA || v.device() != DeviceType::CUDA ||
        k.dtype() != DType::F16 || v.dtype() != DType::F16 ||
        !k.is_contiguous() || !v.is_contiguous() ||
        k.shape().size() != 3 || k.shape()[1] != 8 || k.shape()[2] != 128 ||
        v.shape() != k.shape() || cache_length > static_cast<size_t>(k.shape()[0]))
      throw std::invalid_argument("cuda_gqa_decode_attention_batched: each cache must be CUDA/F16/contiguous [capacity,8,128] with capacity >= cache_length");
    key_ptrs[b] = static_cast<const __half*>(k.data());
    value_ptrs[b] = static_cast<const __half*>(v.data());
  }
  DecodePointerWorkspace& workspace = decode_pointer_workspace();
  CUDA_CHECK(cudaMemcpy(workspace.keys.get(), key_ptrs.data(), batch * sizeof(const __half*), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(workspace.values.get(), value_ptrs.data(), batch * sizeof(const __half*), cudaMemcpyHostToDevice));
  Tensor output(DType::F16, {static_cast<int64_t>(batch), 16, 128}, DeviceType::CUDA);
  const size_t n = batch * 16 * 128;
  gqadk_batched<<<(n + 255) / 256, 256>>>(
      static_cast<const __half*>(q.data()),
      static_cast<const __half* const*>(workspace.keys.get()),
      static_cast<const __half* const*>(workspace.values.get()),
      static_cast<__half*>(output.data()), batch, cache_length, 16, 8, 128,
      1.0f / sqrtf(128.0f));
  CUDA_KERNEL_CHECK();
  return output;
}
void cuda_gqa_decode_attention_batched_out(const Tensor&q,const std::vector<const Tensor*>&keys,const std::vector<const Tensor*>&values,size_t cache_length,Tensor&o){req(q,"cuda_gqa_decode_attention_batched");if(q.dtype()!=DType::F16||q.shape()!=std::vector<int64_t>{static_cast<int64_t>(keys.size()),16,128})throw std::invalid_argument("cuda_gqa_decode_attention_batched: q shape/dtype mismatch");const size_t batch=keys.size();if((batch!=1&&batch!=2&&batch!=4)||values.size()!=batch)throw std::invalid_argument("cuda_gqa_decode_attention_batched: batch/key/value count mismatch");if(cache_length==0)throw std::invalid_argument("cuda_gqa_decode_attention_batched: cache_length must be > 0");std::array<const __half*,4> key_ptrs{};std::array<const __half*,4> value_ptrs{};for(size_t b=0;b<batch;++b){if(!keys[b]||!values[b])throw std::invalid_argument("cuda_gqa_decode_attention_batched: null cache pointer");const Tensor&k=*keys[b];const Tensor&v=*values[b];if(k.device()!=DeviceType::CUDA||v.device()!=DeviceType::CUDA||k.dtype()!=DType::F16||v.dtype()!=DType::F16||!k.is_contiguous()||!v.is_contiguous()||k.shape().size()!=3||k.shape()[1]!=8||k.shape()[2]!=128||v.shape()!=k.shape()||cache_length>static_cast<size_t>(k.shape()[0]))throw std::invalid_argument("cuda_gqa_decode_attention_batched: cache shape mismatch");key_ptrs[b]=static_cast<const __half*>(k.data());value_ptrs[b]=static_cast<const __half*>(v.data());}DecodePointerWorkspace&workspace=decode_pointer_workspace();CUDA_CHECK(cudaMemcpy(workspace.keys.get(),key_ptrs.data(),batch*sizeof(const __half*),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(workspace.values.get(),value_ptrs.data(),batch*sizeof(const __half*),cudaMemcpyHostToDevice));output_req(o,{static_cast<int64_t>(batch),16,128},DType::F16,"cuda_gqa_decode_attention_batched");const size_t n=batch*16*128;gqadk_batched<<<(n+255)/256,256>>>(static_cast<const __half*>(q.data()),static_cast<const __half*const*>(workspace.keys.get()),static_cast<const __half*const*>(workspace.values.get()),static_cast<__half*>(o.data()),batch,cache_length,16,8,128,1.0f/sqrtf(128.0f));CUDA_KERNEL_CHECK();}
int32_t cuda_argmax_last_row(const Tensor&logits){if(logits.device()!=DeviceType::CUDA||!logits.is_contiguous())throw std::invalid_argument("cuda_argmax_last_row: requires contiguous CUDA tensor");if(logits.dtype()!=DType::F32&&logits.dtype()!=DType::F16)throw std::invalid_argument("cuda_argmax_last_row: supports only F32/F16");rank(logits,2,"cuda_argmax_last_row");if(logits.shape()[0]<=0||logits.shape()[1]<=0)throw std::invalid_argument("cuda_argmax_last_row: shape dimensions must be > 0");const size_t vocab=size_t(logits.shape()[1]);return argmax_row_impl(logits,(size_t(logits.shape()[0])-1)*vocab,vocab);}
void cuda_embedding_lookup_out(const Tensor&w,const std::vector<int32_t>&ids,Tensor&ids_device,Tensor&out){req(w,"cuda_embedding_lookup");rank(w,2,"cuda_embedding_lookup");if(ids.empty())throw std::invalid_argument("cuda_embedding_lookup: token_ids must be non-empty");const int64_t vocab=w.shape()[0],hidden=w.shape()[1];if(vocab<=0||hidden<=0)throw std::invalid_argument("cuda_embedding_lookup: weight shape must be positive");for(size_t i=0;i<ids.size();++i)if(ids[i]<0||int64_t(ids[i])>=vocab)throw std::invalid_argument("cuda_embedding_lookup: token id out of range");if(ids_device.device()!=DeviceType::CUDA||ids_device.dtype()!=DType::F32||!ids_device.is_contiguous()||ids_device.numel()<ids.size())throw std::invalid_argument("cuda_embedding_lookup: token storage must be CUDA/F32/contiguous");output_req(out,{static_cast<int64_t>(ids.size()),hidden},w.dtype(),"cuda_embedding_lookup");CUDA_CHECK(cudaMemcpy(ids_device.data(),ids.data(),ids.size()*sizeof(int32_t),cudaMemcpyHostToDevice));size_t n=ids.size()*size_t(hidden);if(w.dtype()==DType::F32)embeddingk<<<(n+255)/256,256>>>((const float*)w.data(),(const int32_t*)ids_device.data(),(float*)out.data(),ids.size(),hidden,vocab);else embeddingk<<<(n+255)/256,256>>>((const __half*)w.data(),(const int32_t*)ids_device.data(),(__half*)out.data(),ids.size(),hidden,vocab);CUDA_KERNEL_CHECK();}
Tensor cuda_embedding_lookup(const Tensor&w,const std::vector<int32_t>&ids){req(w,"cuda_embedding_lookup");rank(w,2,"cuda_embedding_lookup");if(ids.empty())throw std::invalid_argument("cuda_embedding_lookup: token_ids must be non-empty");if(w.shape()[0]<=0||w.shape()[1]<=0)throw std::invalid_argument("cuda_embedding_lookup: weight shape must be positive");Tensor ids_device(DType::F32,{static_cast<int64_t>(ids.size())},DeviceType::CUDA);Tensor out(w.dtype(),{static_cast<int64_t>(ids.size()),w.shape()[1]},DeviceType::CUDA);cuda_embedding_lookup_out(w,ids,ids_device,out);return out;}
Tensor cuda_lm_head(const Tensor&hidden,const Tensor&embedding){if(hidden.device()!=DeviceType::CUDA||embedding.device()!=DeviceType::CUDA)throw std::invalid_argument("cuda_lm_head: hidden and tied embedding must be CUDA");if(hidden.shape().size()!=2||embedding.shape().size()!=2)throw std::invalid_argument("cuda_lm_head: hidden and embedding must be rank-2");if(hidden.shape().back()!=embedding.shape()[1])throw std::invalid_argument("cuda_lm_head: hidden_size mismatch");same_dtype(hidden,embedding,"cuda_lm_head");return cuda_linear(hidden,embedding);}
CudaSampler::CudaSampler(size_t max_vocab) : max_vocab_(max_vocab) {
  if (max_vocab == 0) throw std::invalid_argument("CudaSampler: max_vocab must be > 0");
  workspace_ = new SamplerWorkspace(max_vocab);
}
CudaSampler::~CudaSampler() { delete static_cast<SamplerWorkspace*>(workspace_); }
int32_t CudaSampler::sample_row_offset(const Tensor& logits, size_t row_offset,
                                const SamplingConfig& config, uint64_t draw_offset) {
  if (logits.device() != DeviceType::CUDA || !logits.is_contiguous())
    throw std::invalid_argument("CudaSampler::sample_last_row: requires contiguous CUDA tensor");
  if (logits.dtype() != DType::F32 && logits.dtype() != DType::F16)
    throw std::invalid_argument("CudaSampler::sample_last_row: supports F32/F16 only");
  if ((logits.shape().size() != 2 && logits.shape().size() != 3) || logits.shape().empty() || logits.shape().back() == 0)
    throw std::invalid_argument("CudaSampler::sample_last_row: requires non-empty rank-2 or rank-3 logits");
  const size_t vocab = static_cast<size_t>(logits.shape().back());
  if (vocab > max_vocab_) throw std::invalid_argument("CudaSampler::sample_last_row: vocab exceeds sampler capacity");
  if (!std::isfinite(config.temperature) || config.temperature < 0.0f)
    throw std::invalid_argument("CudaSampler::sample_last_row: temperature must be finite and >= 0");
  if (config.temperature == 0.0f) {
    const bool is_last_rank2_row =
        logits.shape().size() == 2 &&
        row_offset == (static_cast<size_t>(logits.shape()[0]) - 1) * vocab;
    return is_last_rank2_row ? cuda_argmax_last_row(logits)
                            : argmax_row_impl(logits, row_offset, vocab);
  }
  if (config.top_k < 0 || static_cast<size_t>(config.top_k) > vocab)
    throw std::invalid_argument("CudaSampler::sample_last_row: top_k must be in [0,vocab]");
  if (!std::isfinite(config.top_p) || config.top_p <= 0.0f || config.top_p > 1.0f)
    throw std::invalid_argument("CudaSampler::sample_last_row: top_p must be finite in (0,1]");
  auto* ws = static_cast<SamplerWorkspace*>(workspace_);
  const size_t blocks = (vocab + 255) / 256;
  if (logits.dtype() == DType::F32)
    sampler_fill<<<blocks, 256>>>((const float*)logits.data() + row_offset,
                                  (float*)ws->keys_in->get(), (int32_t*)ws->ids_in->get(),
                                  vocab, config.temperature);
  else
    sampler_fill<<<blocks, 256>>>((const __half*)logits.data() + row_offset,
                                  (float*)ws->keys_in->get(), (int32_t*)ws->ids_in->get(),
                                  vocab, config.temperature);
  CUDA_KERNEL_CHECK();
  CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
      ws->cub_temp->get(), ws->cub_bytes, (float*)ws->keys_in->get(), (float*)ws->keys_out->get(),
      (int32_t*)ws->ids_in->get(), (int32_t*)ws->ids_out->get(), static_cast<int>(vocab)));
  sampler_select<<<1, 1>>>((const float*)ws->keys_out->get(),
                           (const int32_t*)ws->ids_out->get(), vocab,
                           static_cast<size_t>(config.top_k), config.top_p,
                           config.seed, draw_offset,
                           (int32_t*)ws->result->get());
  CUDA_KERNEL_CHECK();
  int32_t result = 0;
  CUDA_CHECK(cudaMemcpy(&result, ws->result->get(), sizeof(result), cudaMemcpyDeviceToHost));
  return result;
}
int32_t CudaSampler::sample_last_row(const Tensor& logits, const SamplingConfig& config,
                                     uint64_t draw_offset) {
  if (logits.shape().size() != 2 || logits.shape()[0] == 0)
    throw std::invalid_argument("CudaSampler::sample_last_row: requires non-empty rank-2 logits");
  return sample_row_offset(logits, (static_cast<size_t>(logits.shape()[0]) - 1) *
                              static_cast<size_t>(logits.shape()[1]), config, draw_offset);
}
int32_t CudaSampler::sample_last_token_of_batch(const Tensor& logits, size_t batch_index,
                                                const SamplingConfig& config,
                                                uint64_t draw_offset) {
  if (logits.device() != DeviceType::CUDA || !logits.is_contiguous())
    throw std::invalid_argument("CudaSampler::sample_last_token_of_batch: requires contiguous CUDA tensor");
  if (logits.dtype() != DType::F32 && logits.dtype() != DType::F16)
    throw std::invalid_argument("CudaSampler::sample_last_token_of_batch: supports F32/F16 only");
  if (logits.shape().size() != 3 || logits.shape()[0] == 0 || logits.shape()[1] == 0 || logits.shape()[2] == 0)
    throw std::invalid_argument("CudaSampler::sample_last_token_of_batch: requires non-empty rank-3 logits");
  if (batch_index >= static_cast<size_t>(logits.shape()[0]))
    throw std::out_of_range("CudaSampler::sample_last_token_of_batch: batch_index out of range");
  const size_t vocab = static_cast<size_t>(logits.shape()[2]);
  return sample_row_offset(logits, (batch_index * static_cast<size_t>(logits.shape()[1]) +
                             static_cast<size_t>(logits.shape()[1]) - 1) * vocab,
                    config, draw_offset);
}
int32_t CudaSampler::sample_row(const Tensor& logits, size_t batch_index,
                                const SamplingConfig& config,
                                uint64_t draw_offset) {
  if (logits.shape().size() != 2 || logits.shape()[0] == 0 ||
      logits.shape()[1] == 0)
    throw std::invalid_argument("CudaSampler::sample_row: requires non-empty rank-2 logits");
  if (batch_index >= static_cast<size_t>(logits.shape()[0]))
    throw std::out_of_range("CudaSampler::sample_row: batch_index out of range");
  return sample_row_offset(logits, batch_index * static_cast<size_t>(logits.shape()[1]),
                           config, draw_offset);
}
void cuda_rope_batched_positions_out(const Tensor& x, const Tensor& positions_device,
                                     float theta, Tensor& o) {
  req(x, "cuda_rope_batched_positions");
  if (x.shape().size() != 3 || x.shape()[0] <= 0 ||
      (x.shape()[1] != 16 && x.shape()[1] != 8) ||
      x.shape()[2] != 128 || positions_device.device() != DeviceType::CUDA ||
      positions_device.dtype() != DType::I32 || !positions_device.is_contiguous() ||
      positions_device.shape() != std::vector<int64_t>{x.shape()[0]} || theta <= 0.0f)
    throw std::invalid_argument("cuda_rope_batched_positions: expected input CUDA/F16 [B,16,128] and positions CUDA/I32 [B]");
  output_req(o, x.shape(), x.dtype(), "cuda_rope_batched_positions");
  const size_t n = static_cast<size_t>(x.shape()[0]) *
                   static_cast<size_t>(x.shape()[1]) * 64;
  rope_decode_batched_kernel<<<(n + 255) / 256, 256>>>(
      static_cast<const __half*>(x.data()), static_cast<__half*>(o.data()),
      static_cast<const int32_t*>(positions_device.data()),
      static_cast<size_t>(x.shape()[0]), static_cast<size_t>(x.shape()[1]),
      128, theta);
  CUDA_KERNEL_CHECK();
}
static void validate_variable_decode_static(
    const Tensor& q, const std::vector<const Tensor*>& keys,
    const std::vector<const Tensor*>& values, const Tensor& lengths) {
  req(q, "cuda_gqa_decode_attention_batched_variable_lengths");
  const size_t batch = keys.size();
  if (q.dtype() != DType::F16 || q.shape() !=
      std::vector<int64_t>{static_cast<int64_t>(batch), 16, 128} ||
      (batch != 1 && batch != 2 && batch != 4) || values.size() != batch)
    throw std::invalid_argument("cuda_gqa_decode_attention_batched_variable_lengths: expected CUDA/F16 q=[B,16,128], B in {1,2,4}, matching K/V");
  if (lengths.device() != DeviceType::CUDA || lengths.dtype() != DType::I32 ||
      !lengths.is_contiguous() || lengths.shape() !=
      std::vector<int64_t>{static_cast<int64_t>(batch)})
    throw std::invalid_argument("cuda_gqa_decode_attention_batched_variable_lengths: lengths must be CUDA/I32/contiguous [B]");
  for (size_t b = 0; b < batch; ++b) {
    if (!keys[b] || !values[b])
      throw std::invalid_argument("cuda_gqa_decode_attention_batched_variable_lengths: cache pointer must be non-null");
    const Tensor& k = *keys[b]; const Tensor& v = *values[b];
    if (k.device() != DeviceType::CUDA || v.device() != DeviceType::CUDA ||
        k.dtype() != DType::F16 || v.dtype() != DType::F16 ||
        !k.is_contiguous() || !v.is_contiguous() || k.shape().size() != 3 ||
        k.shape()[1] != 8 || k.shape()[2] != 128 || v.shape() != k.shape())
      throw std::invalid_argument("cuda_gqa_decode_attention_batched_variable_lengths: each K/V cache must be CUDA/F16/contiguous [capacity,8,128]");
  }
}
Tensor cuda_gqa_decode_attention_batched_variable_lengths_checked(
    const Tensor& q, const std::vector<const Tensor*>& keys,
    const std::vector<const Tensor*>& values, const Tensor& lengths) {
  validate_variable_decode_static(q, keys, values, lengths);
  Tensor output(DType::F16, q.shape(), DeviceType::CUDA);
  const size_t batch = keys.size(), n = batch * 16 * 128;
  std::array<const __half*, 4> key_ptrs{};
  std::array<const __half*, 4> value_ptrs{};
  for (size_t b = 0; b < batch; ++b) {
    key_ptrs[b] = static_cast<const __half*>(keys[b]->data());
    value_ptrs[b] = static_cast<const __half*>(values[b]->data());
  }
  DecodePointerWorkspace& workspace = decode_pointer_workspace();
  CUDA_CHECK(cudaMemcpy(workspace.keys.get(), key_ptrs.data(), batch * sizeof(const __half*), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(workspace.values.get(), value_ptrs.data(), batch * sizeof(const __half*), cudaMemcpyHostToDevice));
  gqadk_batched_variable<<<(n + 255) / 256, 256>>>(
      static_cast<const __half*>(q.data()),
      static_cast<const __half* const*>(workspace.keys.get()),
      static_cast<const __half* const*>(workspace.values.get()),
      static_cast<const int32_t*>(lengths.data()),
      static_cast<__half*>(output.data()), batch, 16, 8, 128,
      1.0f / sqrtf(128.0f));
  CUDA_KERNEL_CHECK();
  return output;
}
Tensor cuda_gqa_decode_attention_batched_variable_lengths(
    const Tensor& q, const std::vector<const Tensor*>& keys,
    const std::vector<const Tensor*>& values, const Tensor& lengths) {
  validate_variable_decode_static(q, keys, values, lengths);
  std::vector<int32_t> host(lengths.numel());
  CUDA_CHECK(cudaMemcpy(host.data(), lengths.data(), host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
  g_variable_decode_d2h.fetch_add(1, std::memory_order_relaxed);
  for (size_t b = 0; b < host.size(); ++b)
    if (host[b] <= 0 || static_cast<size_t>(host[b]) > static_cast<size_t>(keys[b]->shape()[0]))
      throw std::invalid_argument("cuda_gqa_decode_attention_batched_variable_lengths: length must be in [1,capacity]");
  return cuda_gqa_decode_attention_batched_variable_lengths_checked(q, keys, values, lengths);
}
void cuda_gqa_decode_attention_batched_variable_lengths_out(
    const Tensor& q, const std::vector<const Tensor*>& keys,
    const std::vector<const Tensor*>& values, const Tensor& lengths, Tensor& output) {
  validate_variable_decode_static(q, keys, values, lengths);
  output_req(output, q.shape(), DType::F16, "cuda_gqa_decode_attention_batched_variable_lengths");
  const size_t batch = keys.size(), n = batch * 16 * 128;
  std::array<const __half*, 4> key_ptrs{};
  std::array<const __half*, 4> value_ptrs{};
  for (size_t b = 0; b < batch; ++b) {
    key_ptrs[b] = static_cast<const __half*>(keys[b]->data());
    value_ptrs[b] = static_cast<const __half*>(values[b]->data());
  }
  DecodePointerWorkspace& workspace = decode_pointer_workspace();
  CUDA_CHECK(cudaMemcpy(workspace.keys.get(), key_ptrs.data(), batch * sizeof(const __half*), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(workspace.values.get(), value_ptrs.data(), batch * sizeof(const __half*), cudaMemcpyHostToDevice));
  gqadk_batched_variable<<<(n + 255) / 256, 256>>>(
      static_cast<const __half*>(q.data()),
      static_cast<const __half* const*>(workspace.keys.get()),
      static_cast<const __half* const*>(workspace.values.get()),
      static_cast<const int32_t*>(lengths.data()), static_cast<__half*>(output.data()),
      batch, 16, 8, 128, 1.0f / sqrtf(128.0f));
  CUDA_KERNEL_CHECK();
}
}
