#include "llm/cuda_check.h"
#include "llm/model_package.h"
#include "llm/qwen3_layer_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
namespace fs = std::filesystem;
struct Entry { std::string file; std::vector<int64_t> shape; };
static void fail(const std::string& s) { throw std::runtime_error("test_qwen3_layer_cuda_fp16: " + s); }
static bool golden_threshold_failed = false;

static fs::path locate(const fs::path& p) {
  if (fs::exists(p)) return p;
  if (fs::exists(fs::path("..") / p)) return fs::path("..") / p;
  fail("artifact missing: " + p.string() + "; run bash tools/run_phase15.sh first");
  return {};
}
static std::map<std::string, Entry> read_manifest(const fs::path& p) {
  std::ifstream in(p); if (!in) fail("manifest missing: " + p.string());
  std::map<std::string, Entry> out; std::string n, f; int rank;
  while (in >> n >> f >> rank) { Entry e{f,{}}; for (int i=0;i<rank;++i) { int64_t d; if (!(in>>d)) fail("malformed manifest"); e.shape.push_back(d); } out[n]=std::move(e); }
  return out;
}
static Tensor fixture(const fs::path& dir, const std::map<std::string, Entry>& m, const std::string& n) {
  auto it=m.find(n); if (it==m.end()) fail("fixture tensor missing: "+n);
  std::ifstream in(dir/it->second.file,std::ios::binary|std::ios::ate); if (!in) fail("fixture file missing: "+n);
  size_t count=1; for (auto d:it->second.shape) count*=size_t(d); if (in.tellg()!=std::streamoff(count*4)) fail("fixture size mismatch: "+n);
  in.seekg(0); std::vector<float> v(count); in.read(reinterpret_cast<char*>(v.data()),std::streamsize(count*4)); return Tensor::from_f32(it->second.shape,v);
}
static Tensor to_f16_cpu(const Tensor& x) {
  Tensor h(DType::F16,x.shape()); for (size_t i=0;i<x.numel();++i) h.set_f32(i,x.get_f32(i)); return h;
}
static Qwen3CudaLayerWeights load_f32(const ModelPackage& p) {
  return {p.load_tensor("layers.0.input_norm"),p.load_tensor("layers.0.q_proj"),p.load_tensor("layers.0.k_proj"),p.load_tensor("layers.0.v_proj"),p.load_tensor("layers.0.q_norm"),p.load_tensor("layers.0.k_norm"),p.load_tensor("layers.0.o_proj"),p.load_tensor("layers.0.post_attention_norm"),p.load_tensor("layers.0.gate_proj"),p.load_tensor("layers.0.up_proj"),p.load_tensor("layers.0.down_proj")};
}
static Qwen3CudaLayerWeights upload_f16(const Qwen3CudaLayerWeights& w) {
  return {to_f16_cpu(w.input_norm).to(DeviceType::CUDA),to_f16_cpu(w.q_proj).to(DeviceType::CUDA),to_f16_cpu(w.k_proj).to(DeviceType::CUDA),to_f16_cpu(w.v_proj).to(DeviceType::CUDA),to_f16_cpu(w.q_norm).to(DeviceType::CUDA),to_f16_cpu(w.k_norm).to(DeviceType::CUDA),to_f16_cpu(w.o_proj).to(DeviceType::CUDA),to_f16_cpu(w.post_attention_norm).to(DeviceType::CUDA),to_f16_cpu(w.gate_proj).to(DeviceType::CUDA),to_f16_cpu(w.up_proj).to(DeviceType::CUDA),to_f16_cpu(w.down_proj).to(DeviceType::CUDA)};
}
static Qwen3CudaLayerWeights upload_f32(const Qwen3CudaLayerWeights& w) {
  return {w.input_norm.to(DeviceType::CUDA),w.q_proj.to(DeviceType::CUDA),w.k_proj.to(DeviceType::CUDA),w.v_proj.to(DeviceType::CUDA),w.q_norm.to(DeviceType::CUDA),w.k_norm.to(DeviceType::CUDA),w.o_proj.to(DeviceType::CUDA),w.post_attention_norm.to(DeviceType::CUDA),w.gate_proj.to(DeviceType::CUDA),w.up_proj.to(DeviceType::CUDA),w.down_proj.to(DeviceType::CUDA)};
}
static double cosine(const Tensor& a,const Tensor& b){double ab=0,aa=0,bb=0;for(size_t i=0;i<a.numel();++i){double x=a.get_f32(i),y=b.get_f32(i);ab+=x*y;aa+=x*x;bb+=y*y;}return aa==0||bb==0?(aa==bb?1:0):ab/std::sqrt(aa*bb);}
static float node_relative_tolerance(const std::string& name) {
  return (name == "k_norm_output" || name == "k_rope") ? 1e-2f : 1e-3f;
}
static void compare(const std::string& n,const Tensor& actual,const Tensor& golden,const Tensor* f32) {
  if(actual.device()!=DeviceType::CUDA||actual.dtype()!=DType::F16||!actual.is_contiguous()) fail(n+" is not contiguous CUDA F16");
  Tensor a=actual.to(DeviceType::CPU); constexpr float abs_tol=5e-3f; const float rel_tol=node_relative_tolerance(n); constexpr double ctol=.999;
  float mx=0; double mean=0, max_ratio=-1; size_t mi=0, ri=0; float ratio_allowed=0;
  for(size_t i=0;i<a.numel();++i){if(!std::isfinite(a.get_f32(i)))fail(n+" produced NaN/Inf");float actual_value=a.get_f32(i), expected_value=golden.get_f32(i), d=std::fabs(actual_value-expected_value);float allowed=std::max(abs_tol,rel_tol*std::fabs(expected_value));double ratio=allowed==0?(d==0?0:std::numeric_limits<double>::infinity()):double(d)/double(allowed);if(d>mx){mx=d;mi=i;}if(ratio>max_ratio){max_ratio=ratio;ri=i;ratio_allowed=allowed;}mean+=d;}
  mean/=a.numel(); double cs=cosine(a,golden); bool over=false; for(size_t i=0;i<a.numel();++i){float d=std::fabs(a.get_f32(i)-golden.get_f32(i)), allowed=std::max(abs_tol,rel_tol*std::fabs(golden.get_f32(i)));if(d>allowed){over=true;break;}}
  std::cout<<n<<" shape=";for(auto d:a.shape())std::cout<<d<<"x";std::cout<<" dtype=F16 device=CUDA max_abs_error="<<mx<<" mean_abs_error="<<mean<<" cosine_similarity="<<cs<<" absolute_tolerance="<<abs_tol<<" relative_tolerance="<<rel_tol;
  if(f32){Tensor f=f32->to(DeviceType::CPU);float md=0;for(size_t i=0;i<a.numel();++i)md=std::max(md,std::fabs(a.get_f32(i)-f.get_f32(i)));std::cout<<" fp16_vs_f32_cuda_max_abs_error="<<md;}
  std::cout<<" max_error_index="<<mi<<" max_ratio_index="<<ri<<" max_error_to_allowed_ratio="<<max_ratio<<"\n";
  if(over||cs<ctol){golden_threshold_failed=true;float actual_value=a.get_f32(ri),expected_value=golden.get_f32(ri),abs_error=std::fabs(actual_value-expected_value);std::string relative_error=expected_value==0?"N/A":std::to_string(abs_error/std::fabs(expected_value));std::cerr<<n<<" threshold failed node="<<n<<" index="<<ri<<" actual="<<actual_value<<" expected="<<expected_value<<" abs_error="<<abs_error<<" relative_error="<<relative_error<<" allowed="<<ratio_allowed<<" error_to_allowed_ratio="<<max_ratio<<" cosine_similarity="<<cs<<" max_abs_index="<<mi<<" max_abs_actual="<<a.get_f32(mi)<<" max_abs_expected="<<golden.get_f32(mi)<<" max_abs_error="<<mx<<"\n";}
}
static void expect_throw(const std::string& n,const std::function<void()>& fn){try{fn();}catch(const std::exception& e){std::cout<<"rejected "<<n<<": "<<e.what()<<"\n";return;}fail(n+" was accepted");}

int main(){try{
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop,0)); std::cout<<"CUDA device="<<prop.name<<" compute_capability="<<prop.major<<"."<<prop.minor<<"\n";
  fs::path pkg=locate("artifacts/phase15/qwen3-0.6b-f32"), dir=locate("artifacts/phase1/qwen3_layer0"); ModelPackage package(pkg); float eps=std::stof(package.config("rms_norm_eps")),theta=std::stof(package.config("rope_theta")); std::cout<<"rms_norm_eps="<<eps<<" rope_theta="<<theta<<"\n";
  auto m=read_manifest(dir/"manifest.txt"); Tensor hidden=fixture(dir,m,"hidden_states"), pos_tensor=fixture(dir,m,"position_ids"); std::vector<int32_t> pos(pos_tensor.numel());for(size_t i=0;i<pos.size();++i)pos[i]=int32_t(pos_tensor.get_f32(i));
  auto wf_cpu=load_f32(package), wf=upload_f32(wf_cpu), wh=upload_f16(wf_cpu); Tensor h16=to_f16_cpu(hidden), hc=hidden.to(DeviceType::CUDA), h16c=h16.to(DeviceType::CUDA);
  auto f32=qwen3_decoder_layer_cuda_trace(hc,pos,wf,eps,theta); auto f16=qwen3_decoder_layer_cuda_trace_fp16(h16c,pos,wh,eps,theta);
  std::map<std::string,const Tensor*> a={{"input_norm",&f16.input_norm},{"q_linear",&f16.q_linear},{"k_linear",&f16.k_linear},{"v_linear",&f16.v_linear},{"q_norm_output",&f16.q_norm_output},{"k_norm_output",&f16.k_norm_output},{"q_rope",&f16.q_rope},{"k_rope",&f16.k_rope},{"attention_output",&f16.attention_output},{"o_proj_output",&f16.o_proj_output},{"attention_residual",&f16.attention_residual},{"post_attention_norm",&f16.post_attention_norm},{"gate_proj_output",&f16.gate_proj_output},{"up_proj_output",&f16.up_proj_output},{"swiglu_output",&f16.swiglu_output},{"down_proj_output",&f16.down_proj_output},{"layer_output",&f16.layer_output}};
  std::map<std::string,const Tensor*> b={{"input_norm",&f32.input_norm},{"q_linear",&f32.q_linear},{"k_linear",&f32.k_linear},{"v_linear",&f32.v_linear},{"q_norm_output",&f32.q_norm_output},{"k_norm_output",&f32.k_norm_output},{"q_rope",&f32.q_rope},{"k_rope",&f32.k_rope},{"attention_output",&f32.attention_output},{"o_proj_output",&f32.o_proj_output},{"attention_residual",&f32.attention_residual},{"post_attention_norm",&f32.post_attention_norm},{"gate_proj_output",&f32.gate_proj_output},{"up_proj_output",&f32.up_proj_output},{"swiglu_output",&f32.swiglu_output},{"down_proj_output",&f32.down_proj_output},{"layer_output",&f32.layer_output}};
  for(const auto& kv:a)compare(kv.first,*kv.second,fixture(dir,m,kv.first),b.at(kv.first));
  expect_throw("CPU hidden_states",[&]{qwen3_decoder_layer_cuda_trace_fp16(hidden,pos,wh,eps,theta);}); expect_throw("F32 CUDA hidden_states",[&]{qwen3_decoder_layer_cuda_trace_fp16(hc,pos,wh,eps,theta);}); auto mixed=wh;mixed.q_proj=wf.q_proj;expect_throw("mixed F32 weight",[&]{qwen3_decoder_layer_cuda_trace_fp16(h16c,pos,mixed,eps,theta);}); expect_throw("invalid hidden shape",[&]{qwen3_decoder_layer_cuda_trace_fp16(Tensor(DType::F16,{4,1023},DeviceType::CUDA),pos,wh,eps,theta);}); expect_throw("position length",[&]{qwen3_decoder_layer_cuda_trace_fp16(h16c,{0,1},wh,eps,theta);}); expect_throw("unsupported seq_len",[&]{qwen3_decoder_layer_cuda_trace_fp16(Tensor(DType::F16,{33,1024},DeviceType::CUDA),std::vector<int32_t>(33),wh,eps,theta);}); auto bad=wh;bad.q_proj=Tensor(DType::F16,{2047,1024},DeviceType::CUDA);expect_throw("weight shape",[&]{qwen3_decoder_layer_cuda_trace_fp16(h16c,pos,bad,eps,theta);});expect_throw("eps",[&]{qwen3_decoder_layer_cuda_trace_fp16(h16c,pos,wh,0,theta);});expect_throw("theta",[&]{qwen3_decoder_layer_cuda_trace_fp16(h16c,pos,wh,eps,0);});expect_throw("F32 API rejects F16",[&]{qwen3_decoder_layer_cuda_trace(h16c,pos,wh,eps,theta);});
  CUDA_CHECK(cudaDeviceSynchronize()); if(golden_threshold_failed) fail("one or more FP16 golden node thresholds failed"); std::cout<<"test_qwen3_layer_cuda_fp16 passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
