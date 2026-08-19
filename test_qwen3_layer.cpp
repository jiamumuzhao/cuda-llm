#include "llm/qwen3_layer.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace llm;
namespace fs = std::filesystem;
struct Entry { std::string file; std::vector<int64_t> shape; };
static void fail(const std::string& s){throw std::runtime_error("test_qwen3_layer: "+s);}
static std::map<std::string,Entry> manifest(const fs::path& p){std::ifstream in(p);if(!in)fail("manifest missing: "+p.string());std::map<std::string,Entry> r;std::string n,f;int rank;while(in>>n>>f>>rank){Entry e{f,{}};for(int i=0;i<rank;++i){int64_t d;if(!(in>>d))fail("malformed manifest entry "+n);e.shape.push_back(d);}r.emplace(n,e);}return r;}
static Tensor load(const fs::path& dir,const std::map<std::string,Entry>& m,const std::string& n){auto it=m.find(n);if(it==m.end())fail("manifest missing tensor "+n);auto p=dir/it->second.file;std::ifstream in(p,std::ios::binary|std::ios::ate);if(!in)fail("file missing for "+n+": "+p.string());auto bytes=in.tellg();size_t num=1;for(auto d:it->second.shape)num*=size_t(d);if(bytes!=std::streamoff(num*sizeof(float)))fail(n+" byte size mismatch");in.seekg(0);std::vector<float> v(num);in.read(reinterpret_cast<char*>(v.data()),bytes);return Tensor::from_f32(it->second.shape,v);}
static Tensor zeros(std::vector<int64_t> s){size_t n=1;for(auto d:s)n*=size_t(d);return Tensor::from_f32(s,std::vector<float>(n,0));}
static Qwen3LayerWeights weights(){return {zeros({1024}),zeros({2048,1024}),zeros({1024,1024}),zeros({1024,1024}),zeros({128}),zeros({128}),zeros({1024,2048}),zeros({1024}),zeros({3072,1024}),zeros({3072,1024}),zeros({1024,3072})};}
static void invalid_input_tests(){auto w=weights();auto h=zeros({1,1024});bool ok=false;try{qwen3_decoder_layer(zeros({1,1023}),{0},w,1e-6f,10000);}catch(const std::exception&e){ok=std::string(e.what()).find("hidden_states")!=std::string::npos;}if(!ok)fail("invalid hidden shape was not rejected with context");ok=false;try{qwen3_decoder_layer(h,{},w,1e-6f,10000);}catch(const std::exception&e){ok=true;}if(!ok)fail("invalid position_ids length was not rejected");auto bad=w;bad.q_norm=zeros({127});ok=false;try{qwen3_decoder_layer(h,{0},bad,1e-6f,10000);}catch(const std::exception&e){ok=true;}if(!ok)fail("invalid q_norm shape was not rejected");}
static std::map<std::string,float> config(const fs::path& p){std::ifstream in(p);if(!in)fail("config missing: "+p.string());std::map<std::string,float> r;std::string k,v;while(in>>k>>v){try{r[k]=std::stof(v);}catch(...){}}return r;}
static double cosine(const Tensor&a,const Tensor&b){double ab=0,aa=0,bb=0;for(size_t i=0;i<a.numel();++i){double x=a.get_f32(i),y=b.get_f32(i);ab+=x*y;aa+=x*x;bb+=y*y;}return aa==0||bb==0?(aa==bb?1:0):ab/std::sqrt(aa*bb);}
static void compare(const std::string& n,const Tensor&a,const Tensor&b){if(a.shape()!=b.shape())fail(n+" shape mismatch");float mx=0;double mean=0;for(size_t i=0;i<a.numel();++i){float d=std::fabs(a.get_f32(i)-b.get_f32(i));mx=std::max(mx,d);mean+=d;}mean/=a.numel();double cs=cosine(a,b);std::cout<<n<<" shape=";for(auto d:a.shape())std::cout<<d<<"x";std::cout<<" max_abs_error="<<mx<<" mean_abs_error="<<mean<<" cosine_similarity="<<cs<<"\n";if(mx>5e-4f||cs<.99999)fail(n+" threshold failed shape="+std::to_string(a.numel())+" max_error="+std::to_string(mx)+" cosine_similarity="+std::to_string(cs));}
int main(){
 try { invalid_input_tests(); } catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
 fs::path dir="artifacts/phase1/qwen3_layer0";if(!fs::exists(dir/"manifest.txt"))fail("fixture manifest missing: "+(dir/"manifest.txt").string());
 try {auto m=manifest(dir/"manifest.txt");auto cfg=config(dir/"config.txt");if(cfg.find("rms_norm_eps")==cfg.end()||cfg.find("rope_theta")==cfg.end())fail("config must contain rms_norm_eps and rope_theta");auto h=load(dir,m,"hidden_states");auto pos=load(dir,m,"position_ids");std::vector<int32_t> p(pos.numel());for(size_t i=0;i<p.size();++i)p[i]=int32_t(pos.get_f32(i));Qwen3LayerWeights w{load(dir,m,"input_layernorm_weight"),load(dir,m,"q_proj_weight"),load(dir,m,"k_proj_weight"),load(dir,m,"v_proj_weight"),load(dir,m,"q_norm_weight"),load(dir,m,"k_norm_weight"),load(dir,m,"o_proj_weight"),load(dir,m,"post_attention_layernorm_weight"),load(dir,m,"gate_proj_weight"),load(dir,m,"up_proj_weight"),load(dir,m,"down_proj_weight")};if(std::fabs(cfg["rms_norm_eps"]-1e-6f)>1e-12f||std::fabs(cfg["rope_theta"]-1000000.f)>1.f)fail("unexpected Qwen3 config eps/theta");auto r=qwen3_decoder_layer_trace(h,p,w,cfg["rms_norm_eps"],cfg["rope_theta"]);std::map<std::string,const Tensor*> got={{"input_norm",&r.input_norm},{"q_linear",&r.q_linear},{"k_linear",&r.k_linear},{"v_linear",&r.v_linear},{"q_norm_output",&r.q_normed},{"k_norm_output",&r.k_normed},{"q_rope",&r.q_rope},{"k_rope",&r.k_rope},{"attention_output",&r.attention_output},{"o_proj_output",&r.o_proj_output},{"attention_residual",&r.attention_residual},{"post_attention_norm",&r.post_attention_norm},{"gate_proj_output",&r.gate_proj_output},{"up_proj_output",&r.up_proj_output},{"swiglu_output",&r.swiglu_output},{"down_proj_output",&r.down_proj_output},{"layer_output",&r.layer_output}};for(auto&kv:got)compare(kv.first,*kv.second,load(dir,m,kv.first));return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
