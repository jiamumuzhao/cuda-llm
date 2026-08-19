#include "llm/model_package.h"
#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace llm {
uint64_t fnv1a64(const uint8_t* p,size_t n){uint64_t h=1469598103934665603ull;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h;}
static void bad(const std::string&s){throw std::runtime_error("ModelPackage: "+s);}
static const std::string& required_config(const std::map<std::string,std::string>& c,const std::string& k){auto i=c.find(k);if(i==c.end())bad("config key "+k+" missing, actual=<missing>, expected=present");return i->second;}
static long long config_int(const std::map<std::string,std::string>&c,const std::string&k,long long expected){const auto&v=required_config(c,k);try{size_t p=0;long long x=std::stoll(v,&p);if(p!=v.size()||x!=expected)bad("config key "+k+" actual="+v+" expected="+std::to_string(expected));return x;}catch(const std::exception&){bad("config key "+k+" actual="+v+" expected="+std::to_string(expected));}return 0;}
static double config_float(const std::map<std::string,std::string>&c,const std::string&k,double expected){const auto&v=required_config(c,k);try{size_t p=0;double x=std::stod(v,&p);if(p!=v.size()||!std::isfinite(x)||x<=0||x!=expected)bad("config key "+k+" actual="+v+" expected="+std::to_string(expected));return x;}catch(const std::exception&){bad("config key "+k+" actual="+v+" expected="+std::to_string(expected));}return 0;}
ModelPackage::ModelPackage(std::filesystem::path r):root_(std::move(r)){if(!std::filesystem::is_directory(root_))bad("package directory missing: "+root_.string());parse_meta();}
void ModelPackage::parse_meta(){
 std::ifstream in(root_/"model.meta");if(!in)bad("model.meta missing");std::string line;
 while(std::getline(in,line)){auto p=line.find('#');if(p!=std::string::npos)line.resize(p);std::istringstream s(line);std::string kind;if(!(s>>kind))continue;
  const bool config_record=kind=="format"||kind=="model_type"||kind=="source_model"||kind=="source_dtype"||kind=="export_dtype"||kind=="hidden_size"||kind=="num_hidden_layers"||kind=="num_attention_heads"||kind=="num_key_value_heads"||kind=="head_dim"||kind=="intermediate_size"||kind=="vocab_size"||kind=="rms_norm_eps"||kind=="rope_theta"||kind=="tie_word_embeddings";
  if(config_record){std::string v;if(!(s>>v)||config_.count(kind))bad("invalid or duplicate config key "+kind);config_[kind]=v;}
  else if(kind=="tensor"){PackageTensorInfo t;int rank;if(!(s>>t.name>>t.dtype>>rank)||rank<0||rank>8||tensors_.count(t.name))bad("invalid or duplicate tensor line");if(t.dtype!="f32")bad(t.name+" has unsupported dtype "+t.dtype);for(int i=0;i<rank;++i){int64_t d;if(!(s>>d)||d<0)bad("invalid shape for "+t.name);t.shape.push_back(d);}std::string off,sz,hash;if(!(s>>off>>sz>>hash))bad("incomplete tensor line "+t.name);try{t.offset=std::stoull(off);t.byte_size=std::stoull(sz);t.checksum=std::stoull(hash,nullptr,16);}catch(...){bad("invalid tensor metadata "+t.name);}if(t.offset%64)bad(t.name+" offset is not 64-byte aligned");size_t num=1;for(auto d:t.shape){if(d&&num>std::numeric_limits<size_t>::max()/size_t(d))bad("shape overflow "+t.name);num*=size_t(d);}if(t.byte_size!=num*4)bad("byte size mismatch "+t.name);tensors_.emplace(t.name,std::move(t));}
  else if(kind=="alias"){std::string n,t;if(!(s>>n>>t)||aliases_.count(n)||tensors_.count(n))bad("invalid or duplicate alias "+n);aliases_[n]=t;}
  else bad("unknown metadata record "+kind);
 }
 auto f=config_.find("format"),m=config_.find("model_type");if(f==config_.end()||m==config_.end()||f->second!="cuda_llm_model_package_v1"||m->second!="qwen3")bad("unsupported package format/model type");
 for(auto&a:aliases_)if(!tensors_.count(a.second)&&!aliases_.count(a.second))bad("alias target missing: "+a.first);
 for(auto&a:aliases_){std::string n=a.first;for(size_t i=0;i<=aliases_.size();++i){auto it=aliases_.find(n);if(it==aliases_.end())break;n=it->second;if(n==a.first)bad("alias cycle at "+a.first);}}
 required_config(config_,"export_dtype");if(config_.at("export_dtype")!="f32")bad("config key export_dtype actual="+config_.at("export_dtype")+" expected=f32");
 config_int(config_,"hidden_size",1024);config_int(config_,"num_hidden_layers",28);config_int(config_,"num_attention_heads",16);config_int(config_,"num_key_value_heads",8);config_int(config_,"head_dim",128);config_int(config_,"intermediate_size",3072);config_int(config_,"vocab_size",151936);config_float(config_,"rms_norm_eps",1e-6);config_float(config_,"rope_theta",1000000);
 std::string tied=required_config(config_,"tie_word_embeddings");if(tied!="true")bad("config key tie_word_embeddings actual="+tied+" expected=true");
}
std::string ModelPackage::config(const std::string&k)const{auto i=config_.find(k);if(i==config_.end())bad("config key missing: "+k);return i->second;}
std::vector<PackageTensorInfo> ModelPackage::tensors()const{std::vector<PackageTensorInfo>v;for(auto&x:tensors_)v.push_back(x.second);return v;}
std::vector<PackageAlias> ModelPackage::aliases()const{std::vector<PackageAlias>v;for(auto&x:aliases_)v.push_back({x.first,x.second});return v;}
std::string ModelPackage::resolve_name(const std::string&n)const{std::string x=n;for(size_t i=0;i<=aliases_.size();++i){auto a=aliases_.find(x);if(a==aliases_.end())return x;x=a->second;}bad("alias cycle resolving "+n);return {};}
Tensor ModelPackage::load_tensor(const std::string&n)const{auto logical=resolve_name(n);auto i=tensors_.find(logical);if(i==tensors_.end())bad("tensor not found: "+n);auto&t=i->second;std::ifstream in(root_/"weights.bin",std::ios::binary|std::ios::ate);if(!in)bad("weights.bin missing");uint64_t total=uint64_t(in.tellg());if(t.offset>total||t.byte_size>total-t.offset)bad("tensor outside weights.bin: "+logical);in.seekg(std::streamoff(t.offset));std::vector<uint8_t>b(size_t(t.byte_size));in.read(reinterpret_cast<char*>(b.data()),std::streamsize(b.size()));if(!in)bad("short read: "+logical);if(fnv1a64(b.data(),b.size())!=t.checksum)bad("checksum mismatch: "+logical);std::vector<float>v(b.size()/4);std::memcpy(v.data(),b.data(),b.size());return Tensor::from_f32(t.shape,v);}
}
