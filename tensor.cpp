#include "llm/tensor.h"
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace llm {
static uint32_t bits(float x){uint32_t u; std::memcpy(&u,&x,4);return u;} static float fp(uint32_t u){float x;std::memcpy(&x,&u,4);return x;}
uint16_t encode_bf16(float x){return uint16_t(bits(x)>>16);} float decode_bf16(uint16_t x){return fp(uint32_t(x)<<16);}
uint16_t encode_f16(float x){uint32_t u=bits(x); uint32_t s=(u>>16)&0x8000,e=(u>>23)&255,m=u&0x7fffff; if(e==255)return uint16_t(s|0x7c00|(m?0x200:0)); int ne=int(e)-127+15; if(ne>=31)return uint16_t(s|0x7c00); if(ne<=0){if(ne<-10)return uint16_t(s); m|=0x800000; return uint16_t(s|((m>>(14-ne)) + ((m>>(13-ne))&1)));} return uint16_t(s|(ne<<10)|(m>>13));}
float decode_f16(uint16_t h){
  const uint32_t sign=uint32_t(h&0x8000)<<16; const uint32_t exp=(h>>10)&0x1f; uint32_t mant=h&0x3ff;
  if(exp==0){
    if(mant==0) return fp(sign); // signed zero
    int unbiased=-14; // normalize the half subnormal significand
    while((mant&0x400)==0){mant<<=1; --unbiased;}
    mant&=0x3ff;
    return fp(sign | uint32_t(unbiased+127)<<23 | mant<<13);
  }
  if(exp==31) return fp(sign | 0x7f800000u | (mant<<13));
  return fp(sign | ((exp+112)<<23) | (mant<<13));
}
static size_t checked_numel(const std::vector<int64_t>& sh){size_t n=1; for(auto x:sh){if(x<0)throw std::invalid_argument("Tensor shape contains negative dimension"); if(x && n>std::numeric_limits<size_t>::max()/size_t(x))throw std::invalid_argument("Tensor shape product overflow"); n*=size_t(x);} return n;}
static std::vector<int64_t> make_strides(const std::vector<int64_t>& sh){std::vector<int64_t>s(sh.size());int64_t x=1;for(int i=int(sh.size())-1;i>=0;--i){s[i]=x; if(sh[i] && x>std::numeric_limits<int64_t>::max()/sh[i])throw std::invalid_argument("stride overflow");x*=sh[i];}return s;}
Tensor::Tensor(DType d,std::vector<int64_t> sh,DeviceType dev):dtype_(d),device_(dev),shape_(std::move(sh)),strides_(make_strides(shape_)),numel_(checked_numel(shape_)){if(dev!=DeviceType::CPU)throw std::runtime_error("Tensor: only CPU device is supported");storage_=std::make_shared<std::vector<uint8_t>>(nbytes());}
Tensor::Tensor(DType d,DeviceType dev,std::vector<int64_t> sh,std::vector<int64_t> st,std::shared_ptr<std::vector<uint8_t>>s,size_t o,size_t n):dtype_(d),device_(dev),shape_(std::move(sh)),strides_(std::move(st)),storage_(std::move(s)),offset_(o),numel_(n){if(dev!=DeviceType::CPU)throw std::runtime_error("Tensor view: only CPU device is supported");if(offset_>storage_bytes()||nbytes()>storage_bytes()-offset_)throw std::runtime_error("Tensor view exceeds storage");}
Tensor Tensor::from_f32(const std::vector<int64_t>& sh,const std::vector<float>&v){Tensor t(DType::F32,sh);if(v.size()!=t.numel())throw std::invalid_argument("from_f32 value count does not match shape");std::memcpy(t.data(),v.data(),v.size()*4);return t;}
bool Tensor::is_contiguous()const{return strides_==make_strides(shape_);}
Tensor Tensor::reshape(const std::vector<int64_t>& sh)const{if(!is_contiguous())throw std::runtime_error("reshape requires contiguous tensor");auto n=checked_numel(sh);if(n!=numel_)throw std::invalid_argument("reshape changes element count");return Tensor(dtype_,device_,sh,make_strides(sh),storage_,offset_,n);}
Tensor Tensor::contiguous()const{if(is_contiguous())return *this;Tensor t(dtype_,shape_,device_);for(size_t i=0;i<numel_;++i)t.set_f32(i,get_f32(i));return t;}
void* Tensor::data(){return storage_?storage_->data()+offset_:nullptr;}const void* Tensor::data()const{return storage_?storage_->data()+offset_:nullptr;}
float Tensor::get_f32(size_t i)const{if(i>=numel_)throw std::out_of_range("Tensor element index");if(dtype_==DType::F32)return static_cast<const float*>(data())[i];if(dtype_==DType::F16)return decode_f16(static_cast<const uint16_t*>(data())[i]);return decode_bf16(static_cast<const uint16_t*>(data())[i]);}
void Tensor::set_f32(size_t i,float v){if(i>=numel_)throw std::out_of_range("Tensor element index");if(dtype_==DType::F32)static_cast<float*>(data())[i]=v;else if(dtype_==DType::F16)static_cast<uint16_t*>(data())[i]=encode_f16(v);else static_cast<uint16_t*>(data())[i]=encode_bf16(v);}
float* Tensor::data_f32(){if(dtype_!=DType::F32)throw std::runtime_error("Tensor dtype is not F32");return static_cast<float*>(data());}const float* Tensor::data_f32()const{if(dtype_!=DType::F32)throw std::runtime_error("Tensor dtype is not F32");return static_cast<const float*>(data());}uint16_t* Tensor::data_f16(){if(dtype_!=DType::F16)throw std::runtime_error("Tensor dtype is not F16");return static_cast<uint16_t*>(data());}uint16_t* Tensor::data_bf16(){if(dtype_!=DType::BF16)throw std::runtime_error("Tensor dtype is not BF16");return static_cast<uint16_t*>(data());}
}
