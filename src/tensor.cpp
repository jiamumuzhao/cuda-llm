#include "llm/tensor.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
namespace llm {
namespace detail { void* cuda_allocate(size_t); void cuda_release(void*); }
static uint32_t bits(float x){uint32_t u;std::memcpy(&u,&x,4);return u;} static float fp(uint32_t u){float x;std::memcpy(&x,&u,4);return x;}
uint16_t encode_bf16(float x){return uint16_t(bits(x)>>16);} float decode_bf16(uint16_t x){return fp(uint32_t(x)<<16);}
uint16_t encode_f16(float x){uint32_t u=bits(x),s=(u>>16)&0x8000,e=(u>>23)&255,m=u&0x7fffff;if(e==255)return uint16_t(s|0x7c00|(m?0x200:0));int ne=int(e)-127+15;if(ne>=31)return uint16_t(s|0x7c00);if(ne<=0){if(ne<-10)return uint16_t(s);m|=0x800000;return uint16_t(s|((m>>(14-ne))+((m>>(13-ne))&1)));}return uint16_t(s|(ne<<10)|(m>>13));}
float decode_f16(uint16_t h){uint32_t sign=uint32_t(h&0x8000)<<16,exp=(h>>10)&31,mant=h&1023;if(exp==0){if(!mant)return fp(sign);int e=-14;while(!(mant&0x400)){mant<<=1;--e;}mant&=1023;return fp(sign|uint32_t(e+127)<<23|mant<<13);}if(exp==31)return fp(sign|0x7f800000u|mant<<13);return fp(sign|uint32_t(exp+112)<<23|mant<<13);}
static size_t checked(const std::vector<int64_t>&sh){size_t n=1;for(auto d:sh){if(d<0)throw std::invalid_argument("Tensor: negative shape dimension");if(d&&n>std::numeric_limits<size_t>::max()/size_t(d))throw std::invalid_argument("Tensor: shape product overflow");n*=size_t(d);}return n;}
static std::vector<int64_t> row_strides(const std::vector<int64_t>&sh){std::vector<int64_t>s(sh.size());int64_t n=1;for(int i=int(sh.size())-1;i>=0;--i){s[i]=n;if(sh[i]&&n>std::numeric_limits<int64_t>::max()/sh[i])throw std::invalid_argument("Tensor: stride overflow");n*=sh[i];}return s;}
Tensor::Tensor(DType d,std::vector<int64_t>sh,DeviceType dev):dtype_(d),device_(dev),shape_(std::move(sh)),strides_(row_strides(shape_)),numel_(checked(shape_)){if(dev==DeviceType::CUDA){if(d==DType::BF16)throw std::runtime_error("Tensor CUDA: BF16 is unsupported on this path");void*p=detail::cuda_allocate(nbytes());storage_=std::shared_ptr<void>(p,detail::cuda_release);storage_size_=nbytes();}else{auto*p=new std::vector<uint8_t>(nbytes());storage_=std::shared_ptr<void>(p,[](void*x){delete static_cast<std::vector<uint8_t>*>(x);});storage_size_=nbytes();}}
Tensor::Tensor(DType d,DeviceType dev,std::vector<int64_t>sh,std::vector<int64_t>st,std::shared_ptr<void>s,size_t bytes,size_t off,size_t n):dtype_(d),device_(dev),shape_(std::move(sh)),strides_(std::move(st)),storage_(std::move(s)),storage_size_(bytes),offset_(off),numel_(n){if(dev==DeviceType::CUDA&&d==DType::BF16)throw std::runtime_error("Tensor view CUDA: BF16 unsupported");if(offset_>storage_size_||nbytes()>storage_size_-offset_)throw std::runtime_error("Tensor view exceeds storage");}
Tensor Tensor::from_f32(const std::vector<int64_t>&sh,const std::vector<float>&v){Tensor t(DType::F32,sh);if(v.size()!=t.numel())throw std::invalid_argument("Tensor::from_f32 value count mismatch");std::memcpy(t.data(),v.data(),v.size()*4);return t;}
bool Tensor::is_contiguous()const{return strides_==row_strides(shape_);}
Tensor Tensor::reshape(const std::vector<int64_t>&sh)const{if(!is_contiguous())throw std::runtime_error("Tensor::reshape requires contiguous tensor");auto n=checked(sh);if(n!=numel_)throw std::invalid_argument("Tensor::reshape changes element count");return Tensor(dtype_,device_,sh,row_strides(sh),storage_,storage_size_,offset_,n);}
Tensor Tensor::prefix_first_dim(size_t size) const {
  if (shape_.empty() || shape_[0] <= 0 || size == 0 || size > static_cast<size_t>(shape_[0]))
    throw std::invalid_argument("Tensor::prefix_first_dim: invalid prefix size");
  if (!is_contiguous()) throw std::runtime_error("Tensor::prefix_first_dim requires contiguous tensor");
  auto shape = shape_;
  shape[0] = static_cast<int64_t>(size);
  const size_t row_numel = numel_ / static_cast<size_t>(shape_[0]);
  return Tensor(dtype_, device_, shape, row_strides(shape), storage_,
                storage_size_, offset_, size * row_numel);
}
Tensor Tensor::slice_first_dim(size_t offset, size_t size) const {
  if (shape_.empty() || shape_[0] <= 0 || size == 0 ||
      offset > static_cast<size_t>(shape_[0]) ||
      size > static_cast<size_t>(shape_[0]) - offset)
    throw std::invalid_argument("Tensor::slice_first_dim: invalid slice");
  if (!is_contiguous())
    throw std::runtime_error("Tensor::slice_first_dim requires contiguous tensor");
  auto shape = shape_;
  shape[0] = static_cast<int64_t>(size);
  const size_t row_numel = numel_ / static_cast<size_t>(shape_[0]);
  return Tensor(dtype_, device_, shape, row_strides(shape), storage_,
                storage_size_, offset_ + offset * row_numel * dtype_size(dtype_),
                size * row_numel);
}
Tensor Tensor::contiguous()const{if(is_contiguous())return *this;if(device_==DeviceType::CUDA)throw std::runtime_error("Tensor::contiguous for non-contiguous CUDA storage is unsupported");Tensor t(dtype_,shape_);for(size_t i=0;i<numel_;++i)t.set_f32(i,get_f32(i));return t;}
void* Tensor::data(){if(!storage_)return nullptr;if(device_==DeviceType::CPU)return static_cast<char*>(static_cast<void*>(static_cast<std::vector<uint8_t>*>(storage_.get())->data()))+offset_;return static_cast<char*>(storage_.get())+offset_;}const void* Tensor::data()const{if(!storage_)return nullptr;if(device_==DeviceType::CPU)return static_cast<const char*>(static_cast<const void*>(static_cast<const std::vector<uint8_t>*>(storage_.get())->data()))+offset_;return static_cast<const char*>(storage_.get())+offset_;}
static void host_only(DeviceType d,const char*n){if(d==DeviceType::CUDA)throw std::runtime_error(std::string("Tensor::")+n+": host access to CUDA storage is forbidden; use to(CPU)");}
float Tensor::get_f32(size_t i)const{host_only(device_,"get_f32");if(dtype_==DType::I32)throw std::runtime_error("Tensor::get_f32 is unavailable for I32");if(i>=numel_)throw std::out_of_range("Tensor element index");const uint8_t*p=static_cast<const uint8_t*>(data());if(dtype_==DType::F32)return static_cast<const float*>(static_cast<const void*>(p))[i];if(dtype_==DType::F16)return decode_f16(static_cast<const uint16_t*>(static_cast<const void*>(p))[i]);return decode_bf16(static_cast<const uint16_t*>(static_cast<const void*>(p))[i]);}
void Tensor::set_f32(size_t i,float v){host_only(device_,"set_f32");if(dtype_==DType::I32)throw std::runtime_error("Tensor::set_f32 is unavailable for I32");if(i>=numel_)throw std::out_of_range("Tensor element index");uint8_t*p=static_cast<uint8_t*>(data());if(dtype_==DType::F32)static_cast<float*>(static_cast<void*>(p))[i]=v;else if(dtype_==DType::F16)static_cast<uint16_t*>(static_cast<void*>(p))[i]=encode_f16(v);else static_cast<uint16_t*>(static_cast<void*>(p))[i]=encode_bf16(v);}
float* Tensor::data_f32(){host_only(device_,"data_f32");if(dtype_!=DType::F32)throw std::runtime_error("Tensor dtype is not F32");return static_cast<float*>(data());}const float* Tensor::data_f32()const{host_only(device_,"data_f32");if(dtype_!=DType::F32)throw std::runtime_error("Tensor dtype is not F32");return static_cast<const float*>(data());}uint16_t* Tensor::data_f16(){host_only(device_,"data_f16");if(dtype_!=DType::F16)throw std::runtime_error("Tensor dtype is not F16");return static_cast<uint16_t*>(data());}uint16_t* Tensor::data_bf16(){host_only(device_,"data_bf16");if(dtype_!=DType::BF16)throw std::runtime_error("Tensor dtype is not BF16");return static_cast<uint16_t*>(data());}
int32_t* Tensor::data_i32(){host_only(device_,"data_i32");if(dtype_!=DType::I32)throw std::runtime_error("Tensor dtype is not I32");return static_cast<int32_t*>(data());}const int32_t* Tensor::data_i32()const{host_only(device_,"data_i32");if(dtype_!=DType::I32)throw std::runtime_error("Tensor dtype is not I32");return static_cast<const int32_t*>(data());}
}
