#include "llm/tensor.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
using namespace llm;
static void check(bool ok,const std::string& item){if(!ok)throw std::runtime_error("test_tensor "+item);}
static void nearv(const std::string& item,float actual,float expected,float tol){if(!(std::isfinite(actual)&&std::fabs(actual-expected)<=tol))throw std::runtime_error("test_tensor "+item+" actual="+std::to_string(actual)+" expected="+std::to_string(expected)+" tolerance="+std::to_string(tol));}
int main(){
 nearv("half min subnormal",decode_f16(0x0001),std::ldexp(1.f,-24),1e-10f);
 nearv("half max subnormal",decode_f16(0x03ff),std::ldexp(1.f,-14)-std::ldexp(1.f,-24),1e-10f);
 nearv("half min normal",decode_f16(0x0400),std::ldexp(1.f,-14),1e-10f);
 float negzero=decode_f16(0x8000); check(negzero==0 && std::signbit(negzero),"negative zero actual="+std::to_string(negzero)+" expected=-0");
 check(std::isinf(decode_f16(0x7c00))&&!std::signbit(decode_f16(0x7c00)),"positive infinity actual="+std::to_string(decode_f16(0x7c00))+" expected=+inf");
 check(std::isnan(decode_f16(0x7e00)),"NaN actual="+std::to_string(decode_f16(0x7e00))+" expected=NaN");
 auto t=Tensor::from_f32({2,3},{1,2,3,4,5,6}); check(t.numel()==6&&t.nbytes()==24,"numel/nbytes actual="+std::to_string(t.numel())+" expected=6/24"); check(t.is_contiguous()&&t.strides()==std::vector<int64_t>({3,1}),"row-major strides"); auto r=t.reshape({3,2});r.set_f32(0,9);nearv("reshape shared storage",t.get_f32(0),9,0);
 for(auto d:{DType::F16,DType::BF16}){Tensor q(d,{2});q.set_f32(0,1.25f);nearv("dtype conversion",q.get_f32(0),1.25f,.02f);}
 bool e=false;try{Tensor bad(DType::F32,{-1});}catch(...){e=true;}check(e,"negative shape exception");e=false;try{t.reshape({4,2});}catch(...){e=true;}check(e,"reshape exception");
 std::cout<<"test_tensor passed\n";
}
