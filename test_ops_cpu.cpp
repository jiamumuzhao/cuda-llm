#include "llm/ops.h"
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
using namespace llm;
static void check(bool ok,const std::string& item){if(!ok)throw std::runtime_error("test_ops_cpu "+item);}
static void nearv(const std::string& item,float actual,float expected,float tol=1e-4f){if(!(std::isfinite(actual)&&std::isfinite(expected)&&std::fabs(actual-expected)<=tol))throw std::runtime_error("test_ops_cpu "+item+" actual="+std::to_string(actual)+" expected="+std::to_string(expected)+" tolerance="+std::to_string(tol));}
static void throws(const std::string& item,const std::function<void()>& fn){try{fn();}catch(const std::exception&){return;}throw std::runtime_error("test_ops_cpu "+item+" expected exception");}
int main(){
 auto w=Tensor::from_f32({3,2},{1,2,3,4,5,6}); auto e=embedding_lookup(w,{2,0}); nearv("embedding shape/value",e.get_f32(0),5);
 throws("embedding_lookup token bounds",[&]{embedding_lookup(w,{3});}); auto x=Tensor::from_f32({1,2},{2,3}); auto l=linear(x,w); check(l.shape()==std::vector<int64_t>({1,3}),"linear shape actual=[1,3] expected=[1,3]"); nearv("linear [1,2]x[3,2]",l.get_f32(0),8); throws("linear feature mismatch",[&]{linear(x,Tensor::from_f32({3,3},{1,2,3,4,5,6,7,8,9}));});
 auto rw=Tensor::from_f32({2},{1,1}); auto n=rms_norm(x,rw,0); nearv("rms_norm shape=[1,2]",n.get_f32(0),2/std::sqrt(6.5f)); throws("rms_norm hidden mismatch",[&]{rms_norm(x,Tensor::from_f32({3},{1,1,1}),0);});
 auto r0=rope(Tensor::from_f32({1,1,4},{1,2,3,4}),{0},10000); for(size_t i=0;i<4;++i)nearv("rope position zero",r0.get_f32(i),float(i+1)); auto r1=rope(Tensor::from_f32({1,1,4},{1,2,3,4}),{1},10000); nearv("rope rotate_half dim0",r1.get_f32(0),std::cos(1.f)-3*std::sin(1.f)); nearv("rope rotate_half dim1",r1.get_f32(1),2*std::cos(.01f)-4*std::sin(.01f)); nearv("rope rotate_half dim2",r1.get_f32(2),std::sin(1.f)+3*std::cos(1.f)); nearv("rope rotate_half dim3",r1.get_f32(3),2*std::sin(.01f)+4*std::cos(.01f)); throws("rope odd head_dim",[&]{rope(Tensor::from_f32({1,1,3},{1,2,3}),{0},10000);}); throws("rope position length",[&]{rope(Tensor::from_f32({2,1,4},{1,2,3,4,1,2,3,4}),{0},10000);});
 auto m=causal_mask(2);check(m.get_f32(0)==0&&std::isinf(m.get_f32(1)),"causal_mask shape=[2,2]");auto sm=softmax_last_dim(Tensor::from_f32({1,2},{1000,1001}));nearv("softmax stable sum",sm.get_f32(0)+sm.get_f32(1),1);
 auto q=Tensor::from_f32({2,4,1},{0,0,0,0,0,0,0,0}),k=Tensor::from_f32({2,2,1},{0,0,0,0}),v=Tensor::from_f32({2,2,1},{10,20,30,40});auto att=gqa_attention(q,k,v);nearv("gqa map q0 kv0",att.get_f32(0),10);nearv("gqa map q1 kv0",att.get_f32(1),10);nearv("gqa map q2 kv1",att.get_f32(2),20);nearv("gqa map q3 kv1",att.get_f32(3),20);auto vfuture=Tensor::from_f32({2,2,1},{10,20,9999,8888});auto early=gqa_attention(q,k,vfuture);for(int i=0;i<4;++i)nearv("gqa causal future",early.get_f32(i),att.get_f32(i));auto huge=gqa_attention(Tensor::from_f32({1,1,1},{1e10f}),Tensor::from_f32({1,1,1},{1e10f}),Tensor::from_f32({1,1,1},{2}));check(std::isfinite(huge.get_f32(0)),"gqa stable large QK");throws("gqa zero q heads",[&]{gqa_attention(Tensor::from_f32({2,0,1},{}),Tensor::from_f32({2,2,1},{}),Tensor::from_f32({2,2,1},{}));});throws("gqa zero kv heads",[&]{gqa_attention(q,Tensor::from_f32({2,0,1},{}),v);});throws("gqa zero head_dim",[&]{gqa_attention(Tensor::from_f32({2,4,0},{}),Tensor::from_f32({2,2,0},{}),Tensor::from_f32({2,2,0},{}));});throws("gqa nondivisible heads",[&]{gqa_attention(q,Tensor::from_f32({2,3,1},{0,0,0,0,0,0}),Tensor::from_f32({2,3,1},{0,0,0,0,0,0}));});
 auto sg=swiglu(x,x);check(sg.get_f32(0)>0,"swiglu shape=[1,2]");auto ad=add(x,x);nearv("add residual",ad.get_f32(0),4);throws("add shape mismatch",[&]{add(x,Tensor::from_f32({1,3},{1,2,3}));});auto lm=lm_head(x,w);nearv("lm_head tied linear",lm.get_f32(0),l.get_f32(0));Tensor f16(DType::F16,{1});throws("non-F32 operator input",[&]{add(f16,f16);});
 std::cout<<"test_ops_cpu passed\n";
}
