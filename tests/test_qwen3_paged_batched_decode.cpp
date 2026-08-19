#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"
#include "llm/qwen3_layer_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
static void fail(const std::string& s) { throw std::runtime_error("test_qwen3_paged_batched_decode: " + s); }
static void expect(bool v, const std::string& s) { if (!v) fail(s); }
static void expect_throw(const std::string& n, const std::function<void()>& f) {
  try { f(); } catch (const std::exception& e) { std::cout << "rejected " << n << ": " << e.what() << "\n"; return; }
  fail(n + " was accepted");
}
static void expect_throw_contains(const std::string& n, const std::string& needle,
                                  const std::function<void()>& f) {
  try {
    f();
  } catch (const std::exception& e) {
    const std::string message = e.what();
    expect(message.find(needle) != std::string::npos,
           n + " error missing '" + needle + "': " + message);
    std::cout << "rejected " << n << ": " << message << "\n";
    return;
  }
  fail(n + " was accepted");
}
struct CacheSnapshot {
  size_t length;
  std::vector<BlockId> table;
  bool in_transaction;
};
struct PoolSnapshot {
  size_t free_blocks;
  size_t used_blocks;
};
static CacheSnapshot snapshot(const Qwen3PagedKvCache& cache) {
  return {cache.length(), cache.block_table(), cache.in_decode_transaction()};
}
static PoolSnapshot snapshot(const PagedKvCachePool& pool) {
  return {pool.free_block_count(), pool.used_block_count()};
}
static void expect_unchanged(const CacheSnapshot& before,
                             const Qwen3PagedKvCache& cache,
                             const std::string& name) {
  expect(cache.length() == before.length, name + " length changed");
  expect(cache.block_table() == before.table, name + " block table changed");
  expect(cache.in_decode_transaction() == before.in_transaction,
         name + " transaction state changed");
}
static void expect_unchanged(const PoolSnapshot& before,
                             const PagedKvCachePool& pool,
                             const std::string& name) {
  expect(pool.free_block_count() == before.free_blocks,
         name + " free block count changed");
  expect(pool.used_block_count() == before.used_blocks,
         name + " used block count changed");
}
static std::vector<int32_t> prompt(size_t n, int salt) {
  std::vector<int32_t> out(n);
  for (size_t i=0;i<n;++i) out[i] = 1 + (salt + int(i*37)) % 150000;
  return out;
}
static std::vector<int32_t> top5(const Tensor& t, size_t row) {
  Tensor c=t.to(DeviceType::CPU); std::vector<int32_t> ids(151936); std::iota(ids.begin(),ids.end(),0);
  std::partial_sort(ids.begin(),ids.begin()+5,ids.end(),[&](int a,int b){
    float x=c.get_f32(row*151936+a), y=c.get_f32(row*151936+b); return x!=y?x>y:a<b;});
  ids.resize(5); return ids;
}
static void compare_row(const Tensor& batch, size_t row, const Tensor& serial,
                        const std::string& name) {
  Tensor a=batch.to(DeviceType::CPU), b=serial.to(DeviceType::CPU);
  float maxe=0; double mean=0,dot=0,aa=0,bb=0;
  for(size_t i=0;i<151936;++i){float x=a.get_f32(row*151936+i),y=b.get_f32(i),e=std::fabs(x-y);
    maxe=std::max(maxe,e);mean+=e;dot+=double(x)*y;aa+=double(x)*x;bb+=double(y)*y;}
  double cos=dot/std::sqrt(aa*bb);
  std::cout<<name<<" max_abs_error="<<maxe<<" mean_abs_error="<<mean/151936
           <<" cosine_similarity="<<cos<<" tolerance=5e-3\n";
  if(maxe>5e-3f||cos<0.999) fail(name+" logits mismatch");
}
static void compare_kv(const Qwen3PagedKvCache& a, const Qwen3PagedKvCache& b,
                       size_t layer, size_t token, const std::string& name) {
  std::vector<uint16_t> ak(1024),av(1024),bk(1024),bv(1024);
  a.copy_layer_kv_to_host(layer,token,ak.data(),av.data(),1024);
  b.copy_layer_kv_to_host(layer,token,bk.data(),bv.data(),1024);
  for(size_t i=0;i<1024;++i) {
    if(ak[i]!=bk[i]) fail(name+" K bit mismatch layer="+std::to_string(layer)+" index="+std::to_string(i));
    if(av[i]!=bv[i]) fail(name+" V bit mismatch layer="+std::to_string(layer)+" index="+std::to_string(i));
  }
}
static PagedKvCachePoolConfig pc(size_t blocks) { return {blocks,28,8,16,128,DType::F16}; }

static void run_case(Qwen3CudaModel& model, const std::vector<size_t>& lengths,
                     size_t steps) {
  const size_t n=lengths.size();
  PagedKvCachePool pool( pc(32) );
  std::vector<Qwen3KvCache> sources; sources.reserve(n);
  std::vector<std::unique_ptr<Qwen3PagedKvCache>> batch, serial; batch.reserve(n); serial.reserve(n);
  std::vector<Qwen3PagedKvCache*> bp, sp;
  for(size_t b=0;b<n;++b) {
    sources.emplace_back(32);
    model.prefill_logits_with_cache(prompt(lengths[b],int(100+b)),sources.back());
    batch.emplace_back(std::make_unique<Qwen3PagedKvCache>(pool,32)); batch.back()->seed_from_contiguous_cache(sources.back());
    serial.emplace_back(std::make_unique<Qwen3PagedKvCache>(pool,32)); serial.back()->seed_from_contiguous_cache(sources.back());
    bp.push_back(batch.back().get()); sp.push_back(serial.back().get());
  }
  for(size_t step=0;step<steps;++step) {
    std::vector<int32_t> ids(n); for(size_t b=0;b<n;++b) ids[b]=700+int(b*17+step);
    std::vector<Tensor> refs; refs.reserve(n);
    for(size_t b=0;b<n;++b) refs.push_back(model.decode_logits_paged(ids[b],*serial[b]));
    const size_t metadata_before = model.paged_decode_metadata_workspace_bytes();
    reset_cuda_allocation_stats();
    reset_cuda_paged_gqa_attention_decode_batch_launch_count();
    Tensor actual=model.decode_logits_paged_batch(ids,bp);
    const std::uint64_t batch_attention_launches =
        cuda_paged_gqa_attention_decode_batch_launch_count();
    expect(batch_attention_launches == 28,
           "paged batch layer must launch exactly one batch attention per layer");
    const CudaAllocationStats decode_allocations = cuda_allocation_stats();
    expect(decode_allocations.cuda_malloc_calls <= 2,
           "steady-state paged decode allocated more than two CUDA tensors");
    const PagedDecodeMetadataUploadStats metadata_stats =
        model.last_paged_decode_metadata_upload_stats();
    expect(metadata_stats.valid, "paged batch metadata stats must be valid");
    expect(metadata_stats.cuda_malloc_calls == 0 &&
               metadata_stats.cuda_free_calls == 0 &&
               metadata_stats.cuda_allocated_bytes == 0 &&
               metadata_stats.cuda_freed_bytes == 0,
           "paged batch metadata path allocated CUDA memory");
    expect(metadata_stats.positions_uploads == 1 &&
               metadata_stats.block_table_uploads == n,
           "paged batch metadata upload counts mismatch");
    const size_t metadata_after = model.paged_decode_metadata_workspace_bytes();
    if (metadata_before != 0)
      expect(metadata_after == metadata_before,
             "paged metadata workspace changed after lazy initialization");
    if (step == 0)
      std::cout << "B=" << n << " metadata_workspace_before=" << metadata_before
                << " after=" << metadata_after
                << " metadata_allocation_after_warmup="
                << "0"
                << " metadata_cuda_malloc_calls=" << metadata_stats.cuda_malloc_calls
                << " metadata_cuda_free_calls=" << metadata_stats.cuda_free_calls
                << " positions_uploads=" << metadata_stats.positions_uploads
                << " block_table_uploads=" << metadata_stats.block_table_uploads
                << " batch_attention_launches=" << batch_attention_launches
                << " total_decode_cuda_malloc_calls="
                << decode_allocations.cuda_malloc_calls << "\n";
    for(size_t b=0;b<n;++b) {
      compare_row(actual,b,refs[b],"paged batch B="+std::to_string(n)+" row="+std::to_string(b));
      if(top5(actual,b)!=top5(refs[b],0)) fail("top5 mismatch");
      CudaSampler sampler(151936); SamplingConfig s; s.temperature=.8f;s.top_k=50;s.top_p=.9f;s.seed=20260727;
      if(sampler.sample_row(actual,b,s,11+step)!=sampler.sample_last_row(refs[b],s,11+step))
        fail("sampling mismatch row="+std::to_string(b));
      expect(batch[b]->length()==lengths[b]+step+1,"batch cache length mismatch");
      compare_kv(*batch[b],*serial[b],7,batch[b]->length()-1,"layer7");
      compare_kv(*batch[b],*serial[b],27,batch[b]->length()-1,"layer27");
    }
  }
  for(auto& c:batch)c->release_all(); for(auto& c:serial)c->release_all();
  expect(pool.used_block_count()==0,"batch case leaked blocks");
  std::cout<<"B="<<n<<" mixed-length paged batch steps="<<steps<<" passed\n";
}

int main(){
  int d=0;CUDA_CHECK(cudaGetDeviceCount(&d)); if(!d){std::cout<<"SKIP test_qwen3_paged_batched_decode: no CUDA device\n";return 0;}
  try{
    cudaDeviceProp p{};CUDA_CHECK(cudaGetDeviceProperties(&p,0));
    std::cout<<"CUDA device="<<p.name<<" compute_capability="<<p.major<<"."<<p.minor<<"\n";
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    // Warm up the single-request metadata path so all later observations are
    // steady-state and exclude lazy workspace initialization.
    PagedKvCachePool warm_pool(pc(8));
    Qwen3KvCache warm_source(32);
    model.prefill_logits_with_cache(prompt(4,90), warm_source);
    Qwen3PagedKvCache warm_cache(warm_pool,32);
    warm_cache.seed_from_contiguous_cache(warm_source);
    Tensor warmup_logits = model.decode_logits_paged(901, warm_cache);
    const size_t warmup_workspace_bytes = model.paged_decode_metadata_workspace_bytes();
    expect(warmup_workspace_bytes == 48, "paged metadata workspace warmup bytes");
    Tensor steady_logits = model.decode_logits_paged(902, warm_cache);
    const PagedDecodeMetadataUploadStats single_metadata =
        model.last_paged_decode_metadata_upload_stats();
    expect(single_metadata.valid && single_metadata.cuda_malloc_calls == 0 &&
               single_metadata.cuda_free_calls == 0 &&
               single_metadata.cuda_allocated_bytes == 0 &&
               single_metadata.cuda_freed_bytes == 0 &&
               single_metadata.positions_uploads == 1 &&
               single_metadata.block_table_uploads == 1,
           "single paged decode metadata stats mismatch");
    std::cout << "B=1 single metadata_cuda_malloc_calls="
              << single_metadata.cuda_malloc_calls
              << " metadata_cuda_free_calls=" << single_metadata.cuda_free_calls
              << " positions_uploads=" << single_metadata.positions_uploads
              << " block_table_uploads=" << single_metadata.block_table_uploads
              << " resident_bytes=" << warmup_workspace_bytes << "\n";
    warm_cache.release_all();
    run_case(model,{4,6},1); run_case(model,{3,4,6,8},1); run_case(model,{4},3); run_case(model,{15,16},3);
    expect(model.paged_decode_metadata_workspace_bytes() == 48,
           "block_size=16 metadata workspace resident bytes must be 48");
    std::cout << "paged_decode_metadata_workspace=enabled resident_bytes=48 "
                 "(positions[4] + block_tables[4,2]); metadata allocations are 0 "
                 "after lazy initialization; layer temporary allocations remain measured separately\n";

    PagedKvCachePool pool(pc(16)); std::vector<Qwen3KvCache> src; std::vector<std::unique_ptr<Qwen3PagedKvCache>> c; std::vector<Qwen3PagedKvCache*> cp;
    for(size_t b=0;b<2;++b){src.emplace_back(32);model.prefill_logits_with_cache(prompt(4,int(300+b)),src.back());c.emplace_back(std::make_unique<Qwen3PagedKvCache>(pool,32));c.back()->seed_from_contiguous_cache(src.back());cp.push_back(c.back().get());}
    auto before0=c[0]->block_table(),before1=c[1]->block_table();size_t free0=pool.free_block_count(),used0=pool.used_block_count();
    qwen3_set_paged_batch_fault_for_testing(1,3);
    expect_throw("injected batch rollback",[&]{model.decode_logits_paged_batch({901,902},cp);});
    qwen3_clear_paged_batch_fault_for_testing();
    expect(c[0]->length()==4&&c[1]->length()==4&&c[0]->block_table()==before0&&c[1]->block_table()==before1,
           "rollback changed cache state");
    expect(pool.free_block_count()==free0&&pool.used_block_count()==used0,"rollback changed pool state");
    Tensor recovered=model.decode_logits_paged_batch({901,902},cp);
    expect(recovered.shape()==std::vector<int64_t>{2,151936},"recovery output shape");
    for(auto& x:c)x->release_all();expect(pool.used_block_count()==0,"rollback recovery leaked blocks");
    std::cout<<"B=2 injected row/layer fault all-or-nothing rollback/recovery passed\n";

    PagedKvCachePool reject_pool(pc(8));Qwen3PagedKvCache a(reject_pool,32),b(reject_pool,32);
    Qwen3KvCache s(32);model.prefill_logits_with_cache(prompt(4,500),s);a.seed_from_contiguous_cache(s);b.seed_from_contiguous_cache(s);
    expect_throw("batch size 3",[&]{model.decode_logits_paged_batch({1,2,3},{&a,&b,nullptr});});
    expect_throw("token/cache mismatch",[&]{model.decode_logits_paged_batch({1},{&a,&b});});
    expect_throw("duplicate cache",[&]{model.decode_logits_paged_batch({1,2},{&a,&a});});
    expect_throw("invalid token",[&]{model.decode_logits_paged_batch({151936,2},{&a,&b});});
    a.begin_decode();expect_throw("active transaction",[&]{model.decode_logits_paged_batch({1,2},{&a,&b});});a.abort_decode();
    a.release_all();b.release_all();

    // Every case below uses a valid B=2 shape so that the intended validation
    // branch, rather than an earlier batch-size/count check, is exercised.
    PagedKvCachePool null_pool(pc(8));
    Qwen3PagedKvCache null_a(null_pool,32);
    null_a.seed_from_contiguous_cache(s);
    const CacheSnapshot null_a_before = snapshot(null_a);
    const PoolSnapshot null_pool_before = snapshot(null_pool);
    expect_throw_contains("B=2 null cache pointer", "null cache", [&] {
      model.decode_logits_paged_batch({11,12}, {&null_a, nullptr});
    });
    expect_unchanged(null_a_before, null_a, "null-cache row 0");
    expect_unchanged(null_pool_before, null_pool, "null-cache pool");
    null_a.release_all();
    std::cout << "B=2 null cache rejection preserved row-0 state and pool state\n";

    PagedKvCachePool different_pool_a(pc(8)), different_pool_b(pc(8));
    Qwen3PagedKvCache different_a(different_pool_a,32), different_b(different_pool_b,32);
    different_a.seed_from_contiguous_cache(s);
    different_b.seed_from_contiguous_cache(s);
    const CacheSnapshot different_a_before = snapshot(different_a);
    const CacheSnapshot different_b_before = snapshot(different_b);
    const PoolSnapshot different_pool_a_before = snapshot(different_pool_a);
    const PoolSnapshot different_pool_b_before = snapshot(different_pool_b);
    expect_throw_contains("different PagedKvCachePool instances",
                          "caches must share one PagedKvCachePool", [&] {
      model.decode_logits_paged_batch({21,22}, {&different_a, &different_b});
    });
    expect_unchanged(different_a_before, different_a, "different-pool cache A");
    expect_unchanged(different_b_before, different_b, "different-pool cache B");
    expect_unchanged(different_pool_a_before, different_pool_a, "different-pool A");
    expect_unchanged(different_pool_b_before, different_pool_b, "different-pool B");
    different_a.release_all();
    different_b.release_all();
    std::cout << "B=2 different-pool rejection preserved both caches and pools\n";

    PagedKvCachePool full_pool(pc(8));
    Qwen3PagedKvCache full_cache(full_pool,16), nonfull_cache(full_pool,32);
    Qwen3KvCache full_source(16), short_source(32);
    model.prefill_logits_with_cache(prompt(16,600), full_source);
    model.prefill_logits_with_cache(prompt(4,601), short_source);
    full_cache.seed_from_contiguous_cache(full_source);
    nonfull_cache.seed_from_contiguous_cache(short_source);
    const CacheSnapshot full_before = snapshot(full_cache);
    const CacheSnapshot nonfull_before = snapshot(nonfull_cache);
    const PoolSnapshot full_pool_before = snapshot(full_pool);
    expect_throw_contains("B=2 full cache", "invalid length/transaction state", [&] {
      model.decode_logits_paged_batch({31,32}, {&full_cache, &nonfull_cache});
    });
    expect_unchanged(full_before, full_cache, "full cache");
    expect_unchanged(nonfull_before, nonfull_cache, "non-full peer cache");
    expect_unchanged(full_pool_before, full_pool, "full-cache pool");
    full_cache.release_all();
    nonfull_cache.release_all();
    std::cout << "B=2 full-cache rejection preserved both cache states and pool state\n";

    // Three blocks are occupied: row 0 can begin in its existing block, row 1
    // crosses block 16 and fails in begin_decode, proving this is not a
    // preflight free-block check. The model must abort row 0 as well.
    PagedKvCachePool exhausted_pool(pc(3));
    Qwen3PagedKvCache begin_a(exhausted_pool,32), begin_b(exhausted_pool,32), blocker(exhausted_pool,32);
    Qwen3KvCache begin_source_a(32), begin_source_b(32), blocker_source(32);
    model.prefill_logits_with_cache(prompt(4,700), begin_source_a);
    model.prefill_logits_with_cache(prompt(16,701), begin_source_b);
    model.prefill_logits_with_cache(prompt(4,702), blocker_source);
    begin_a.seed_from_contiguous_cache(begin_source_a);
    begin_b.seed_from_contiguous_cache(begin_source_b);
    blocker.seed_from_contiguous_cache(blocker_source);
    const CacheSnapshot begin_a_before = snapshot(begin_a);
    const CacheSnapshot begin_b_before = snapshot(begin_b);
    const CacheSnapshot blocker_before = snapshot(blocker);
    const PoolSnapshot exhausted_before = snapshot(exhausted_pool);
    expect(exhausted_pool.free_block_count() == 0 && exhausted_pool.used_block_count() == 3,
           "begin-exhaustion setup did not occupy the final physical block");
    expect_throw_contains("begin_decode pool exhaustion", "block pool exhausted", [&] {
      model.decode_logits_paged_batch({41,42}, {&begin_a, &begin_b});
    });
    expect_unchanged(begin_a_before, begin_a, "begin-exhaustion cache A");
    expect_unchanged(begin_b_before, begin_b, "begin-exhaustion cache B");
    expect_unchanged(blocker_before, blocker, "begin-exhaustion blocker");
    expect_unchanged(exhausted_before, exhausted_pool, "begin-exhaustion pool");
    blocker.release_all();
    Tensor recovered_begin = model.decode_logits_paged_batch({41,42}, {&begin_a, &begin_b});
    expect(recovered_begin.shape() == std::vector<int64_t>{2,151936},
           "begin-exhaustion recovery output shape");
    expect(begin_a.length() == 5 && begin_b.length() == 17,
           "begin-exhaustion recovery lengths");
    begin_a.release_all();
    begin_b.release_all();
    expect(exhausted_pool.used_block_count() == 0,
           "begin-exhaustion recovery leaked blocks");
    std::cout << "B=2 begin_decode pool exhaustion aborted all begun rows and recovered successfully\n";

    std::cout<<"paged batched rejection paths passed\n";
    CUDA_CHECK(cudaDeviceSynchronize());std::cout<<"test_qwen3_paged_batched_decode passed\n";return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
