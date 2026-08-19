#include "llm/cuda_check.h"
#include "llm/qwen3_cuda_model.h"

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
static void fail(const std::string& s) { throw std::runtime_error("test_qwen3_paged_batched_prefill: " + s); }
static void expect(bool ok, const std::string& s) { if (!ok) fail(s); }
static void expect_throw(const std::string& name, const std::function<void()>& fn) {
  try { fn(); } catch (const std::exception& e) { std::cout << "rejected " << name << ": " << e.what() << "\n"; return; }
  fail(name + " was accepted");
}
static std::vector<int32_t> prompt(size_t n, int salt) {
  std::vector<int32_t> ids(n);
  for (size_t i = 0; i < n; ++i) ids[i] = 1 + (salt + static_cast<int>(i * 37)) % 150000;
  return ids;
}
static PagedKvCachePoolConfig pc(size_t blocks) { return {blocks,28,8,16,128,DType::F16}; }
static std::vector<int32_t> top5(const Tensor& x, size_t row, size_t rows) {
  Tensor c = x.to(DeviceType::CPU);
  std::vector<int32_t> ids(151936); std::iota(ids.begin(), ids.end(), 0);
  const size_t base = row * 151936;
  std::partial_sort(ids.begin(), ids.begin()+5, ids.end(), [&](int a, int b) {
    const float av = c.get_f32(base+a), bv = c.get_f32(base+b);
    return av != bv ? av > bv : a < b;
  });
  ids.resize(5); (void)rows; return ids;
}
static void compare(const Tensor& a, const Tensor& b, const std::string& name) {
  Tensor ac=a.to(DeviceType::CPU), bc=b.to(DeviceType::CPU);
  expect(ac.shape()==bc.shape(), name+" shape mismatch");
  float max_abs=0;
  for(size_t i=0;i<ac.numel();++i) max_abs=std::max(max_abs,std::fabs(ac.get_f32(i)-bc.get_f32(i)));
  expect(max_abs<=5e-3f,name+" max_abs_error="+std::to_string(max_abs));
  std::cout<<name<<" max_abs_error="<<max_abs<<"\n";
}
static void compare_kv(const Qwen3PagedKvCache& paged, const Qwen3KvCache& ref, size_t seq) {
  for(size_t layer=0;layer<28;++layer){
    Tensor k=ref.key_cache(layer).to(DeviceType::CPU),v=ref.value_cache(layer).to(DeviceType::CPU);
    for(size_t t=0;t<seq;++t){
      std::vector<uint16_t> pk(1024),pv(1024); paged.copy_layer_kv_to_host(layer,t,pk.data(),pv.data(),1024);
      for(size_t i=0;i<1024;++i) if(pk[i]!=k.data_f16()[t*1024+i]||pv[i]!=v.data_f16()[t*1024+i])
        fail("K/V bit mismatch layer="+std::to_string(layer)+" token="+std::to_string(t)+" index="+std::to_string(i));
    }
  }
}

struct CacheSnapshot {
  size_t length;
  std::vector<BlockId> table;
  bool prefill;
  bool decode;
};
struct PoolSnapshot { size_t free_blocks; size_t used_blocks; };
static CacheSnapshot snapshot(const Qwen3PagedKvCache& c) {
  return {c.length(), c.block_table(), c.in_prefill_transaction(), c.in_decode_transaction()};
}
static PoolSnapshot snapshot(const PagedKvCachePool& p) {
  return {p.free_block_count(), p.used_block_count()};
}
static void expect_same(const CacheSnapshot& a, const CacheSnapshot& b, const std::string& name) {
  expect(a.length == b.length && a.table == b.table && a.prefill == b.prefill && a.decode == b.decode,
         name + " cache state changed");
}
static void expect_same(const PoolSnapshot& a, const PoolSnapshot& b, const std::string& name) {
  expect(a.free_blocks == b.free_blocks && a.used_blocks == b.used_blocks,
         name + " pool state changed");
}

int main(){
  int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices));
  if(!devices){std::cout<<"SKIP test_qwen3_paged_batched_prefill: no CUDA device\n";return 0;}
  try{
    struct FaultGuard { ~FaultGuard() noexcept { qwen3_clear_paged_prefill_fault_for_testing(); } } fault_guard;
    Qwen3CudaModel model("artifacts/phase15/qwen3-0.6b-f32");
    for(size_t batch:{size_t(1),size_t(2),size_t(4)}) for(size_t seq:{size_t(1),size_t(4),size_t(16),size_t(17),size_t(31)}){
      PagedKvCachePool pool(pc(64));
      const BlockId blocker_a=pool.block_manager().allocate();
      const BlockId blocker_b=pool.block_manager().allocate();
      pool.block_manager().release(blocker_a);
      std::vector<std::vector<int32_t>> prompts; prompts.reserve(batch);
      std::vector<Qwen3KvCache> refs; refs.reserve(batch);
      std::vector<std::unique_ptr<Qwen3PagedKvCache>> paged; paged.reserve(batch);
      std::vector<Qwen3PagedKvCache*> pp; pp.reserve(batch);
      for(size_t b=0;b<batch;++b){prompts.push_back(prompt(seq,100+int(b*13+seq)));refs.emplace_back(32);paged.emplace_back(std::make_unique<Qwen3PagedKvCache>(pool,32));pp.push_back(paged.back().get());}
      Tensor expected=model.prefill_logits_batch_with_caches(prompts,[&]{std::vector<Qwen3KvCache*> x;for(auto& c:refs)x.push_back(&c);return x;}());
      Tensor actual=model.prefill_logits_paged_batch(prompts,pp);
      compare(actual,expected,"B="+std::to_string(batch)+" seq="+std::to_string(seq)+" prefill");
      const size_t row_count=batch*seq;
      Tensor af=actual.reshape({static_cast<int64_t>(row_count),151936});
      Tensor ef=expected.reshape({static_cast<int64_t>(row_count),151936});
      for(size_t b=0;b<batch;++b){
        expect(top5(af,b*seq+seq-1,row_count)==top5(ef,b*seq+seq-1,row_count),"last-token top5 mismatch");
        CudaSampler sampler(151936); SamplingConfig s;s.temperature=.8f;s.top_k=50;s.top_p=.9f;s.seed=20260727;
        expect(sampler.sample_row(af,b*seq+seq-1,s,11)==sampler.sample_row(ef,b*seq+seq-1,s,11),"sampling mismatch");
        compare_kv(*paged[b],refs[b],seq);
      }
      std::vector<int32_t> next(batch);for(size_t b=0;b<batch;++b)next[b]=901+int(b);
      Tensor paged_decode=model.decode_logits_paged_batch(next,pp);
      std::vector<Qwen3KvCache*> rp;for(auto& c:refs)rp.push_back(&c);
      Tensor ref_decode=model.decode_logits_batch_with_caches(next,rp);
      compare(paged_decode,ref_decode,"B="+std::to_string(batch)+" seq="+std::to_string(seq)+" decode");
      for(size_t b=0;b<batch;++b)expect(paged[b]->length()==seq+1,"paged cache length");
      for(auto& c:paged)c->release_all();pool.block_manager().release(blocker_b);
      expect(pool.used_block_count()==0,"batch paged prefill leaked blocks");
      if(seq>=17){bool noncont=false;for(const auto& c:paged) (void)c; /* tables are checked before release below in dedicated case */ (void)noncont;}
    }

    // Independent B=2/B=4 fragmented physical-table coverage.
    for (size_t batch : {size_t(2), size_t(4)}) {
      // Make the free-list allocation order [0,2,3,5,6,8,...], leaving
      // 1,4,7,... retained as blockers.  Thus every row's two logical
      // blocks are physically non-contiguous, not just the first row.
      PagedKvCachePool pool(pc(batch * 3));
      std::vector<BlockId> all_blocks;
      for (size_t i = 0; i < batch * 3; ++i) all_blocks.push_back(pool.block_manager().allocate());
      std::vector<BlockId> blockers;
      std::vector<BlockId> to_release;
      for (size_t i = 0; i < all_blocks.size(); ++i) {
        if (i % 3 == 1) blockers.push_back(all_blocks[i]);
        else to_release.push_back(all_blocks[i]);
      }
      for (auto it = to_release.rbegin(); it != to_release.rend(); ++it) pool.block_manager().release(*it);
      std::vector<std::vector<int32_t>> ids;
      std::vector<Qwen3KvCache> refs;
      std::vector<std::unique_ptr<Qwen3PagedKvCache>> paged;
      std::vector<Qwen3PagedKvCache*> caches;
      for (size_t bidx = 0; bidx < batch; ++bidx) {
        ids.push_back(prompt(17, 700 + static_cast<int>(bidx)));
        refs.emplace_back(32);
        paged.emplace_back(std::make_unique<Qwen3PagedKvCache>(pool, 32));
        caches.push_back(paged.back().get());
      }
      std::vector<Qwen3KvCache*> ref_ptrs;
      for (auto& ref : refs) ref_ptrs.push_back(&ref);
      Tensor expected = model.prefill_logits_batch_with_caches(ids, ref_ptrs);
      Tensor actual = model.prefill_logits_paged_batch(ids, caches);
      compare(actual, expected, "fragmented B=" + std::to_string(batch) + " prefill");
      Tensor actual_rows = actual.reshape({static_cast<int64_t>(batch * 17), 151936});
      Tensor expected_rows = expected.reshape({static_cast<int64_t>(batch * 17), 151936});
      for (size_t bidx = 0; bidx < batch; ++bidx) {
        const auto& table = paged[bidx]->block_table();
        expect(table.size() >= 2, "fragmented B=" + std::to_string(batch) + " row has fewer than two blocks");
        std::cout << "fragmented B=" << batch << " row=" << bidx << " table=[";
        for (size_t j = 0; j < table.size(); ++j) std::cout << (j ? "," : "") << table[j];
        std::cout << "] blockers=" << blockers.size() << "\n";
        bool noncontiguous = false;
        for (size_t i = 0; i + 1 < table.size(); ++i)
          noncontiguous |= table[i + 1] != table[i] + 1;
        expect(noncontiguous, "fragmented B=" + std::to_string(batch) + " row is contiguous");
        for (const BlockId blocker : blockers)
          expect(std::find(table.begin(), table.end(), blocker) == table.end(),
                 "fragmented blocker entered B=" + std::to_string(batch) + " row");
        expect(top5(actual_rows, bidx * 17 + 16, batch * 17) ==
                   top5(expected_rows, bidx * 17 + 16, batch * 17),
               "fragmented top5 mismatch");
        CudaSampler sampler(151936); SamplingConfig sampling;
        sampling.temperature = .8f; sampling.top_k = 50; sampling.top_p = .9f; sampling.seed = 20260727;
        expect(sampler.sample_row(actual_rows, bidx * 17 + 16, sampling, 11) ==
                   sampler.sample_row(expected_rows, bidx * 17 + 16, sampling, 11),
               "fragmented sampling mismatch");
        compare_kv(*paged[bidx], refs[bidx], 17);
      }
      std::vector<int32_t> next(batch, 901);
      Tensor paged_decode = model.decode_logits_paged_batch(next, caches);
      Tensor ref_decode = model.decode_logits_batch_with_caches(next, ref_ptrs);
      compare(paged_decode, ref_decode, "fragmented B=" + std::to_string(batch) + " decode");
      for (auto& cache : paged) {
        expect(cache->length() == 18, "fragmented decode length mismatch");
        cache->release_all();
      }
      for (const BlockId blocker : blockers) pool.block_manager().release(blocker);
      expect(pool.used_block_count() == 0, "fragmented B=" + std::to_string(batch) + " leaked blocks");
      std::cout << "B=" << batch << " fragmented table/KV/top5/sampling/decode passed\n";
    }

    // Batch-input/state rejection coverage.  Every case snapshots all cache
    // and pool metadata so validation is proved to be pre-transactional.
    {
      PagedKvCachePool pool(pc(32));
      Qwen3PagedKvCache c0(pool, 32), c1(pool, 32), c2(pool, 32), c3(pool, 32);
      const auto reject = [&](const std::string& name,
                              const std::vector<std::vector<int32_t>>& ids,
                              const std::vector<Qwen3PagedKvCache*>& caches,
                              const std::function<void()>& fn) {
        std::vector<CacheSnapshot> before;
        for (auto* cache : caches) if (cache) before.push_back(snapshot(*cache));
        const PoolSnapshot pool_before = snapshot(pool);
        expect_throw(name, fn);
        size_t index = 0;
        for (auto* cache : caches) if (cache) expect_same(before[index++], snapshot(*cache), name);
        expect_same(pool_before, snapshot(pool), name);
        (void)ids;
      };
      const auto one = prompt(4, 1);
      const auto two = prompt(4, 2);
      reject("batch size 0", {}, {}, [&] { model.prefill_logits_paged_batch({}, {}); });
      reject("batch size 3", {one, two, one}, {&c0, &c1, &c2}, [&] {
        model.prefill_logits_paged_batch({one, two, one}, {&c0, &c1, &c2});
      });
      reject("prompt/cache count mismatch", {one, two}, {&c0}, [&] {
        model.prefill_logits_paged_batch({one, two}, {&c0});
      });
      reject("null cache", {one, two}, {&c0, nullptr}, [&] {
        model.prefill_logits_paged_batch({one, two}, {&c0, nullptr});
      });
      reject("duplicate cache", {one, two}, {&c0, &c0}, [&] {
        model.prefill_logits_paged_batch({one, two}, {&c0, &c0});
      });
      {
        PagedKvCachePool other_pool(pc(8)); Qwen3PagedKvCache other(other_pool, 32);
        reject("different pools", {one, two}, {&c0, &other}, [&] {
          model.prefill_logits_paged_batch({one, two}, {&c0, &other});
        });
      }
      reject("different prompt lengths", {one, prompt(5, 3)}, {&c0, &c1}, [&] {
        model.prefill_logits_paged_batch({one, prompt(5, 3)}, {&c0, &c1});
      });
      {
        Qwen3PagedKvCache small(pool, 16);
        reject("capacity too small", {prompt(17, 4), prompt(17, 5)}, {&small, &c1}, [&] {
          model.prefill_logits_paged_batch({prompt(17, 4), prompt(17, 5)}, {&small, &c1});
        });
      }
      auto negative = one; negative[0] = -1;
      auto upper = one; upper[0] = 151936;
      reject("negative token", {negative}, {&c0}, [&] { model.prefill_logits_paged_batch({negative}, {&c0}); });
      reject("vocab upper-bound token", {upper}, {&c0}, [&] { model.prefill_logits_paged_batch({upper}, {&c0}); });
      model.prefill_logits_paged({one}, c0);
      reject("non-empty cache", {one}, {&c0}, [&] { model.prefill_logits_paged_batch({one}, {&c0}); });
      c0.release_all();
      c1.begin_prefill(1);
      reject("active prefill transaction", {one}, {&c1}, [&] { model.prefill_logits_paged_batch({one}, {&c1}); });
      c1.abort_prefill();
      model.prefill_logits_paged({one}, c1);
      c1.begin_decode();
      reject("active decode transaction", {one}, {&c1}, [&] { model.prefill_logits_paged_batch({one}, {&c1}); });
      c1.abort_decode(); c1.release_all();
      // Reuse caches after rejection: no transaction or block metadata is dirty.
      Tensor reusable = model.prefill_logits_paged_batch({one, two}, {&c0, &c1});
      expect(reusable.shape() == std::vector<int64_t>{2, 4, 151936}, "reusable batch shape");
      c0.release_all(); c1.release_all();
      expect(pool.used_block_count() == 0, "rejection/reuse scenario leaked blocks");
      for (size_t bad_layers : {size_t(27)}) {
        expect_throw("incompatible pool layers", [&] {
          PagedKvCachePool bad({8, bad_layers, 8, 16, 128, DType::F16}); Qwen3PagedKvCache bad_cache(bad, 32);
        });
      }
      expect_throw("incompatible pool KV heads", [&] {
        PagedKvCachePool bad({8, 28, 4, 16, 128, DType::F16}); Qwen3PagedKvCache bad_cache(bad, 32);
      });
      expect_throw("incompatible pool head dim", [&] {
        PagedKvCachePool bad({8, 28, 8, 16, 64, DType::F16}); Qwen3PagedKvCache bad_cache(bad, 32);
      });
      expect_throw("incompatible pool dtype", [&] {
        PagedKvCachePool bad({8, 28, 8, 16, 128, DType::F32}); Qwen3PagedKvCache bad_cache(bad, 32);
      });
      std::cout << "batched prefill input/state rejection and cache reuse passed\n";
    }

    // Pool exhaustion during begin_prefill: earlier rows have already begun,
    // so the model catch path must abort every row transaction.
    for (size_t batch : {size_t(2), size_t(4)}) {
      const size_t total_blocks = batch * 2;
      PagedKvCachePool pool(pc(total_blocks));
      const BlockId blocker = pool.block_manager().allocate();
      std::vector<std::unique_ptr<Qwen3PagedKvCache>> owned;
      std::vector<Qwen3PagedKvCache*> caches;
      std::vector<std::vector<int32_t>> ids;
      for (size_t i = 0; i < batch; ++i) {
        owned.emplace_back(std::make_unique<Qwen3PagedKvCache>(pool, 32));
        caches.push_back(owned.back().get()); ids.push_back(prompt(17, 900 + static_cast<int>(i)));
      }
      const PoolSnapshot before = snapshot(pool);
      expect_throw("B=" + std::to_string(batch) + " begin_prefill pool exhaustion", [&] {
        model.prefill_logits_paged_batch(ids, caches);
      });
      expect_same(before, snapshot(pool), "B=" + std::to_string(batch) + " begin exhaustion");
      for (auto* cache : caches) {
        expect(cache->length() == 0 && cache->block_table().empty() &&
                   !cache->in_prefill_transaction() && !cache->in_decode_transaction(),
               "begin exhaustion did not abort every row");
      }
      expect(pool.block_manager().is_allocated(blocker), "begin exhaustion changed blocker");
      pool.block_manager().release(blocker);
      Tensor recovered = model.prefill_logits_paged_batch(ids, caches);
      expect(recovered.shape() == std::vector<int64_t>{static_cast<int64_t>(batch), 17, 151936},
             "begin exhaustion recovery shape");
      for (auto* cache : caches) cache->release_all();
      expect(pool.used_block_count() == 0, "begin exhaustion recovery leaked blocks");
      std::cout << "B=" << batch << " begin-prefill exhaustion all-row rollback/recovery passed\n";
    }

    // Fragmentation evidence with a live table.  Keep this successful case
    // separate from the fault/retry case below: aborting multiple rows may
    // legitimately change the free-list order, while the allocator must still
    // be proven to hand a live sequence non-contiguous physical blocks.
    {
      PagedKvCachePool fragmented_pool(pc(16));
      const BlockId occupied_a = fragmented_pool.block_manager().allocate();
      const BlockId occupied_b = fragmented_pool.block_manager().allocate();
      fragmented_pool.block_manager().release(occupied_a);
      Qwen3PagedKvCache fragmented_cache(fragmented_pool,32);
      const std::vector<std::vector<int32_t>> fragmented_ids={prompt(17,17)};
      model.prefill_logits_paged_batch(fragmented_ids,{&fragmented_cache});
      expect(fragmented_cache.block_table().size()>=2,"fragmented table has fewer than two blocks");
      bool noncontiguous=false;
      for(size_t i=0;i+1<fragmented_cache.block_table().size();++i)
        noncontiguous|=fragmented_cache.block_table()[i+1]!=fragmented_cache.block_table()[i]+1;
      expect(noncontiguous,"fragmented table did not contain non-contiguous physical blocks");
      expect(std::find(fragmented_cache.block_table().begin(),fragmented_cache.block_table().end(),occupied_b)==fragmented_cache.block_table().end(),"fragmented blocker entered direct table");
      fragmented_cache.release_all();
      fragmented_pool.block_manager().release(occupied_b);
      expect(fragmented_pool.used_block_count()==0,"fragmented evidence leaked blocks");
      std::cout<<"fragmented direct table evidence passed\n";
    }

    // All-row rollback on fault, followed by successful recovery.
    PagedKvCachePool pool(pc(16)); BlockId a=pool.block_manager().allocate(), b=pool.block_manager().allocate(); pool.block_manager().release(a);
    std::vector<std::vector<int32_t>> ids={prompt(17,7),prompt(17,8)};
    Qwen3PagedKvCache c0(pool,32),c1(pool,32); std::vector<Qwen3PagedKvCache*> cs={&c0,&c1};
    qwen3_set_paged_prefill_fault_for_testing(11);
    expect_throw("batched paged prefill injected fault",[&]{model.prefill_logits_paged_batch(ids,cs);});
    qwen3_clear_paged_prefill_fault_for_testing();
    expect(c0.length()==0&&c1.length()==0&&c0.block_table().empty()&&c1.block_table().empty()&&pool.used_block_count()==1,"batch fault did not rollback all rows");
    Tensor recovered=model.prefill_logits_paged_batch(ids,cs);
    expect(recovered.shape()==std::vector<int64_t>{2,17,151936},"batch recovery shape");
    expect(c0.block_table().size()>=2,"recovered batch table has fewer than two blocks");
    std::cout<<"all-row fault rollback and recovery passed\n";
    c0.release_all();c1.release_all();pool.block_manager().release(b);expect(pool.used_block_count()==0,"recovery leaked blocks");
    {
      PagedKvCachePool fault_pool(pc(32));
      std::vector<std::unique_ptr<Qwen3PagedKvCache>> fault_owned;
      std::vector<Qwen3PagedKvCache*> fault_caches;
      std::vector<std::vector<int32_t>> fault_ids;
      for (size_t i = 0; i < 4; ++i) {
        fault_owned.emplace_back(std::make_unique<Qwen3PagedKvCache>(fault_pool, 32));
        fault_caches.push_back(fault_owned.back().get());
        fault_ids.push_back(prompt(17, 1000 + static_cast<int>(i)));
      }
      const PoolSnapshot before = snapshot(fault_pool);
      qwen3_set_paged_prefill_fault_for_testing(13);
      expect_throw("B=4 batched paged prefill injected fault", [&] {
        model.prefill_logits_paged_batch(fault_ids, fault_caches);
      });
      qwen3_clear_paged_prefill_fault_for_testing();
      expect_same(before, snapshot(fault_pool), "B=4 layer fault");
      for (auto* cache : fault_caches)
        expect(cache->length() == 0 && cache->block_table().empty() &&
                   !cache->in_prefill_transaction() && !cache->in_decode_transaction(),
               "B=4 layer fault did not rollback every row");
      Tensor recovered4 = model.prefill_logits_paged_batch(fault_ids, fault_caches);
      expect(recovered4.shape() == std::vector<int64_t>{4, 17, 151936}, "B=4 fault recovery shape");
      for (auto* cache : fault_caches) cache->release_all();
      expect(fault_pool.used_block_count() == 0, "B=4 fault recovery leaked blocks");
      std::cout << "B=4 layer fault all-row rollback/recovery passed\n";
    }
    CUDA_CHECK(cudaDeviceSynchronize()); std::cout<<"test_qwen3_paged_batched_prefill passed\n";return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
