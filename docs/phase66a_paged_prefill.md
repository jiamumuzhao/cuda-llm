# Phase 6.6a：单请求 Direct Paged Prefill

`Qwen3CudaModel::prefill_logits_paged` 支持 Qwen3 FP16、B=1、`1..32` token 的
direct paged prefill。embedding 和已有 prefill layer 计算仍按已验证公式执行；每层
生成的 RoPE 后 K 与 V projection 直接以 CUDA D2D copy 写入
`Qwen3PagedKvCache` 的 physical block pool。不会构造 contiguous cache，也不会
调用 `seed_from_contiguous_cache`。

## 事务语义

`Qwen3PagedKvCache` 使用 `None/Decode/Prefill` 三态事务。`begin_prefill(S)` 仅
接受空 cache，并一次性申请 `ceil(S / block_size)` 个逻辑 block。每层只能写一次
`[S,8,128]` CUDA/F16 contiguous K/V；28 层全部完成后 `commit_prefill()` 在同步
确认 CUDA 工作成功后提交 length=S。任何 shape、重复 layer、遗漏 layer、pool
耗尽或 CUDA 异常都可以通过 `abort_prefill()` 回滚到 begin 前的 block table、pool
引用和 length。

## 覆盖范围

`test_qwen3_paged_prefill` 覆盖 context 1、4、15、16、17、31，确定性 fragmented
non-contiguous block table（例如 `[0,2]`，保留 block 不进入 direct table），
logits/top-5/sampling、全部 28 层全部 token 的 K/V bit-exact 对齐，以及 direct
prefill 后 paged decode 的 logits 和 cache length 接续。cache 单元测试覆盖
prefill transaction 的完整 28 层 commit、显式 abort、pool exhaustion 回滚、
重复/遗漏 layer 和 decode/prefill 冲突；模型级测试另覆盖中间层 fault injection
后的 catch→abort→同 cache 重试恢复，以及输入/状态拒绝路径。

本阶段仅支持单请求 direct prefill；B=2/4 paged prefill、paged prefill attention、
scheduler direct admission、长上下文、prefix sharing/COW、FlashAttention、
CUDA Graph 和性能优化仍未实现。
