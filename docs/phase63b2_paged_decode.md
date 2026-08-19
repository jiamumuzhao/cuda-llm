# Phase 6.3b-2：Qwen3 单请求 Paged Decode

本阶段接入了 `Qwen3CudaModel::decode_logits_paged`，仅支持 batch=1、Qwen3-0.6B
FP16 和 `max_seq_len <= 32`。每次调用先开启 `Qwen3PagedKvCache` 的 decode
transaction，再按 28 层执行现有 Q/K/V projection、Q/K RMSNorm、Qwen3 RoPE、
GQA causal attention、residual 和 MLP；每层 K/V 通过 checked D2D copy 写入
physical block，paged attention 通过一次上传的 device block table 读取逻辑序列。
所有层成功后才 commit；异常会 abort 并回滚 block table、长度和新增 block。

`tests/test_qwen3_paged_decode.cpp` 覆盖 context 4、15、16、17、31，包含
跨 block、非连续 physical block table、K/V bit-exact 检查、contiguous logits/
top-5/固定 seed sampling 对齐、多步 decode、pool exhaustion 和非法输入路径。

本阶段不实现 batch/scheduler paged decode、Paged continuous batching、prefix
sharing/COW、Paged Attention 优化、long context 或性能优化。
