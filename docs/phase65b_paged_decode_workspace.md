# Phase 6.5b：Paged Decode Workspace

Phase 6.5b 将 Qwen3 FP16 paged decode 的层内临时结果接入模型独占的
`DecodeWorkspace`。`hidden_a/hidden_b`、input RMSNorm、Q/K/V projection、
attention、residual、post-attention RMSNorm、gate/up/SwiGLU、down 和 final
RMSNorm buffer 按最大 batch=4、FP16 预分配，并在 B=1/2/4 时使用前缀 view。

每层通过 `*_out` 算子写入 caller-provided buffer；K/V 直接从 workspace 的
non-owning row view 追加到 paged cache，paged attention 仍使用 Phase 6.5a 的
一次 batch kernel launch。positions 与 block-table 继续由 metadata workspace
复用。最终 logits 保留一次独立返回分配，避免下一次 decode 静默覆盖调用方仍持有
的 Tensor；因此 steady-state 总 allocation 允许不超过 2 次，层内临时 allocation
目标为 0。

workspace 由 `Qwen3CudaModel` 通过 Tensor RAII 持有，首次初始化必须在 warmup
期间完成；同一 model 的 paged decode 当前禁止并发调用。测试覆盖 B=1/2/4、连续
decode、serial 数值对齐、metadata 48 bytes 常驻和 warmup 后 allocation 上限。

本阶段不改变模型公式、KV pool/block layout、事务回滚、采样语义或 contiguous
decode；真正 batched paged prefill、paged scheduler 生产化接入、prefix sharing/COW、
long-context、FlashAttention/warp-tiled 融合和 CUDA Graph 仍未完成。
