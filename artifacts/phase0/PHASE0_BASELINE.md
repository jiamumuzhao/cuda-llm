# Phase 0 基线报告

生成时间：Qwen/Qwen3-0.6B / NVIDIA GeForce RTX 2080 Ti

## 环境

- GPU：`0, NVIDIA GeForce RTX 2080 Ti, 7.5, 11264 MiB, 595.71.05, P8
1, NVIDIA GeForce RTX 2080 Ti, 7.5, 11264 MiB, 595.71.05, P2`
- Python：`3.13.9 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 19:16:10) [GCC 11.2.0]`
- PyTorch：`2.11.0+cu130`，CUDA runtime：`13.0`
- Transformers：`5.12.1`
- CUDA compiler：`/usr/local/cuda/bin/nvcc`
- CUDA smoke：`artifacts/phase0/cuda_smoke`（sm_75 kernel round-trip）

## Qwen3-0.6B reference

- dtype：`torch.float16`
- prompt tokens：`13`
- next token：`481` (` -`)
- prefill：`61.87` tokens/s
- greedy generation：`33.53` tokens/s
- peak allocated memory：`1154.99` MiB

## Golden 文件

- `environment.json`：工具链、GPU、Python 包和 CUDA 状态
- `reference.json`：可读的 prompt、配置、速度和生成结果
- `golden.pt`：输入 token、最后位置 logits、生成 token，可供后续 kernel 对比

## Phase 0 状态

- [x] 环境和版本记录
- [x] GPU/CUDA 可用性检查
- [x] Qwen3-0.6B PyTorch reference
- [x] deterministic greedy golden
- [x] prefill/decode 基线指标
- [x] 显存峰值记录
- [x] `nvcc -arch=sm_75` 编译并执行 CUDA smoke kernel
- [x] 两次运行 golden 一致（next token、generated ids、logits max diff=0）
- [ ] CUDA/C++ kernel 基线（Phase 1/2）
