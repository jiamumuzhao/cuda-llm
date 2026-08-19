#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
out = root / "artifacts/phase0"
env = json.loads((out / "environment.json").read_text())
ref = json.loads((out / "reference.json").read_text())
gpu = env.get("gpu_query", {}).get("stdout", "unknown")
report = f'''# Phase 0 基线报告

生成时间：{ref.get("model", "unknown")} / {ref.get("gpu", "unknown")}

## 环境

- GPU：`{gpu}`
- Python：`{env.get("python", "unknown").splitlines()[0]}`
- PyTorch：`{env.get("packages", {}).get("torch")}`，CUDA runtime：`{ref.get("cuda")}`
- Transformers：`{env.get("packages", {}).get("transformers")}`
- CUDA compiler：`{env.get("tools", {}).get("nvcc")}`
- CUDA smoke：`artifacts/phase0/cuda_smoke`（sm_75 kernel round-trip）

## Qwen3-0.6B reference

- dtype：`{ref.get("dtype")}`
- prompt tokens：`{ref.get("prompt_length")}`
- next token：`{ref.get("next_token_id")}` (`{ref.get("next_token")}`)
- prefill：`{ref.get("prefill_tokens_per_second"):.2f}` tokens/s
- greedy generation：`{ref.get("generation_tokens_per_second"):.2f}` tokens/s
- peak allocated memory：`{ref.get("peak_memory_bytes") / 1024 / 1024:.2f}` MiB

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
'''
(out / "PHASE0_BASELINE.md").write_text(report)
print(report)
