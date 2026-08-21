# CUDA/C++ 大语言模型推理框架路线

> 目标：从零实现一个以 CUDA/C++ 为核心的轻量 LLM 推理框架，首先稳定支持 Qwen3-0.6B，再逐步接近 llama.cpp、vLLM、SGLang 的核心运行时思想。
>
> 文档版本：2026-07-20

## 1. 结论先行

建议采用“两条线并行”的路线：

1. **主线：从零实现最小运行时**。先支持单卡、BF16/FP16、单请求和简单 batch，把 Transformer 前向、KV cache、采样、权重加载和基准测试做正确。这条线用于真正理解推理核心算法。
2. **工程线：以 llama.cpp/ggml 或 tiny-vllm 为对照和可选底座**。前者成熟、可移植、CUDA 后端完整；后者更接近教学型 C++/CUDA + Paged KV Cache。若目标是尽快形成可用服务，可以先基于 llama.cpp 扩展 Qwen3 与调度能力，再把从零实现的模块逐步替换进去。

不建议一开始直接复刻 vLLM 或 SGLang 的全部系统。它们的关键价值不只是 CUDA kernel，还包括请求调度、KV 内存管理、前缀复用、CUDA Graph、量化、分布式和服务协议。对于 0.6B 模型，先做可观测、可验证的单卡 runtime，收益最高。

最终建议形成如下层次：

```text
OpenAI-compatible HTTP / C++ API
          |
Request Manager + Continuous Batching Scheduler
          |
Sequence State + Block/Page KV Cache Manager
          |
Model Runner: prefill / decode / graph capture
          |
Qwen3 Model: RMSNorm, QKV, RoPE, GQA, SwiGLU, LM Head
          |
CUDA kernels + cuBLASLt/CUTLASS + CUDA runtime
```

## 2. 目标、边界和非目标

### 2.1 第一阶段的明确目标

- NVIDIA 单 GPU，Linux 优先；CUDA 12.x、C++17、CMake。
- 支持 Hugging Face 的 Qwen3-0.6B 权重转换到自己的二进制格式。
- 支持 BF16/FP16 推理；FP32 只用于 reference 和 debug。
- 支持 prefill、decode、KV cache、greedy、temperature/top-k/top-p 采样。
- 支持单请求，再支持静态 batch，最后支持 continuous batching。
- 提供 CLI、C++ API、基准测试和与 Transformers 的逐 token 对比工具。
- 有明确的内存统计、kernel 计时、TTFT、tokens/s、P50/P99 延迟指标。

### 2.2 暂不做

- MoE、视觉模型、训练、LoRA、分布式 tensor/pipeline/expert parallel。
- Beam search、复杂 grammar decoding、工具调用编排。
- 多后端支持。CUDA 稳定后再考虑 CPU/Metal/Vulkan。
- 一开始就支持所有 Qwen/Llama 变体。先把模型抽象设计成可扩展，但只实现 Qwen3 dense。

## 3. Qwen3-0.6B 的实现基线

Qwen3-0.6B 官方配置可见模型仓库的 `config.json`，核心参数为：

| 参数 | 数值 | 实现含义 |
|---|---:|---|
| hidden_size | 1024 | token hidden 向量宽度 |
| num_hidden_layers | 28 | Transformer block 数量 |
| num_attention_heads | 16 | Q 头数量 |
| num_key_value_heads | 8 | KV 头数量，属于 GQA |
| head_dim | 128 | 每个 attention head 的维度 |
| intermediate_size | 3072 | SwiGLU 中间层宽度 |
| vocab_size | 151936 | embedding/LM head 词表宽度 |
| max_position_embeddings | 40960 | 模型配置的最大位置长度 |
| rope_theta | 1000000 | RoPE 基频 |
| norm | RMSNorm | epsilon 为 1e-6 |
| dtype | bfloat16 | 原始权重推荐计算类型 |
| tie_word_embeddings | true | embedding 与 lm_head 共享权重 |

这组配置意味着：

- 每层 Q 投影输出 `16 * 128 = 2048`，K/V 各输出 `8 * 128 = 1024`。
- GQA 中每个 KV 头服务两个 Q 头，attention kernel 中要正确处理 `q_head / num_q_heads_per_kv` 的映射。
- BF16 KV cache 每个 token 的理论占用为 `28 * 8 * 128 * 2(K,V) * 2 bytes = 114,688 bytes`，约 112 KiB；4K 上下文单请求约 448 MiB，不应把 KV cache 设计成 `[batch, max_seq, ...]` 的大连续矩阵。
- 0.6B 很小，瓶颈会在不同阶段变化：单请求 decode 主要受内存带宽和 kernel launch 影响；prefill 更容易受 GEMM/attention 算力影响；多请求则受 KV 管理、调度和 batch 形状影响。

注意：Qwen3 的 chat 模式必须遵循官方 tokenizer/chat template；thinking 与 non-thinking 是生成配置和模板层面的行为，不能只靠模型前向层“猜”。

## 4. 参考项目与选择建议

| 项目 | 技术路线 | 适合借鉴 | 二次开发难度 | 建议 |
|---|---|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | C/C++、ggml、多个后端、CUDA kernel、GGUF | 张量抽象、权重格式、量化、CUDA backend、server API | 中 | **最稳妥的工程底座**；若想快速交付，优先考虑基于它扩展 |
| [tiny-vllm](https://github.com/jmaczan/tiny-vllm) | 教学型 C++/CUDA runtime | Paged KV、scheduler、从零理解 vLLM 思路 | 低到中 | **最适合学习和移植核心算法**；需审查项目活跃度和生产完整性 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA 优化 runtime、C++ executor、Python builder、专用 kernel | in-flight batching、paged KV、CUDA Graph、量化和高性能 kernel | 高 | 作为性能和设计参考，不建议作为第一版源码底座 |
| [MLC LLM](https://github.com/mlc-ai/mlc-llm) | TVM/编译器、MLCEngine、多后端 | graph/compiler、跨平台 runtime、模型编译 | 高 | 适合后期需要多后端或 AOT 的方向 |
| [SGLang](https://github.com/sgl-project/sglang) | Python 前端 + runtime + CUDA/C++ kernel | RadixAttention、prefix cache、结构化生成、服务调度 | 高 | 学习服务层和 prefix reuse；不适合作为轻量 C++ 起点 |
| [MNN-LLM](https://github.com/alibaba/MNN) | C++ 移动/边缘推理引擎 | 轻量部署、量化、端侧 memory planning | 中到高 | 若后续面向移动端/嵌入式可研究 |
| [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) | 轻量 C++、CUDA、边缘设备 | 端侧 runtime、C++ API、轻量服务 | 中到高 | 面向 Jetson/嵌入式时值得评估 |

### 4.1 推荐决策

- **学习算法 + 建立自己的框架**：从零实现，参考 tiny-vllm 的 Paged KV 思路，参考 llama.cpp 的 tensor/backend 边界。
- **尽快支持更多模型和可用 API**：fork llama.cpp，增加 Qwen3 模型映射、权重转换和必要 kernel；不要重写已成熟的 tokenizer、GGUF、后端抽象。
- **只追求 NVIDIA 峰值性能**：后期引入 TensorRT-LLM、cuBLASLt、CUTLASS、FlashInfer 或 FlashAttention 的 kernel，保留自己的 scheduler 和 cache manager。

### 4.2 不建议的底座

直接 fork vLLM/SGLang 后把 Python runtime 改成纯 C++，通常会把大量复杂性一起带入：依赖矩阵、模型注册、分布式、异步调度、编译扩展和服务层耦合都很重。它们更适合作为行为和算法参考实现。

## 5. 推荐仓库结构

```text
cuda-llm/
├── CMakeLists.txt
├── cmake/
├── include/llm/
│   ├── tensor.h              # shape/stride/dtype/device/view
│   ├── model_config.h
│   ├── model.h
│   ├── runtime.h
│   ├── kv_cache.h
│   └── sampler.h
├── src/
│   ├── tensor/
│   ├── runtime/
│   │   ├── engine.cpp
│   │   ├── scheduler.cpp
│   │   └── request.cpp
│   ├── model/qwen3/
│   ├── io/weights.cpp
│   └── tokenizer/
├── cuda/
│   ├── elementwise.cu
│   ├── rmsnorm.cu
│   ├── rope.cu
│   ├── attention.cu
│   ├── paged_attention.cu
│   ├── quant_matmul.cu
│   └── kernels.cuh
├── tools/
│   ├── convert_hf.py
│   ├── dump_reference.py
│   └── benchmark.cpp
├── tests/
│   ├── cpu_reference/
│   ├── test_kernels.cu
│   ├── test_qwen3.cpp
│   └── test_scheduler.cpp
└── docs/
```

原则是：model layer 不直接调用裸 CUDA API；所有 device allocation、stream、event、workspace 和 kernel dispatch 经过 runtime/backend 层。这样后续才能替换 kernel、接 CUDA Graph 或加入 CPU reference。

## 6. 分阶段实施路线

状态标记：`[x]` 已完成，`[ ]` 未完成，`[-]` 进行中或部分完成。

### [x] Phase 0：环境、基线和可观测性

**目标**：先建立可信的参考结果和性能测量方式。

工作项：

- 固定 CUDA、驱动、编译器、GPU 型号和 commit；加入 `--version` 与构建信息。
- 使用 Transformers/PyTorch 加载 Qwen3-0.6B，保存 tokenizer、chat template、短输入的 logits、hidden states、greedy token 序列。
- 建立 `golden` 测试：输入 token ids、position ids、每层输出或最终 logits 的 FP32/BF16 参考值。
- 建立 CUDA 错误宏、NVTX range、kernel time、显存使用和 request trace。
- 先测 baseline：PyTorch eager、PyTorch SDPA/FlashAttention、llama.cpp（若格式支持）、自己的 naive runtime。

**验收**：同一 prompt 可复现；每个阶段都有 tokens/s、TTFT、显存、kernel 时间；错误能定位到 layer/op，而不是只报“结果不一致”。

### [x] Phase 1：CPU 算子 reference 与最小张量系统

本次已完成 Tensor 抽象和 CPU reference 算子：`CMakeLists.txt`、`include/llm/dtype.h`、`include/llm/device.h`、`include/llm/tensor.h`、`include/llm/ops.h`、`src/tensor.cpp`、`src/ops_cpu.cpp`、`tests/test_tensor.cpp`、`tests/test_ops_cpu.cpp`。已实现 embedding lookup、linear、RMSNorm、RoPE、causal mask、stable softmax、GQA causal attention、SwiGLU、add 和 tied lm_head。单层 Qwen3 forward 与 Transformers 逐层 golden 对比尚未完成。

已新增 Qwen3-0.6B layer 0 CPU F32 prefill 实现（`include/llm/qwen3_layer.h`、`src/qwen3_layer.cpp`）、离线 fixture 导出入口（`tools/export_qwen3_layer0_reference.py`、`tools/run_phase1_layer0.sh`）及逐节点 loader/对比测试。FP16 子正规数、Qwen3 RoPE、稳定 GQA 与异常路径测试已补齐；已完成与 Transformers 真实 causal prefill 的逐节点 golden 对齐。fixture 是离线、CPU F32、layer-0 golden，仅用于 Phase 1，不等价于 Phase 1.5 的通用 safetensors 权重加载。

**目标**：建立可逐算子、逐层比较的 reference，不把 CPU 全模型性能作为前期门槛。

实现：

- `Tensor`：dtype、shape、stride、device、contiguous/view、byte size。
- BF16/FP16/FP32 转换与安全的数值比较。
- CPU reference：embedding、RMSNorm、linear、RoPE、GQA attention、SwiGLU、residual、LM head。
- 明确 layout：初期统一 row-major，避免在正确性阶段混入过多 transpose。
- 已补齐 IEEE-754 FP16 子正规数、Qwen3 rotate_half 语义的 RoPE、稳定 softmax GQA，以及对应边界/异常 reference 测试。
- 明确 residual 顺序和 Qwen3 的 Q/K RMSNorm；不能按普通 Llama 结构想当然。

**验收**：CPU reference 与 Transformers 在算子、单层中间张量和最终 logits 接近；greedy 输出一致或仅在允许的浮点误差下偏离。CPU 全模型仅作为 debug 工具，不作为后续 CUDA 开发的阻塞条件。

### [x] Phase 1.5：Qwen3 配置、tokenizer 与权重格式适配

**目标**：在写 CUDA model runner 之前，锁定 Qwen3 的结构、权重名称、数据布局和输入协议。

实现：

- 解析 `config.json`，校验 hidden size、层数、Q/KV 头数、head_dim、MLP 宽度、RoPE theta、词表和 tied embedding。
- 读取 Hugging Face safetensors metadata，完成权重名称映射、shape 校验、dtype 转换和 mmap/offset 记录。
- 明确 row-major/column-major、QKV 是否融合、linear 权重是否转置，以及 C++ 侧的权重生命周期。
- 已完成 `cuda_llm_model_package_v1`：离线 CPU F32 导出、64-byte 对齐 `weights.bin`、FNV-1a64 校验、tied `lm_head` alias、tokenizer artifacts/fixtures、C++ 按需 loader，以及从 package 加载 layer 0 权重到 golden 的验证。C++ runtime 当前只接受 token IDs，不包含 C++ BPE tokenizer；模型包不等价于通用 safetensors loader。
- 全部 28 层、每层 11 个逻辑权重均已做 name/shape 契约验证；C++ loader 会验证并实际使用 package config 的 RMSNorm epsilon 与 RoPE theta。
- 保存 tokenizer、chat template、generation config；区分普通文本、chat、thinking/non-thinking 输入。
- 生成一份可独立加载的自有权重格式，并对转换前后的每个 tensor 做 checksum 或抽样数值校验。

Qwen3-0.6B 的适配清单：

```text
[ ] hidden_size = 1024, num_layers = 28
[ ] num_attention_heads = 16, num_key_value_heads = 8, head_dim = 128
[ ] intermediate_size = 3072, vocab_size = 151936
[ ] Q/K RMSNorm、RoPE(theta=1_000_000)、SwiGLU
[ ] tied embedding/lm_head
[ ] causal mask、position id、GQA head mapping
[ ] tokenizer/chat template/thinking 配置
```

**验收**：所有 tensor 名称和 shape 映射完整；C++ 能加载自有格式；embedding/lm_head 共享关系正确；输入 token 与 Transformers 完全一致。
### [-] Phase 2：权重加载与 Qwen3 单请求 CUDA 前向

- Phase 2.1：CUDA RAII Tensor、CPU↔GPU copy、F32/F16 device storage 与 cuBLAS GEMM 基线已实现。
- Phase 2.2：RMSNorm、add、SwiGLU、Qwen3 RoPE、stable softmax 和单 batch causal GQA correctness baseline 已实现并通过 RTX 2080 Ti SM 7.5 对照测试。
- Phase 2.3：已完成 Qwen3-0.6B Layer-0 CUDA F32 causal prefill，复用 `cuda_llm_model_package_v1` 加载 layer-0 权重，并与 `artifacts/phase1/qwen3_layer0` 的 Transformers golden 完成 17 个中间节点逐项对齐。trace 中间 Tensor 全部保持 CUDA/F32/contiguous，仅在比较时下载到 CPU；默认节点阈值为 `max_abs_error <= 1e-4`、`cosine_similarity >= 0.99999`，`k_norm_output` 与 `k_rope` 使用 Phase 1 已确认数值预算 `max_abs_error <= 5e-4`。已覆盖 CPU/F16 hidden state、shape、position length、seq_len、权重 shape、epsilon/theta 等拒绝路径。
- Phase 2.4：已完成 Qwen3-0.6B Layer-0 CUDA FP16 causal prefill trace，17 个中间节点与现有 `artifacts/phase1/qwen3_layer0` Transformers F32 golden 对齐。trace Tensor 均保持 CUDA/F16/contiguous，仅比较时下载到 CPU；FP16 GEMM 使用 FP16 I/O、FP32 accumulate，RMSNorm、softmax、GQA 累加保持 FP32。验收采用逐元素 `allowed = max(5e-3, rtol × abs(expected))`，默认 `rtol=1e-3`，仅 `k_norm_output`、`k_rope` 使用 `rtol=1e-2`，所有节点 `cosine_similarity >= 0.999`。已覆盖 CPU/F32/mixed-dtype 输入、shape、position length、seq_len、权重 shape、epsilon/theta，以及 F32 trace 拒绝 F16 的错误路径。
- Phase 2.5a：已完成 CUDA embedding lookup、final RMSNorm 接线与 tied LM head 基线，并完成 F32/F16 对照。final 路径为 `final_hidden = cuda_rms_norm(last_hidden, final_norm_weight, rms_norm_eps)` 后接 `logits = cuda_lm_head(final_hidden, tied_token_embedding)`；embedding 与 LM head 复用同一个 tied embedding Tensor。Phase 2 总状态保持 `[-]`。
- Phase 2.5b：已完成 28 层单请求 CUDA FP16 prefill hidden-state runner，GPU 常驻全部 FP16 权重，token embedding 只存一份并保留 tied embedding 语义；runner 依次执行 28 个 decoder layer，释放每层非 `layer_output` 的中间 Tensor，并在末尾复用 `cuda_rms_norm` 完成 final RMSNorm。Layer-0 runner 对照、28 层 hidden-state smoke 和重复 final hidden 一致性测试均通过。Phase 2 总状态保持 `[-]`。
- Phase 2.5c：已完成两个 full-prefill F32 Transformers golden case（`case_seq4`、`case_seq8`）及 CUDA FP16 embedding/final hidden/logits 对齐；final hidden 采用逐元素 `allowed = max(1e-1, 2e-2 × abs(expected))`、`cosine_similarity >= 0.995`，logits 采用 `allowed = max(1e-1, 2e-2 × abs(expected))`、`cosine_similarity >= 0.99`，并通过最后 token greedy 与 top-5 一致性验证。tied LM head 已接入 runner，logits 由同一份 token embedding 权重计算。Phase 2 总状态保持 `[-]`。
- Phase 2.6：已完成 batch=1 contiguous CUDA/F16 KV Cache（capacity 可配置，默认 32），支持 1..32 token prefill 写入 K/RoPE 与 V，并完成单 token CUDA decode attention；decode 通过 cache-vs-full-prefill logits、greedy 和 top-5 等价性验证，cache reset、容量、shape、dtype、顺序和越界错误路径均已覆盖。Phase 2 总状态保持 `[-]`。
- Phase 2.7：已完成 GPU last-row argmax（F32/F16、多 block、相同最大值时选择最小 token ID）、batch=1 greedy generation、EOS/max-new/cache-capacity 停止语义和严格 token-ID CLI。generation 复用现有 CUDA prefill/decode、KV Cache 与 tied LM head，不执行 CPU logits argmax；argmax、生成循环、EOS、容量和 CLI smoke 测试均已通过。Phase 2 总状态保持 `[-]`。
- Phase 2.8：已完成 vendored 原生 C++ Qwen3 tokenizer（`tokenizers-cpp` 固定 commit `4bb753377680e249345b54c6b10e6d0674c8af03d`），离线读取 Hugging Face `tokenizer.json`，无 Python runtime 依赖；EOS 从同目录 `tokenizer_config.json` 的 `eos_token` 实际编码得到。已与 Transformers reference 对齐 ASCII、中文、中英文混合空格/标点及 special token/EOS decode；对象可重复复用且不在 encode/decode 时重复加载 JSON。已完成 `cuda_llm_chat` plain-text CLI，复用现有 GPU argmax、KV Cache 与 `generate_greedy`；chat CLI 与 ID CLI 的 prompt、generated IDs、停止原因、cache 长度及 generated text 一致性测试均通过。Phase 2 总状态保持 `[-]`。
- Phase 2.9a：已完成 GPU sampling 与单请求停止控制基线。新增可复用 RAII `CudaSampler`，在预分配 workspace 上对 CUDA F32/F16 logits 执行 temperature、稳定 softmax、稳定降序排序、top-k、包含 crossing token 的最小 top-p 前缀和重归一化；temperature=0 精确复用 GPU argmax。采样使用 `seed + draw_offset` 的无全局状态 counter-based RNG，同 seed/offset/input 可重复，平分时优先较小 token ID；生成复用原有 prefill/decode、KV Cache、EOS/max-new/cache-capacity 语义，并同时接入 `cuda_llm_generate` 与 `cuda_llm_chat` 的可选采样参数。已通过 F32/F16、top-k/top-p 边界、大词表、确定性、错误路径、采样生成和 chat/ID CLI 对齐测试；Phase 2 总状态保持 `[-]`。
- Phase 2.9b：已完成固定 Qwen3 原生 chat template、多轮 system/user/assistant messages 输入与文本 CLI 接线。`Qwen3ChatTemplate` 使用 vendored nlohmann/json v3.11.2 校验 messages，并按本地 Qwen3 template 逐字节渲染 special token、role 标记、换行和 assistant generation prompt；离线 Transformers reference 的 rendered text 与 token IDs 完全对齐。`cuda_llm_chat --messages-file` 复用现有 tokenizer、GPU generation、sampling、KV Cache 和 EOS 语义，messages CLI 与 ID CLI 的 greedy/sampling 结果完全一致。已通过 20 项 CTest、template 错误路径和 messages CLI 重复性测试；Phase 2 总状态保持 `[-]`。当前限制是仅固定 Qwen3 文本 template，不支持通用 Jinja 解释器、tools/function calling、多模态 content、tool/developer role 或 thinking 分支。
- 尚未完成：streaming、FlashAttention/paged attention、continuous batching、CUDA Graph、量化与性能优化。

**目标**：GPU 上实现完整 prefill 和 decode，先以 RTX 2080 Ti 可执行的 FP16 路径为主，不追求最优 kernel。

实现顺序：

1. 读取 HF safetensors 的 metadata，转换为自有格式；第一版可在 Python 中完成转换，C++ 只负责读取。
2. 共享 embedding/lm_head；权重按 layer 连续组织，减少小块 allocation。
3. 先用 cuBLASLt 做 FP16 GEMM，必要时 FP32 累加；BF16 作为可选 dtype，不作为 Compute Capability 7.5 的主验收路径。
4. 写 CUDA kernels：RMSNorm、RoPE、SwiGLU elementwise、residual、temperature/logits mask。
5. attention 先实现 naive contiguous KV；prefill 与 decode 分开处理。
6. 使用固定 workspace 和 CUDA stream；避免每个 token `cudaMalloc/cudaFree`。

建议的 forward 伪代码：

```cpp
x = embedding(input_ids);
for (layer : layers) {
    x = rmsnorm(x, layer.input_norm);
    q = matmul(x, layer.q_proj);
    k = matmul(x, layer.k_proj);
    v = matmul(x, layer.v_proj);
    q = q_norm_and_rope(q, position_ids);
    k = k_norm_and_rope(k, position_ids);
    append_kv_cache(layer, k, v);
    x = x + attention(q, kv_cache[layer]);
    h = x;
    x = rmsnorm(x, layer.post_attention_norm);
    x = h + swiglu_mlp(x, layer.gate_proj, layer.up_proj, layer.down_proj);
}
logits = lm_head(rmsnorm(x, final_norm));
```

**验收**：单请求 1/16/128/1024 token 的 prefill 和 decode 都能运行；与 PyTorch 对比逐层 logits；无 device memory leak；显存使用可解释；FP16 greedy 结果稳定。

### [-] Phase 3：请求调度与批处理

- Phase 3.1a：已完成单线程 Qwen3 request scheduler simulation。scheduler 只持有并复用一个已加载的 `Qwen3CudaModel`，按 Waiting FIFO 执行 prefill，按 Decode round-robin 执行单 token decode，并将请求推进到 Finished。每个请求拥有独立 contiguous KV Cache；请求级 `SamplingConfig.seed + draw_offset` 保证采样结果不受提交顺序和 scheduler interleaving 影响。已完成 greedy/sampled 与串行 generation baseline 对齐，以及 EOS、max_new_tokens、cache_capacity、请求状态和错误路径测试。
- Phase 3.1b：已完成 equal-length static batch prefill。`Qwen3CudaModel::prefill_logits_batch_with_caches` 对 batch=1/2/4 使用单次 batched CUDA forward，hidden/logits 采用 `[B,S,hidden]` / `[B,S,vocab]`，Q/K/V 采用 `[B,S,16,128]` / `[B,S,8,128]`；每个请求仍写入独立 contiguous KV Cache。新增 batched RoPE、causal GQA、batch-aware sampler 与 decode 对照，覆盖 batch size 1/2/4、batch isolation、串行 logits/top-5/greedy 对齐、cache 长度和拒绝路径。scheduler 提供 opt-in `prefill_waiting_static_batch(max_batch_size)`：以最早 waiting 请求的 prompt length 为锚点，只选择相同长度，按 FIFO 保留未选请求，并只提交受支持的 batch=1/2/4；默认 `step()` 仍保持 Phase 3.1a 单请求路径。
- Phase 3.1c：已完成 B=1/2/4 equal-length static batch decode。`decode_logits_batch_with_caches` 使用 `[B,1024]` hidden、`[B,16,128]` Q、`[B,8,128]` K/V 和 `[B,151936]` logits；batched decode attention 通过可复用的 GPU K/V pointer array 直接访问每个 request 的独立 contiguous KV Cache，不复制完整 cache。每个 layer 在 cache 的 `L` 位置追加一 token K/V，28 层完成后统一 commit 到 `L+1`；scheduler 提供 opt-in `decode_active_static_batch(max_batch_size)`，按最早 Decode request 的 cache length 分组，保留未选请求 FIFO，并独立处理 greedy/sampled、EOS、max_new_tokens 与 cache_capacity。新增模型级 logits/greedy/top-5/sampling 对齐、batch isolation、cache transaction 拒绝路径和 scheduler static decode 回归。
- 尚未完成：variable-length padding/attention mask、Paged KV Cache/block manager、continuous batching、异步/streaming server、CUDA Graph。

### [-] Phase 3.2：专门优化 decode kernel 和内存复用

- Phase 3.2a：进行中。新增独立的 `qwen3_cuda_benchmark` CUDA Event benchmark，默认覆盖当前 prefill API 支持的 context `{4,16}`，分别观测 batched prefill、steady-state decode 和 GPU sampling，并输出确定性的 mean/p50/p99、tokens/s、device memory 辅助信息。Tensor CUDA 分配唯一路径新增线程安全的 allocation/free/live/peak 统计接口，allocation snapshot 按 iteration 聚合，并有 helper 与 `test_cuda_allocation_stats` CTest 覆盖。128+ 长上下文 sweep 仍等待 long-prefill；workspace 复用、fused kernel 与 attention 优化尚未完成。
- Phase 3.2b：已完成 Qwen3 FP16 decode workspace 复用。`Qwen3CudaModel` 持有单一 batch-4 `DecodeWorkspace`，hidden/Q/K/V/attention/residual/RMSNorm/MLP/final-norm 与 token/position 临时 buffer 在首次 decode 前 resident；B=1/2/4 使用共享 storage 前缀 view，KV Cache 仍独立存储。decode 层通过 `*_out` CUDA 接口覆写预分配 buffer，steady-state 只保留返回 logits 的必要 Tensor 分配。新增 `test_qwen3_decode_workspace` 覆盖数值、greedy/sampling、cache length、B=1/2/4、拒绝路径和 allocation health check。

**目标**：让单请求 decode 成为一个有意义的 CUDA runtime，而不是“cuBLAS 调用串”。

优化顺序：

- fused RMSNorm + quantization/activation（有收益时再融合）。
- fused QKV projection 或至少减少中间 tensor。
- decode attention 使用单 token query 的专用 kernel。
- 通过 persistent workspace、预分配 buffer、CUDA event 复用降低 launch/allocation 开销。
- cuBLASLt algorithm selection、workspace tuning；必要时引入 CUTLASS。
- CUDA Graph 暂不在本阶段作为主线；待 scheduler、paged KV 和 batch metadata 稳定后再做多 shape graph pool。

此阶段不要过早手写所有 GEMM。GEMM 通常应先交给 cuBLASLt/CUTLASS；自己重点写 layout、fusion、attention、采样和 cache 相关 kernel。

### [x] Phase 4：静态 batch 与 contiguous KV

**目标**：先把多个同形状请求放入一个规整 batch，建立 batch metadata 和 batch=1 等价性。

核心对象：

- `Request`：prompt tokens、生成参数、状态、停止条件、输出队列。
- `Sequence`：当前长度、block table、最后一个 token、finished 状态。
- `Scheduler`：waiting/running/finished 队列，每一步决定本轮要执行的 token。
- `ModelRunner`：prefill batch、decode batch、batch metadata、CUDA stream。

调度策略：

- 先实现简单 FIFO + token budget。
- prefill 可按长度分组；decode 尽量把不同请求的一个 token 合并为 batch。
- 固定 batch size、prompt length 和 decode step，先验证静态 batch 的 prefill/decode。
- 先限制最大 batch、最大总 token 和最大上下文，避免调度器复杂到无法调试。

**验收**：静态 batch 与 batch=1 结果一致；batch size 增大时吞吐和显存变化可解释；没有把最长序列错误地广播到所有请求。

#### [x] Phase 4.1：right-padded variable-length static batch prefill

已完成：`PaddedPrefillBatch` host metadata builder、Qwen3 FP16 right-padded
prefill、逐请求 valid-length KV 写入/commit，以及独立的 last-valid logits
选择。`B=1/2/4`、`S_max<=32` 支持 `[B,S_max,1024]` hidden、`[B,S_max,16,128]`
Q、`[B,S_max,8,128]` K/V 和 `[B,S_max,151936]` logits；padding 使用 token 0，
不改变输入。每个 cache 只提交自身真实 prompt 长度，异常路径通过
`abort_prefill()` 保持事务不前进。新增 `test_qwen3_padded_prefill` 覆盖
`[4,6]` 与 `[3,4,6,8]`、serial logits/top-5/greedy/sampling 对齐、短 cache
容量、隔离、拒绝路径和 CUDA health；`test_qwen3_scheduler_padded_prefill`
覆盖 FIFO opt-in 调度及“三个可选请求只提交前两个、第三个保持 Waiting”。

#### [x] Phase 4.2：显式 per-sequence valid-length attention mask

已完成：padded static prefill 在首层前一次性创建 CUDA/I32 `[B]`
`valid_lengths` metadata，并由 28 层复用。masked GQA 显式约束 query/key
有效区与 causal 可见性；无效 query 输出为零，padding K/V 仍不写入 KV
Cache。新增 `test_gqa_valid_lengths_cuda` 覆盖 F32/F16 的 row isolation、
valid prefix、invalid query zero output 与 metadata 拒绝路径；模型级测试
输出 `padded prefill attention_mask=valid_lengths_cuda passed`，并继续验证
serial logits、top-5、greedy、sampling 与每条 cache 的 valid length。
模型热路径进一步拆分为无 D2H 的 checked kernel 入口；valid-length metadata
一次 H2D 上传后复用 28 层，测试计数确认 D2H 为 0。

当前仍限定右侧 padding、`S_max<=32` 和共享 `0..S_max-1` positions；left
padding、任意 attention mask 和长上下文 prefill 尚未完成。

#### [x] Phase 4.3：variable-length decode batch

已完成：新增 `decode_logits_variable_length_batch_with_caches`，支持 `B=1/2/4`
的 mixed-length static decode。每行使用独立的 CUDA/I32 RoPE position、KV
attention visible length、contiguous KV append index 和 decode commit length；
仍复用每请求独立 KV cache 与 GPU pointer array，不复制完整 KV Cache。模型热
路径的 per-row metadata 一次 H2D 上传，28 层复用且无 D2H validation copy；
`test_gqa_variable_lengths_cuda`、`test_qwen3_variable_length_decode` 和
`test_qwen3_scheduler_variable_length_decode` 覆盖 `[4,6]`、`[3,4,6,8]`、
serial logits/top-5/greedy/sampling、`L→L+1` commit、隔离、拒绝路径和
FIFO 前缀调度。

仍未完成：continuous batching、Paged KV、block manager 和长上下文。

#### [x] Phase 4.4：静态 token-budget batching 与调度可观测性

已完成 opt-in FIFO token-budget prefill/decode 调度。Prefill 按候选前缀的
`B*S_max` padded token 成本、有效 token 数和 padding ratio 选择合法的
`B=1/2/4` batch；decode 使用 Phase 4.3 的 mixed-length 路径并同时限制
batch 数、decode token 数和最大上下文。调度器保留原有默认 `step()`、等长
static API、padded API 和 variable-length API 的选择行为不变。

每轮输出不可变 batch metrics，包括 selected request IDs、队列状态、token
成本、padding ratio、skip-budget 计数和 `steady_clock` elapsed time，并支持
稳定 CSV 导出。每个请求记录提交时间、首次生成 token 的逻辑 round 和 TTFT，
且首次 token 只记录一次。该实现仍是静态 batch，不支持运行中动态加入/退出。

Phase 5 仍未完成：continuous batching、动态请求加入/退出、取消、超时和
backpressure。

### [x] Phase 5：continuous batching 与请求控制

**目标**：实现教学版 in-flight batching，允许请求在 decode 过程中加入和退出，但暂时仍使用每请求 contiguous KV。

实现：

- `Request`、`Sequence`、`Scheduler`、`ModelRunner` 分层，scheduler 只输出纯 host batch metadata。
- 使用 FIFO + token budget；每轮 decode 允许新请求进入，完成请求立即退出。
- 每条 sequence 维护独立 KV 指针和长度；先接受内存浪费，重点验证调度正确性。
- Phase 5.1/5.2 已完成 continuous static batching、FIFO/token budget、prefill/decode
  fairness 和 scheduler/request metrics。
- Phase 5.3 已完成请求取消、queue/request timeout、注入单调时钟、waiting/active
  admission 限制和基于实际 contiguous KV resident bytes 的资源背压。

**验收**：混合短/长请求时请求不会互相阻塞；单请求结果与 batch=1 一致；请求完成、
取消和超时后 KV 正确释放；P50/P99、active sequences、TTFT 和 token throughput
可测。Phase 5 仍是同步 scheduler，不支持真正异步取消。

#### [x] Phase 5.1/5.2：continuous batching、token budget 与 observability

已完成 Phase 5.1/5.2 的 FIFO continuous prefill/decode、静态 token budget、
fairness、max-active admission 和 batch/request CSV metrics。

#### [x] Phase 5.3：请求取消、超时与资源背压

已完成 `cancel_request`、batch-boundary active cancellation、queue/request
timeout、可注入 monotonic clock、`max_waiting_requests` 和
`max_total_kv_cache_bytes`。KV budget 使用 Qwen3 `Qwen3KvCache::resident_bytes()`
的实际布局计算，预算不足的请求保持 FIFO Waiting，完成/取消/超时后归还。
`test_qwen3_scheduler_controls` 覆盖取消优先级、超时、无 CUDA admission reject、
KV budget defer/reclaim 和 CSV termination observability。

尚未完成：Paged KV Cache、prefix cache、CUDA Graph、HTTP/streaming server、
真正异步取消、swap/offload 和服务化 backpressure。

### [ ] Phase 6：Paged KV Cache / block manager

**目标**：实现 vLLM 的关键内存思想：逻辑序列连续，物理 KV block 可不连续。

设计：

- 设 block size 为 16 或 32 tokens；GPU pool 的最终 row-major KV layout 为
  `[num_layers, total_blocks, 2, block_size, num_kv_heads, head_dim]`，保证
  单 token 的 `[num_kv_heads, head_dim]` 连续。
- CPU 维护 free block pool、sequence block table、refcount。
- 新 token 到达时只为当前 block 追加；block 满了再申请新 block。
- attention kernel 读取 block table，把逻辑 token 映射到物理 block。
- 先做 copy-on-write/refcount 的数据结构，再实现 prefix sharing；不要一开始做复杂 radix tree。

Paged attention 的逻辑索引为：

```text
logical_token = past_len + local_token
block_id      = logical_token / block_size
offset        = logical_token % block_size
physical      = block_table[request][block_id]
K = kv[physical][offset]
V = kv[physical][offset]
```

**验收**：同样总 token 数下，碎片化请求不会因为最长序列而分配大块连续 cache；随机释放/复用后结果仍正确；block table 与实际访问一致。

#### [x] Phase 6.1：CPU Block Manager 与 Sequence Block Table

已完成纯 CPU block 元数据层：`BlockManager` 提供 free-list、稳定
`BlockId`、allocate/retain/release 和引用计数；`SequenceBlockTable` 提供逻辑
token 到 `{block_id, offset}` 的映射、跨 block 原子 append、max token 校验和
幂等释放。多 block 分配失败时完整回滚，析构自动归还 block。

本阶段不改动 CUDA attention、`Qwen3KvCache`、scheduler 或模型公式。尚未完成
共享 GPU block pool、Paged KV 写入、paged attention、scheduler 接入、prefix
sharing 和 copy-on-write。

#### [x] Phase 6.2：共享 GPU Paged KV Block Pool

已完成共享 CUDA FP16 storage 与 Phase 6.1 CPU block metadata 的绑定。Pool 使用
一次性 RAII Tensor allocation，固定 layout 为
`[num_layers,total_blocks,2,block_size,num_kv_heads,head_dim]`；当前 Qwen3
配置的完整 K+V physical block 为 `1,835,008` bytes（单层单 K/V
`kv_plane_bytes` 为 `32,768` bytes；跨 28 层的单 K/V plane 总计
`917,504` bytes）。`PagedSequenceKvCache` 通过逻辑
token→`{BlockId,offset}` 映射访问 K/V，并提供受检查的 host FP16 copy 验证接口。

已验证跨 sequence block 隔离、K/V/layer/token 写读 bit-exact、release/reuse、
pool exhaustion 原子回滚和 pool 构造后的 steady-state 零额外 Tensor
cudaMalloc/cudaFree。尚未接入 Q/K/V 写入、Paged Attention、scheduler、prefix
sharing/COW 或 GPU block table metadata。

#### [x] Phase 6.3a：单 sequence decode Paged GQA Attention correctness baseline

已完成独立的 `cuda_paged_gqa_attention_decode` CUDA F16 单 token decode
correctness kernel，以及 `PagedSequenceKvCache::make_device_block_table_i32()`
GPU block table 上传接口。kernel 按 Phase 6.2 layout 做 logical token→physical
block lookup，使用 Q/K/V 的 GQA head mapping 和 FP32 stable softmax/累加。
已通过 CPU reference 与现有 contiguous GQA 对齐、跨 block、非连续 physical
block table、GQA mapping、tail masking 及错误路径测试。

本阶段不接入 Qwen3 model decode 或 scheduler；尚未完成 Q/K/V projection 直接
写入 pool、batch/prefill paged attention、scheduler/block-manager 接入、prefix
sharing/COW 和性能优化。

#### [x] Phase 6.3b-1：Qwen3PagedKvCache transaction 与逐层 K/V D2D 验证

已完成真实 Qwen3 规格的 `Qwen3PagedKvCache`：支持单 token
begin/append-layer-KV/commit/abort 状态机，28 层 K/V 全部写入后才提交可见
length；失败或 abort 会恢复 block table 并归还新增 physical block。逐层 K/V
写入使用 checked CUDA D2D copy，seed helper 可从现有 contiguous
`Qwen3KvCache` 做纯 D2D 迁移并逐 bit 对齐验证。

本阶段仍未接入模型 `decode_logits_paged`、batch/scheduler、prefix sharing/COW
或性能优化。

#### [x] Phase 6.3b-2：Qwen3 单请求 Paged Decode

已完成 `Qwen3CudaModel::decode_logits_paged`：仅支持 batch=1、Qwen3-0.6B
FP16、`max_seq_len<=32`，按现有 28 层 decode 公式执行 projection、Q/K
RMSNorm、RoPE、paged GQA causal attention、residual 和 MLP。每个 decode
transaction 只上传一次 device block table；所有层 K/V 写入成功后才 commit，
异常会 abort 并回滚新增 physical block。

`tests/test_qwen3_paged_decode.cpp` 已验证 context 4/15/16/17/31、跨 block 和
非连续 physical block table、K/V bit-exact、contiguous logits/top-5/固定 seed
sampling 对齐、多步 decode、pool exhaustion 与错误路径。Phase 6 仍未完成
batch/scheduler paged decode、Paged continuous batching、prefix sharing/COW、
long context 和性能优化。

#### [x] Phase 6.4a：单请求 Paged Scheduler 与 BlockManager Admission

已完成独立、opt-in 的 `Qwen3PagedRequestScheduler`：仅支持 B=1，每次 step
最多执行一个请求的一次 contiguous prefill→paged seed 或一次
`decode_logits_paged`。admission 按 `ceil(max_seq_len / block_size)` 做保守
block 预算判断，实际 physical block 仍由 paged cache lazy 分配；pool 不足时
严格 FIFO defer，不触发 prefill/CUDA。完成、取消和 fake-clock timeout 均会
回收 paged blocks，并记录 action、队列、pool free/used 和 resident bytes。

当前仍未完成 paged prefill、B=2/4 paged decode、paged batch attention、替换原
scheduler、GPU block-table workspace、prefix sharing/COW 和性能优化。

#### [x] Phase 6.4b-1：B=2/4 Paged Decode correctness batch

已完成独立的 `decode_logits_paged_batch`：支持 B=2/4 的不同 cache length，所有
行在同一调用内完成 K/Q/V/O/MLP 计算，paged attention 仍按行读取各自的
physical block table。模型在进入 28 层计算前一次性上传每行 block table 和
position metadata，并在所有行、所有层成功后统一 commit；任一层/行异常会对
所有已开始的 cache 统一 abort，保持 all-or-nothing 事务语义。

`test_qwen3_paged_batched_decode` 已覆盖 `[4,6]`、`[3,4,6,8]`、跨 block 多步
decode、layer 7/27 的 K/V bit-exact 对齐、top-5/固定 seed sampling、pool/table
隔离、故障注入回滚和拒绝路径。当前仍未完成真正 batched paged attention kernel、
paged scheduler 的 B=2/4 接入、paged prefill、GPU block-table workspace、
prefix sharing/COW、long-context 和性能优化。

#### [x] Phase 6.4b-2：Paged Decode GPU Metadata Workspace

已将 paged batch decode 的 positions 与 per-row GPU block table metadata 收归
`Qwen3CudaModel` 独占的 RAII workspace。当前 Qwen3 block size=16 时复用
`positions[4]` 与 `block_tables[4,2]`，resident bytes 为 48；每次 decode 只
向 caller-provided row view 做 checked H2D copy，28 层复用同一组 metadata，
不再调用会分配 Tensor 的 `make_device_block_table_i32()` 热路径。

`test_qwen3_paged_batched_decode` 已覆盖 B=1/2/4、mixed length、跨 block、多步
数值/top-5/固定 seed sampling 对齐，并报告 warmup 后 metadata allocation=0；
模型快照 `last_paged_decode_metadata_upload_stats()` 只覆盖 metadata view 与 H2D
copy 区间，审计确认 B=1/2/4 的 metadata malloc/free 与 allocated/freed bytes
增量均为 0，positions 上传 1 次、block-table 上传 B 次。整次 decode 的层内临时
Tensor 与返回 logits 分配仍单独保留。当前仍未完成真正
batched paged attention kernel、paged scheduler B=2/4、paged prefill、prefix
sharing/COW、long-context 及其它层内性能优化。

#### [x] Phase 6.4c：Paged Scheduler B=2/4 Decode Batching

已将独立的 `Qwen3PagedRequestScheduler` 扩展为可配置的 B=1/2/4 paged decode
batch。默认 `max_decode_batch_size=1` 保持既有行为；配置为 2 或 4 时，scheduler
采用 correctness-first 的 bounded batch-fill policy：当 waiting 非空且 active 未
达到目标 batch 时，每个 step 只做一次单请求 admission prefill；最多连续执行
`max_decode_batch_size-active_before` 次后，再通过一次
`decode_logits_paged_batch` 执行同批 decode。每个 row 保持独立 paged KV cache、
cache length、采样状态和完成原因；B=3 因模型 API 限制被拆为 FIFO 的 B=2 后 B=1。

批次指标追加 `decode_batch_size`、`selected_request_ids` 和
`decode_batch_error`，同时保留 `selected_request_id` 兼容字段。模型 batch 失败时
所有 cache 事务统一 abort，scheduler 将选中的 request IDs 按原顺序恢复到 active
队首，不推进 token/draw offset，并记录 `decode_error`；成功、EOS、max-token、
取消和 timeout 均正确回收 blocks。新增测试覆盖 B=2/B=4 真实结果对齐、FIFO 选择、
B=2/B=1 收缩、bounded fill 上界、批次异常前后状态快照、动态加入、batch 边界取消、
active timeout 和 CUDA health check。该策略不是最终 token-budget prefill/decode
fairness policy，复杂竞争仍留给 Phase 7。

本阶段仍未完成真正融合的 batched paged-attention kernel、paged prefill、paged
scheduler 的完整生产化接入、prefix sharing/COW、long-context、CUDA Graph 和
其它性能优化。

#### [x] Phase 6.5a：真正 Batched Paged GQA Decode Attention

已将 Qwen3 FP16 paged decode layer 的逐 row single attention 路径替换为一次覆盖
整个 B=1/2/4 batch 的 `cuda_paged_gqa_attention_decode_batch_out` kernel launch。
kernel 使用 `[B,q_heads,head_dim]` Q、`[B,max_blocks]` GPU block table 和 `[B]`
position metadata，按 row 独立计算 `positions[row]+1` KV length、logical→physical
block 映射和 GQA head mapping；FP32 dot/稳定 softmax/V 累加后写 FP16。B=1 的
`decode_logits_paged` 也复用该路径。

Qwen3 layer 现在先完成所有 row 的 K/V append，再统一调用一次 batch attention；
移除了每 row 的 Q/K/V Tensor、D2D staging copy、single attention 调用和输出拷贝。
`_out` operator 不创建 CUDA Tensor，测试已验证 B=1/2/4 operator allocation 为
0；Qwen3 batch 测试进一步确认每个 decode 调用 28 层各一次 batch attention launch，
并保持 logits、K/V bit-exact、top-5、sampling、rollback 与拒绝路径正确。

本阶段仍未完成 paged prefill、FlashAttention/warp-tiled 优化、fused QKV、CUDA
Graph、prefix sharing/COW、long-context 和 Phase 7 token-budget fairness。

#### [x] Phase 6.5b：Paged Decode Workspace 与层内临时显存消除

已将 Qwen3 FP16 paged decode B=1/2/4 的层内中间结果接入模型独占的
`DecodeWorkspace`。hidden ping-pong、input RMSNorm、Q/K/V projection、RoPE、
attention、residual、post-attention RMSNorm、gate/up/SwiGLU、down 和 final
RMSNorm 均使用预分配 FP16 buffer 的前缀 view；每层通过 `*_out` 接口写入 caller-
provided output，K/V 直接以 non-owning row view 追加到 paged cache。模型 paged
batch decode 保持 28 层一次 batch attention launch、all-or-nothing transaction 和
原有数值/采样语义。

`test_qwen3_paged_decode_workspace` 与 batched decode 回归验证 B=1/2/4 的 serial
对齐、metadata workspace 常驻 48 bytes、warmup 后 steady-state 项目 Tensor
allocation 不超过 2 次（仅允许返回 logits 等必要分配），并保留 metadata
allocation=0 的独立统计。workspace 使用 Tensor RAII；同一模型 paged decode
不支持并发调用。

本阶段仍未完成 paged prefill、paged scheduler B=2/4 的生产化迁移、prefix
sharing/COW、long-context、FlashAttention/warp-tiled 融合、CUDA Graph 和其它
性能优化。

#### [x] Phase 6.6a：单请求 Direct Paged Prefill

已完成 Qwen3 FP16 B=1 direct paged prefill。`Qwen3CudaModel::prefill_logits_paged`
直接使用已有 prefill layer 公式，将每层 RoPE 后 K 与 V projection 通过
`Qwen3PagedKvCache` 的显式 prefill transaction 写入 GPU physical block pool，
不构造 contiguous `Qwen3KvCache`、不执行 seed/copy，也不经过 scheduler。

`begin_prefill(S)` 原子申请跨 block 的逻辑表；28 层全部通过
`append_prefill_layer_kv` 写入后，`commit_prefill` 才提交逻辑长度，异常则
`abort_prefill` 回滚 block table、引用计数和长度。测试覆盖 context 1/4/15/16/17/31、
非连续 physical block、全层全 token K/V bit-exact、logits/top-5/sampling 和
prefill 后 paged decode 接续，以及事务遗漏/重复/耗尽等回滚路径。

本阶段仍未完成 B=2/4 paged prefill、paged prefill attention、scheduler direct
admission、long-context、prefix sharing/COW、FlashAttention/warp-tiled 优化和
CUDA Graph。

#### [x] Phase 6.6b：B=1/2/4 Direct Paged Prefill 与 Scheduler Direct Admission

已完成相同长度 B=1/2/4 direct paged prefill。`prefill_logits_paged_batch` 在任何
GPU 工作或事务开始前完成 batch/prompt/cache/pool 校验；所有 row 先
`begin_prefill`，随后复用 batched layer trace，将每层 K/V 直接写入各自 paged
block pool，全部 28 层和 logits 工作确认成功后统一 commit，异常统一 abort。
单请求 `prefill_logits_paged` 复用该核心路径。

`Qwen3PagedRequestScheduler` admission 已移除 contiguous prefill→seed 过渡，直接
调用 paged prefill，metrics 使用 `paged_prefill`；原有单请求 admission、FIFO、
取消/timeout、pool defer 和 bounded batch-fill decode 行为保持不变。

测试覆盖 B=1/2/4、fragmented block table、全 row/layer/token K/V bit-exact、logits/
top-5/sampling、prefill 后 decode、batch pool exhaustion/fault rollback 和 scheduler
admission metrics。仍未完成变长/packed paged prefill、scheduler prefill batching、
paged prefill attention、prefix sharing/COW、long-context、FlashAttention、CUDA
Graph 和性能优化。

#### [x] Phase 6.6c：Variable-Length Direct Paged Prefill

已完成 right-padded B=1/2/4 variable-length paged prefill。每个 row 在同一个
`PagedKvCachePool` 中为自己的 valid length 开启事务；已有 valid-length attention
path 负责忽略 padding，K/V 写入只复制每行有效 token，padding 不会写入 paged block
table。所有 row、pool、cache、token 和 padding 在 GPU 工作前校验；任一 begin、CUDA
或 fault failure 都会回滚全批已开启事务，成功时 28 层与 logits 后统一 commit。

测试覆盖 B=2 lengths 4/6、B=4 lengths 3/4/6/8 的 last-valid logits、全层有效 K/V
bit-exact、variable-length paged decode 接续、非法右 padding 的无副作用拒绝、begin
pool exhaustion 与故障回滚后的 cache/pool 重试。仍未完成 packed prefill、scheduler prefill batching、
prefix sharing/COW、long-context、FlashAttention、CUDA Graph 和性能优化。

### [-] Phase 7：paged continuous batching

**目标**：把 Phase 5 的 scheduler 接到 Phase 6 的 block manager，形成 vLLM 风格的核心运行时。

实现：

- scheduler 每轮生成 active sequence、input token、sequence length 和 block table。
- prefill 与 decode 使用不同 batch metadata；新请求优先进入 prefill 队列，避免长 prompt 长时间霸占 decode。
- 用 token budget 控制 prefill/decode 竞争；记录 queue time、TTFT、decode latency 和 block utilization。
- 此阶段再评估 CUDA Graph：只对稳定的 decode batch shape 建 graph，动态形状走 eager fallback。

**验收**：paged 与 contiguous KV 输出一致；随机请求进入/退出后无 cache 泄漏；吞吐、TTFT 和显存碎片率均可比较。

#### Phase 7.1：paged scheduler continuous-batching core

已接入 paged scheduler 的 continuous-batching 主路径：等待队列可按
`max_prefill_batch_size` 以 B=1/2/4 进行原子 direct paged prefill，变长 prompt
复用 right-padded paged prefill；prefill 失败会恢复 FIFO 队列并回收所有已分配
block。新增 `max_prefill_tokens`、`max_decode_tokens`、`max_consecutive_prefill`
和 `max_consecutive_decode` 配置，批次 metrics 记录有效/填充 token 数、decode
token 数、批大小、队列选择和连续轮次。decode 仍使用真正的 batched paged
attention，并在请求完成、取消、超时和 CUDA batch 失败时保持事务回滚。

当前仍是同步、单线程 scheduler；异步 streaming、prefix sharing/COW、CUDA
Graph 和服务层 backpressure 留给后续阶段。因此 Phase 7
整体保持进行中状态，Phase 7.1 主路径已可用。

#### Phase 7.2：scheduler packed-prefill admission

已新增 `prefer_packed_prefill`：当等待队列前缀请求长度相同且批大小为 2/4
时，scheduler 直接调用 equal-length `prefill_logits_paged_batch`，不创建
padding token 或 valid-length metadata；不同长度请求保留 right-padded fallback。
批次 action 会标记为 `paged_prefill_packed`，并将 padding ratio 记录为 0。
该阶段完成调度层无 padding 的 equal-length packed admission；测试验证 B=2
packed prefill 的采样和最终结果与单请求基线一致，并在 RTX 2080 Ti CUDA
回归中通过。任意长度 token-packed attention kernel 仍未实现，继续使用
right-padded fallback。

#### Phase 7.2a：token-packed GQA attention 与完整 packed prefill

新增 `cuda_gqa_attention_packed`：输入使用扁平 `[T,heads,head_dim]` 布局和
CUDA/I32 `offsets[B+1]`，kernel 在每个序列内部执行 causal GQA，序列之间
不会互相注意。已覆盖 F32、非等长两序列、终止 offset 和序列隔离测试，并在
RTX 2080 Ti 上通过 `ops_cuda` 回归。完整模型 packed prefill 接线已完成：新增 token-major RoPE、packed layer trace、
变长 paged prefill 入口和 paged KV slice 写入；scheduler 的 prefer_packed_prefill
直接使用扁平 token 路径，并按真实 token 数计算预算。变长 prompt 的 last-valid
logits、KV ownership、zero padding ratio、pool 回收和后续 paged decode 均已回归
验证，RTX 2080 Ti 测试通过。

### [ ] Phase 8：prefix cache、Radix 思路和更聪明的调度

**目标**：吸收 SGLang 的 prefix reuse 思路，但保持实现轻量。

路线：

1. 先对完整 token prefix 做 hash，命中时共享只读 KV blocks；只共享已经完成 prefill 的完整 block。
2. 用 hash map 管理 prefix，记录 block refcount、last-used time、token length。
3. 在 hash 版本稳定后再升级为 radix tree，支持不同请求共享最长公共 token 前缀。
4. 只在 prefix 完成 prefill 后加入 cache；正在写入的 block 不共享。
5. 采用 LRU 或 token-budget eviction；把 prefix hit/miss、节省的 prefill token 数写入 trace。

这对应 vLLM 的 prefix caching 和 SGLang 的 RadixAttention，但第一版不需要实现完整的结构化程序执行或复杂缓存一致性。

### [ ] Phase 9：服务化与 OpenAI-compatible API

**目标**：将已经稳定的 scheduler/runtime 暴露为可测试的服务，而不是把服务逻辑和量化 kernel 混在一起。

实现：

- `/v1/completions`、`/v1/chat/completions`、SSE streaming、health check。
- 请求取消、超时、最大 token、stop string、backpressure 和优雅退出。
- 服务层不直接操作 CUDA tensor，只通过 C++ runtime API 提交/消费 token。
- 为每个 request 记录 queue、prefill、decode、cache hit/miss 和错误原因。

**验收**：CLI 与 HTTP 结果一致；流式输出可中途取消；服务重载和 OOM 行为明确；压测报告包含 TTFT、P50/P99 和 tokens/s。

### [ ] Phase 10：量化与性能增强

建议顺序：

1. 权重 FP16/BF16 baseline。
2. W8A16 或 INT8 weight-only，使用 dequant + GEMM 或 CUTLASS kernel。
3. INT4 GPTQ/AWQ/GGUF 兼容格式之一；不要同时实现多个格式。
4. KV cache FP8（必须有误差和长上下文测试）。

量化验收不能只看 tokens/s：至少报告 perplexity/任务集、logits cosine similarity、首 token 和长上下文退化。

### [ ] Phase 11：扩展模型、多卡和 speculative decoding

最后再做：

- Llama/Qwen2/Qwen3 共享 decoder-only block 接口。
- GQA/MHA/MQA 参数化；RoPE scaling 参数化。
- CUDA Graph 的多 shape graph pool、graph update 和 fallback。
- tensor parallel：先切 QKV/MLP 线性层并用 NCCL all-reduce；再研究 sequence/context parallel。
- speculative decoding：先 n-gram，再小 draft model；验证 acceptance rate 和端到端收益。
- disaggregated prefill/decode 只在有真实多 GPU/服务需求时进入路线。

## 7. 核心算法实现要点

### 7.1 Prefill 与 decode 必须是两个执行路径

- **Prefill**：输入是 `[batch, prompt_len, hidden]`，GEMM 规整，适合大矩阵和 FlashAttention 类 kernel；需要一次写入多个 KV。
- **Decode**：每条序列每轮通常只输入 1 token，形状是 `[active_batch, 1, hidden]`；必须读取历史 KV，适合 paged decode attention、融合 kernel、CUDA Graph。
- 两条路径共享模型权重，但不应强行共享所有 kernel 和 batch metadata。

### 7.2 GQA attention

Q 头数为 16，KV 头数为 8。每个 Q 头通过：

```text
kv_head = q_head / (num_attention_heads / num_key_value_heads)
```

映射到 KV 头。常见 bug 包括把 K/V 重复 materialize 成 16 个头、把 head_dim 和 hidden_size 混淆、RoPE 只作用于 Q 不作用于 K，以及 K/V cache layout 在 prefill/decode 不一致。

### 7.3 RoPE、RMSNorm 和数值稳定性

- RoPE 的 cos/sin 可以按最大位置预计算，也可以按 batch position gather。
- RMSNorm 使用 `rsqrt(mean(x^2) + eps)`，累加建议 FP32。
- softmax 使用 max subtraction；长上下文和 BF16 下必须避免直接 `exp(score)`。
- 对每个 kernel 提供 FP32 debug 版本或 reference path，便于定位误差。

### 7.4 KV cache 设计

最低限度要区分：

- logical sequence length 与 allocated physical blocks；
- prompt KV 与 generated KV；
- 可共享的 immutable prefix blocks 与当前可写 block；
- cache dtype、block size、layout、refcount 和 eviction 状态。

推荐把 block metadata 保存在 host，把每次 decode 所需的 block table 以 compact int32 数组上传/更新到 device；当 batch 稳定后再考虑 device-side scheduler metadata。

### 7.5 采样

先实现：greedy、temperature、top-k、top-p、min-p 可选。采样 kernel 只处理最后一行 logits；多请求时每条序列必须有独立 RNG state，且要能在 debug 模式下固定 seed。停止条件包括 eos、max_new_tokens、stop token/string（字符串停止需要 tokenizer-aware 的 host 状态机）。

## 8. 测试与验收体系

### 8.1 正确性分层

1. CUDA 基础：allocation、copy、stream、event、dtype conversion。
2. 单 kernel：RMSNorm、RoPE、softmax、SwiGLU、paged gather/scatter。
3. 单层：与 CPU/PyTorch 对比 Q/K/V、attention 输出、MLP 输出。
4. 全模型：不同长度、batch、position、dtype、cache 状态。
5. 生成：greedy token 对齐；采样只比较统计行为和 seed 可复现性。
6. 调度/cache：随机请求进出、block 回收、prefix hit、OOM/backpressure。

### 8.2 性能指标

- TTFT：从提交到首 token。
- prefill throughput：prompt tokens/s。
- decode throughput：generated tokens/s。
- end-to-end latency：P50/P95/P99。
- GPU utilization、HBM/显存带宽、FLOP/s、kernel launch 数。
- KV cache 使用率、block 内部浪费率、prefix hit rate。
- batch size、active sequences、平均/最大上下文长度。

建议每次优化都保留基准 JSON，不能只比较一次命令行输出。使用 Nsight Systems 看 launch/stream/CPU gap，使用 Nsight Compute 针对热点 kernel 看 occupancy、memory throughput、warp stall 和 register pressure。

### 8.3 通过标准

- 结果：固定 greedy prompt 与 reference 一致；BF16 误差在预设阈值内。
- 稳定性：长时间多请求运行无增长型显存泄漏；OOM 能返回可理解错误。
- 性能：每个阶段必须相对上一阶段有可解释收益，或明确记录“功能收益而非速度收益”。
- 可维护性：模型、cache、scheduler、kernel、API 之间有独立单元测试。

## 9. 预计里程碑

| 里程碑 | 交付物 | 是否可用 |
|---|---|---|
| M0 | reference、golden、环境检查、profiling | 研究基线 |
| M1 | CPU 算子 reference + Qwen3 权重/Tokenizer 适配 | 可验证 |
| M2 | 单请求 FP16 CUDA prefill/decode | 单机 demo |
| M3 | contiguous KV + 静态 batch | 基础批处理 |
| M4 | 简单 continuous batching | 调度原型 |
| M5 | Paged KV + paged continuous batching | 推理核心 |
| M6 | Prefix cache / Radix 思路 | 高复用 runtime |
| M7 | OpenAI-compatible API 与 streaming | 可用 server |
| M8 | INT8/INT4、FP8 KV、CUDA Graph | 工程版本 |
| M9 | 多模型、多卡、speculative decoding | 扩展版本 |

个人开发时，M0-M2 是最关键的正确性阶段；M3-M6 开始体现与 llama/vLLM/SGLang 相近的 runtime 思想；M7 以后进入服务工程和性能竞争。

## 10. 具体的第一周执行清单

1. 确认 GPU compute capability、CUDA、驱动、CMake、编译器版本。
2. 下载 Qwen/Qwen3-0.6B，保存官方 `config.json`、tokenizer 和 generation config。
3. 写 Python reference，导出 3 组短 prompt 的 token ids、logits、greedy 输出。
4. 建立 CMake 工程和 `Tensor/DeviceBuffer/Stream/Event` 最小 API。
5. 实现并测试 RMSNorm、RoPE、SwiGLU、FP16/BF16 conversion；RTX 2080 Ti 先以 FP16 验收。
6. 用 cuBLASLt 完成一个 linear，并与 NumPy/PyTorch 对比。
7. 完成单层 Qwen3 forward，再扩展到 28 层。
8. 先用 contiguous KV cache 跑通一轮生成，再实现简单 continuous batching，最后开始 paged KV。
9. 每天保留一份 correctness/performance JSON 和 Nsight trace。

## 11. 关键风险与取舍

### 风险一：把模型结构实现错

Qwen3 的 GQA、Q/K RMSNorm、RoPE、权重 tied、chat template 都容易出现“能跑但结果错”。解决办法是保存逐层 golden，先对齐单层，再对齐整模。

### 风险二：过早手写 GEMM

0.6B 模型上，工程时间很容易被 GEMM kernel 吞掉，但第一版手写 GEMM 通常不如 cuBLASLt。优先把数据 layout、fusion、attention 和 cache 做正确，使用 profiler 证明某个 GEMM 真的是瓶颈后再替换。

### 风险三：连续 batching 与 cache 互相耦合

Scheduler、block manager、model runner 一起写会导致无法测试。建议 scheduler 先输出纯 host batch metadata，paged attention 先接受固定 block table，逐步连接。

### 风险四：量化掩盖基础问题

先 FP16 结果和性能稳定，再加一种量化格式。BF16/FP8 是否启用取决于硬件；量化格式、scale layout、group size 和 kernel 不应同时变化。

### 风险五：对标指标不公平

必须固定模型文件、精度、prompt 长度、输出长度、采样策略、并发数、GPU、CUDA 版本和是否使用 graph。单请求 decode 速度、prefill 速度和多请求吞吐要分开报告。

## 12. 参考资料

- [Qwen3-0.6B 官方模型配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)
- [Qwen3 Transformers 文档](https://huggingface.co/docs/transformers/main/model_doc/qwen3)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [vLLM 文档](https://docs.vllm.ai/en/stable/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [SGLang 文档](https://docs.sglang.io/)
- [SGLang 论文](https://arxiv.org/abs/2312.07104)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [MLC LLM](https://github.com/mlc-ai/mlc-llm)
- [tiny-vllm](https://github.com/jmaczan/tiny-vllm)
- [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)

## 最终建议

如果重点是“理解推理核心思想和算法”，采用 **从零实现 M0-M6 + 对照 llama.cpp/tiny-vllm**；如果重点是“尽快得到能服务的产品”，采用 **llama.cpp/ggml 作为基础设施，自己实现 Qwen3 适配、调度实验和 paged/prefix cache**。两条路线最终可以合并：从零版本作为可读的实验场，成熟底座作为性能和兼容性验证场。
