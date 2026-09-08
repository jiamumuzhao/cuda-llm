# GPT-5.6-terra 对话与操作记忆

更新时间：2026-08-06

## 会话信息

- 使用模型：GPT-5.6-terra（本项目协作会话）
- 工作目录：`/root/workspace/cuda-llm`
- 用户语言偏好：中文
- 当前会话目的：理解 CUDA LLM 推理项目，并持续记录后续开发上下文。

## 对话记录

### 2026-08-19

1. 用户要求完成 Phase 6.6c。已将其明确为 6.6b 之后的 right-padded、variable-length
   direct paged prefill：B=1/2/4，每 row 只为 valid length 分配并写入 paged K/V。
2. 新增 `build_paged_padded_prefill_batch()`、
   `Qwen3CudaModel::prefill_logits_paged_padded_batch()` 和 paged cache 的
   `append_prefill_layer_kv_batched_valid_slice()`。实现复用 valid-length attention，
   在 CUDA 操作前校验 metadata/pool/cache/padding；全批 begin，成功后统一 commit，任何
   begin/CUDA/fault 失败均回滚所有已开始 row。
3. 新增 `test_qwen3_paged_padded_prefill` 与 `docs/phase66c_paged_padded_prefill.md`，
   覆盖 B=2 4/6、B=4 3/4/6/8、last-valid logits、所有 layer/token K/V bit-exact、decode
   接续、padding 拒绝、begin pool exhaustion 与故障回滚重试。构建已通过；本 sandbox
   无 CUDA-capable device，故 GPU ctest 无法执行，需在有 GPU 的环境运行。

### 2026-08-06

1. 用户打招呼：“nihao”。已用中文回应。
2. 用户请求解读项目，并特别指出 `CUDA_LLM_INFERENCE_ROADMAP.md` 是项目进展记录。
3. 已完成项目结构和路线图阅读，并向用户说明：项目是从零实现的 CUDA/C++ Qwen3-0.6B 推理运行时，当前已完成 CPU reference、权重包、tokenizer、chat template、CUDA FP16 单请求推理、KV cache、采样和静态 batch；下一阶段重点是 continuous batching 与 Paged KV Cache。
4. 用户请求建立 memory，记录 GPT-5.6-terra 的对话和操作。本文件由该请求创建。
5. 用户请求创建双 agent 协作 skill：当前 agent 负责用户沟通、生成实现提示词和验收，另一个 agent 负责写代码。已创建并安装 `lead-delegated-development` 到 `/root/.codex/skills/lead-delegated-development`，并通过官方 `quick_validate.py` 校验。
6. 用户询问下一步工作。当前建议先固定 Phase 3.2a 性能基线，再以现有 contiguous KV 和 variable-length/padded API 为基础实现 opt-in Phase 5 continuous batching。
7. 用户补充协作规则：编码 agent 也可以编译、运行测试和基准；主 agent 负责审查代码、独立复核关键结果并做最终验收。已同步更新 `lead-delegated-development` skill。
8. Phase 5.1 代码已出现：新增 `step_continuous()`、连续调度测试和 CMake 注册；静态审查发现 padded prefill 中 cache 分配发生在 try/catch 外，后续修复已将准备、cache 分配和 prefill 纳入同一 try/catch，并新增可注入的部分准备失败回滚测试。普通 sandbox 看不到 GPU，但在可见 RTX 2080 Ti 的权限环境中，`cmake --build build -j` 通过，`qwen3_scheduler`、静态 prefill/decode、padded prefill、variable-length decode、token-budget 和 `qwen3_scheduler_continuous` 共 7 项测试全部通过（100%，196.89 秒）。Phase 5.1 修复验收通过。

## 已执行的操作

- 只读查看项目文件列表：使用 `rg --files`、`find` 和 `ls`。
- 阅读路线图：`CUDA_LLM_INFERENCE_ROADMAP.md`。
- 阅读核心实现接口：
  - `include/llm/qwen3_cuda_model.h`
  - `include/llm/qwen3_scheduler.h`
  - `include/llm/qwen3_decode_workspace.h`
  - `src/qwen3_cuda_model.cpp`
- 阅读阶段说明：
  - `docs/phase43_variable_length_decode.md`
  - `docs/phase44_token_budget_scheduler.md`
- 检查项目根目录是否已有 memory 或 `AGENTS.md`。
- 本次没有运行编译、测试、基准测试，也没有修改源码逻辑。
- 本次新增文件：`MEMORY.md`。
- 新增全局 skill：`/root/.codex/skills/lead-delegated-development/SKILL.md` 及其 `agents/openai.yaml`。

## 当前项目理解

项目定位是研究型、可验证的单卡 CUDA LLM runtime，而不是完整生产服务。

当前已具备的主要能力：

- Qwen3-0.6B，28 层，FP16 CUDA 推理。
- Q/K RMSNorm、RoPE、GQA、SwiGLU、tied embedding/lm_head。
- prefill 与 decode 两条执行路径。
- 每请求 contiguous KV cache。
- greedy、temperature、top-k、top-p sampling。
- 原生 C++ tokenizer、固定 Qwen3 chat template、多轮 messages CLI。
- 静态 batch：等长、right-padded variable-length prefill、variable-length decode。
- token-budget 调度、TTFT 和 batch metrics/CSV。
- decode workspace 复用和 CUDA allocation 统计。

当前主要缺口：

- continuous batching、请求动态加入/退出、取消、超时和 backpressure。
- Paged KV Cache、block manager 和 paged continuous batching。
- prefix cache/Radix 思路。
- streaming、OpenAI-compatible HTTP API。
- 量化、CUDA Graph、多 GPU 和 speculative decoding。
- 当前许多模型 API 受 batch size `1/2/4`、prefill 长度约 `32` 等原型限制。

## 后续工作建议

优先顺序：

1. 完成 Phase 3.2 的性能基线和 decode kernel 优化。
2. 实现 Phase 5 continuous batching。
3. 实现 Phase 6 Paged KV Cache/block manager。
4. 将 scheduler 接入 paged cache，形成核心 runtime。
5. 再做 prefix cache、服务化和量化。

## 维护规则

- 每次重要对话后追加日期、用户目标、已执行操作和结论。
- 只记录对后续开发有用的事实，不记录敏感凭据或完整系统提示。
- 源码变更应记录文件路径、行为变化和验证命令。
- 测试失败、环境限制和未完成事项也要记录，避免后续重复排查。

### 2026-08-22：项目范围确认

用户明确项目重点是 CUDA 底层优化和可验证的推理 runtime，不涉及：

- prefix cache / Radix cache；
- streaming 和 OpenAI-compatible HTTP API；
- 量化（W8A16、INT8、FP8 等）。

后续重点应放在 FP16 CUDA kernel、paged/contiguous KV cache、prefill/decode
执行路径、batch scheduler、workspace/内存复用、CUDA Graph 评估、kernel benchmark
和正确性回归。路线图已同步调整 Phase 8-10 的描述。
