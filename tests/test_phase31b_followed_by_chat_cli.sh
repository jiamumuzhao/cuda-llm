#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(basename "$(dirname "$SCRIPT_DIR")")" == "build" ]]; then
  ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "${ROOT}"
BIN="${ROOT}/build"

timeout 180 "${BIN}/test_qwen3_batched_prefill"
timeout 180 "${BIN}/test_qwen3_scheduler_static_prefill"
output="$(timeout 180 "${BIN}/cuda_llm_chat" \
  --model "${ROOT}/artifacts/phase15/qwen3-0.6b-f32" \
  --tokenizer /root/huggingface/Qwen3-0.6B/tokenizer.json \
  --prompt '你好，CUDA LLM！' --max-new-tokens 3 --max-seq-len 32)"
printf '%s\n' "${output}"
grep -q '^prompt_ids=\[[0-9,]*\]$' <<<"${output}"
grep -q '^generated_ids=\[[0-9,]*\]$' <<<"${output}"
grep -q '^stop_reason=' <<<"${output}"
grep -q '^final_cache_length=[0-9][0-9]*$' <<<"${output}"
echo "test_phase31b_followed_by_chat_cli passed"
