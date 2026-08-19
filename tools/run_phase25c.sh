#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"

if ! "$PYTHON_BIN" -c 'import torch, transformers'; then
  echo "BLOCKER: Python interpreter $PYTHON_BIN cannot import torch and transformers" >&2
  exit 2
fi
if ! "$PYTHON_BIN" - <<'PY'
from transformers import AutoConfig
try:
    AutoConfig.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
except Exception as exc:
    raise SystemExit("BLOCKER: Qwen/Qwen3-0.6B is not available in the local cache: " + repr(exc))
PY
then
  exit 2
fi

"$PYTHON_BIN" tools/export_qwen3_full_prefill_reference.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
./build/test_qwen3_cuda_model_logits
ctest --test-dir build --output-on-failure
