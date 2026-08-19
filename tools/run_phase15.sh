#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
if ! "$PYTHON_BIN" -c 'import torch, transformers, safetensors'; then
  echo "BLOCKER: Python interpreter $PYTHON_BIN cannot import torch, transformers, and safetensors" >&2
  exit 2
fi
bash tools/run_phase1_layer0.sh
"$PYTHON_BIN" tests/test_convert_qwen3_contract.py
"$PYTHON_BIN" tools/convert_qwen3_hf.py
"$PYTHON_BIN" tools/verify_qwen3_package.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
./build/test_model_package
ctest --test-dir build --output-on-failure
