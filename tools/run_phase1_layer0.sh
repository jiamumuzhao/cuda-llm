#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
if ! "$PYTHON_BIN" -c 'import torch, transformers'; then
  echo "BLOCKER: Python interpreter $PYTHON_BIN cannot import torch and transformers" >&2
  exit 2
fi
"$PYTHON_BIN" tools/export_qwen3_layer0_reference.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
./build/test_qwen3_layer
ctest --test-dir build -R '^(tensor|ops_cpu|qwen3_layer)$' --output-on-failure
