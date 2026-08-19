#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
exec "$PYTHON_BIN" "$(dirname "$0")/test_qwen3_messages_cli.py"
