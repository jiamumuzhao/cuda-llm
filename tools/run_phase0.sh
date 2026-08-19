#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p artifacts/phase0
python tools/check_env.py
python tools/run_reference.py "$@"
python tools/make_report.py
