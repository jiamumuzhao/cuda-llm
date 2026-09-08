#!/usr/bin/env bash
set -euo pipefail

PACKAGE=${1:-artifacts/phase15/qwen3-0.6b-f32}
ITERATIONS=${2:-10}
OUTPUT=${3:-benchmark_decode.csv}
BENCH=${BENCHMARK_BIN:-./build/benchmark_paged_decode_e2e}

if [[ ! -x "$BENCH" ]]; then
  echo "benchmark executable not found: $BENCH" >&2
  exit 1
fi

printf 'suite,batch,context,steps,path,avg_ms,tokens_per_sec,cuda_malloc_calls,cuda_free_calls,allocated_bytes,freed_bytes,attention_launches\n' > "$OUTPUT"

run_case() {
  local suite=$1 batch=$2 context=$3 steps=$4 path=$5 mode=$6
  local line
  if [[ "$batch" == 1 ]]; then
    line=$("$BENCH" "$PACKAGE" "$steps" "$context" "$mode" | tail -n 1)
  else
    line=$("$BENCH" "$PACKAGE" "$steps" "$context" "$mode" "$batch" | tail -n 1)
  fi
  local avg tok malloc free allocated freed launches
  avg=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="avg_ms"{print $2}')
  tok=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="tokens_per_sec"{print $2}')
  malloc=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="steady_state_cuda_malloc_calls"{print $2}')
  free=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="cuda_free_calls"{print $2}')
  allocated=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="allocated_bytes"{print $2}')
  freed=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="freed_bytes"{print $2}')
  launches=$(echo "$line" | tr ' ' '\n' | awk -F= '$1=="batch_attention_launches"{print $2}')
  # Graph replay does not re-enter the host attention wrapper. Its captured
  # graph contains one attention node per decoder layer.
  if [[ "$path" != eager ]]; then launches=$((28 * steps)); fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$suite" "$batch" "$context" "$steps" "$path" "$avg" "$tok" \
    "$malloc" "$free" "$allocated" "$freed" "$launches" >> "$OUTPUT"
  echo "suite=$suite batch=$batch context=$context steps=$steps path=$path avg_ms=$avg tokens_per_sec=$tok"
}

run_suite() {
  local suite=$1 steps=$2 contexts=$3
  for batch in 1 2 4; do
    for context in $contexts; do
      if [[ "$batch" == 1 ]]; then
        run_case "$suite" 1 "$context" "$steps" eager eager
        run_case "$suite" 1 "$context" "$steps" reference_graph graph
        run_case "$suite" 1 "$context" "$steps" flash_graph graph_flash
      else
        run_case "$suite" "$batch" "$context" "$steps" eager batch_eager
        run_case "$suite" "$batch" "$context" "$steps" reference_graph batch_graph
        run_case "$suite" "$batch" "$context" "$steps" flash_graph batch_graph_flash
      fi
    done
  done
}

run_suite quick_regression 3 "15 128"
run_suite performance_report 10 "15 32 128 256"
run_suite stability 64 "128 256"
run_suite capacity_boundary 1 "511"

echo "wrote $OUTPUT"
