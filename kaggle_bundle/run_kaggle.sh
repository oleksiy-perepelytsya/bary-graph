#!/usr/bin/env bash
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export US_MATRIX_GOLD_DIR="$HERE"
export US_MATRIX_CASES_DIR="$HERE/cases"
export US_MATRIX_RESULTS_DIR="/kaggle/working/results_us_matrix"
export PYTHONPATH="$HERE"
MODEL="${MODEL:-qwen3:14b}"
BASE="${BASE:-http://localhost:11434}"
CTX="${CTX:-24576}"
LIMIT="${LIMIT:-0}"
ARMS="${ARMS:-llm_fulltext llm_graph}"
LIMIT_ARGS=()
if [ "$LIMIT" != "0" ]; then LIMIT_ARGS=(--limit "$LIMIT"); fi
mkdir -p "$US_MATRIX_RESULTS_DIR"
echo "ollama check..."
curl -s -m 20 "$BASE/api/ps" || true
echo
TOKEN_ARGS=()
if [ -n "${TOKEN:-}" ]; then TOKEN_ARGS=(--token "$TOKEN"); fi
for arm in $ARMS; do
  echo "=== $arm ==="
  if [ "$arm" = "llm_graph" ]; then
    python3 -u run_eval_us_matrix.py --arm "$arm" --model "$MODEL" --base-url "$BASE" \
      --num-ctx "$CTX" --overlays "$HERE/overlays" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"}
  else
    python3 -u run_eval_us_matrix.py --arm "$arm" --model "$MODEL" --base-url "$BASE" \
      --num-ctx "$CTX" ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} ${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"}
  fi
done
ls -la "$US_MATRIX_RESULTS_DIR"
