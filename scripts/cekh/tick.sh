#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG="${CEKH_CONFIG:-scripts/cekh/config.env}"
if [ ! -f "$CONFIG" ]; then
  echo "missing $CONFIG (copy from config.example.env)" >&2
  exit 1
fi
source "$CONFIG"

LOG_DIR="${CEKH_LOG_DIR:-pipeline_state/cekh_logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TIMEOUT="${AGENT_TIMEOUT:-900}"

for i in $(seq 1 "${AGENT_COUNT:-3}"); do
  sig_var="AGENT_${i}_SIG"
  model_var="AGENT_${i}_MODEL"
  sig="${!sig_var:-}"
  model="${!model_var:-}"
  if [ -z "$sig" ] || [ -z "$model" ]; then
    echo "== agent $i misconfigured, skipping ==" >&2
    continue
  fi
  safe_sig="${sig//\//@}"
  log="$LOG_DIR/${safe_sig}_${STAMP}.log"
  echo "=== [$STAMP] shift: $sig ($model) ==="
  timeout "$TIMEOUT" opencode run \
    --model "$model" \
    "Shift start. Your signature is: ${sig}. Read AGENTS.md and perform exactly one ORPA task." \
    2>&1 | tee "$log"
done

echo "=== tick complete ==="
