#!/usr/bin/env bash
# One-shot bootstrap: Kaggle GPU notebook -> BaryGraph job API + Ollama proxy
# (tensegrity variant: ollama + embedding + tev1 + LLM + proxy + tunnel).
#
# Running this ONE script brings up the whole chain on the Kaggle host:
#   0. deps, 1. repo, 2. ollama + model pulls (embedding + tev1 + LLM),
#   3. job API + Ollama proxy (:8765), 4. Cloudflare tunnel,
#   5. US-matrix extraction eval (both arms) run locally on the host.
#
# Typical use from a %%bash cell (no "!" prefix):
#   curl -fsSL https://raw.githubusercontent.com/oleksiy-perepelytsya/bary-graph/tensegrity-pipeline/scripts/kaggle_start.sh -o /tmp/s.sh
#   RUN_US_MATRIX=1 bash /tmp/s.sh
#
# Useful knobs (env):
#   LLM_MODEL=qwen3:14b       model used by the extraction eval arms
#   LLM_CTX=24576             context window for the eval
#   LLM_NUM_PREDICT unset     cap generated tokens per case (e.g. 4000)
#   LLM_NO_THINK=1            append qwen3 '/no_think' (disable reasoning)
#   RUN_US_MATRIX=1           run the US-matrix eval after bootstrap (0 to skip)
#   US_MATRIX_ARMS="llm_fulltext llm_graph_struct"
#   US_MATRIX_IDS_FILE=$BUNDLE/cases_pilot30.txt   case list (one id per line)
#   US_MATRIX_LIMIT=0         >0 runs only the first N cases
#   OLLAMA_BASE_PORT=11434    first instance port; one ollama per GPU
#   KAGGLE_API_TOKEN=...      reuse a fixed token instead of a fresh one

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/oleksiy-perepelytsya/bary-graph}"
BRANCH="${BRANCH:-tensegrity-pipeline}"
REPO_DIR="${REPO_DIR:-/kaggle/working/bary-graph}"
WORK="${KAGGLE_WORK_DIR:-$REPO_DIR/kaggle_work}"

# 0. packages
if ! command -v git >/dev/null || ! command -v curl >/dev/null; then
  echo "== installing deps =="
  apt-get update -qq && apt-get install -y -qq git curl
fi

# 1. repo
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "== cloning $REPO_URL @ $BRANCH =="
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
  echo "== updating repo =="
  git -C "$REPO_DIR" fetch origin "$BRANCH" && git -C "$REPO_DIR" checkout "$BRANCH" && git -C "$REPO_DIR" pull --ff-only origin "$BRANCH" || true
fi
cd "$REPO_DIR"
mkdir -p "$WORK"

export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(python3 -c 'import secrets;print(secrets.token_hex(16))')}"
export KAGGLE_API_PORT="${KAGGLE_API_PORT:-8765}"
export KAGGLE_WORK_DIR="$WORK"
export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"

echo "== token: $KAGGLE_API_TOKEN =="

# 2. ollama
if ! command -v ollama >/dev/null 2>&1; then
  echo "== installing ollama =="
  curl -fsSL https://ollama.com/install.sh | sh
fi
# The installer drops the binary somewhere on PATH; in fresh shells it may not
# resolve, which is the "ollama: command not found" people hit. Resolve it hard.
OLLAMA_BIN="$(command -v ollama 2>/dev/null || true)"
if [ -z "$OLLAMA_BIN" ]; then
  for c in /usr/local/bin/ollama /usr/bin/ollama "$HOME/.ollama/bin/ollama" /opt/ollama/ollama; do
    if [ -x "$c" ]; then OLLAMA_BIN="$c"; break; fi
  done
fi
if [ -z "$OLLAMA_BIN" ]; then
  echo "== ERROR: ollama binary not found after install =="
  exit 1
fi
echo "== ollama: $OLLAMA_BIN =="

# Two (or more) GPUs are used by running one ollama instance per GPU and
# sharding the independent eval cases across them -- ollama's own multi-GPU
# tensor split is unreliable, and per-case work is embarrassingly parallel.
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
[ "${NGPU:-0}" -lt 1 ] && NGPU=1
OLLAMA_BASE_PORT="${OLLAMA_BASE_PORT:-11434}"
echo "== GPUs detected: $NGPU (ollama on ports $OLLAMA_BASE_PORT..$((OLLAMA_BASE_PORT+NGPU-1))) =="

for i in $(seq 0 $((NGPU-1))); do
  port=$((OLLAMA_BASE_PORT+i))
  if ! curl -sf "http://localhost:$port/api/version" >/dev/null 2>&1; then
    echo "== starting ollama on GPU $i :$port =="
    CUDA_VISIBLE_DEVICES="$i" OLLAMA_HOST="0.0.0.0:$port" \
      nohup "$OLLAMA_BIN" serve > "$WORK/ollama_$i.log" 2>&1 &
  fi
done

for i in $(seq 0 $((NGPU-1))); do
  port=$((OLLAMA_BASE_PORT+i))
  ollama_up=0
  for _ in $(seq 1 60); do
    if curl -sf "http://localhost:$port/api/version" >/dev/null; then ollama_up=1; break; fi
    sleep 2
  done
  if [ "$ollama_up" != "1" ]; then
    echo "== ERROR: ollama :$port did not come up (see $WORK/ollama_$i.log) =="
    tail -20 "$WORK/ollama_$i.log" || true
    exit 1
  fi
done
# The CLI (list/pull) talks to instance 0; all instances share OLLAMA_MODELS.
export OLLAMA_HOST="127.0.0.1:$OLLAMA_BASE_PORT"

# --- model pulls -----------------------------------------------------------
# Exact-name check against the first column of `ollama list` (a substring match
# would treat qwen3-embedding:0.6b as satisfying a request for qwen3:14b).
have_model() { "$OLLAMA_BIN" list 2>/dev/null | awk 'NR>1{print $1}' | grep -qx "$1"; }

# embedding model (tensegrity t02)
if ! have_model 'qwen3-embedding:0.6b'; then
  echo "== pulling qwen3-embedding:0.6b =="
  "$OLLAMA_BIN" pull qwen3-embedding:0.6b
fi
# tev1 is the decision model for edge roles (tensegrity t04..t09); it is served
# through Ollama's Jev endpoint, which needs ollama >= 0.35.
if ! have_model 'tev1:0.8b'; then
  echo "== pulling tev1:0.8b =="
  "$OLLAMA_BIN" pull tev1:0.8b
fi
# LLM used by the US-matrix extraction eval arms
LLM_MODEL="${LLM_MODEL:-qwen3:14b}"
if ! have_model "$LLM_MODEL"; then
  echo "== pulling $LLM_MODEL (this is large; needs several GB free) =="
  df -h / | tail -1 || true
  if ! "$OLLAMA_BIN" pull "$LLM_MODEL"; then
    echo "== WARN: could not pull $LLM_MODEL — the eval arms will fail unless you set LLM_MODEL to an available model =="
  fi
fi
if ! have_model "$LLM_MODEL"; then
  echo "== ERROR: LLM '$LLM_MODEL' is still not installed; aborting before the eval =="
  "$OLLAMA_BIN" list || true
  exit 1
fi

syscode=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
    http://localhost:11434/v1/systemone -d '{}' || true)
case "$syscode" in
    404|000|"")
        echo "== WARN: /v1/systemone unavailable (HTTP ${syscode:-no response}) — install/upgrade ollama >= 0.35 for tev1 role decisions ==" ;;
esac

# 3. job API + proxy
if ! curl -sf -H "Authorization: Bearer $KAGGLE_API_TOKEN" "http://localhost:$KAGGLE_API_PORT/v1/health" >/dev/null 2>&1; then
  echo "== starting kaggle embed server+proxy on :$KAGGLE_API_PORT =="
  nohup python3 -m scripts.kaggle_embed_server > "$WORK/api.log" 2>&1 &
  sleep 3
fi

# 4. tunnel
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  dpkg -i /tmp/cloudflared.deb || apt-get install -f -y -qq
fi
pkill -f "cloudflared tunnel" 2>/dev/null || true
nohup cloudflared tunnel --no-autoupdate --url "http://localhost:$KAGGLE_API_PORT" > "$WORK/cloudflared.log" 2>&1 &
sleep 8

TUNNEL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$WORK/cloudflared.log" | head -1 || true)

echo "=================================================="
echo "  token:  $KAGGLE_API_TOKEN"
echo "  tunnel: $TUNNEL"
echo "  API:    ${TUNNEL:-http://localhost:$KAGGLE_API_PORT}/v1/health"
echo "  embed (tensegrity t02): OLLAMA_URL=${TUNNEL:-http://localhost:$KAGGLE_API_PORT}"
echo "  tev1   (tensegrity t04+): TEV1_URL=${TUNNEL:-http://localhost:$KAGGLE_API_PORT} (via /v1/systemone)"
echo "  LLM:    $LLM_MODEL (local: http://localhost:$KAGGLE_API_PORT)"
echo "=================================================="

# Preflight: make sure the chat endpoint + LLM actually answer through the
# proxy before kicking off the (long) eval. A 404 here means "model not found".
echo "== preflight: chat via http://localhost:$KAGGLE_API_PORT/api/chat =="
curl -sS -m 120 "http://localhost:$KAGGLE_API_PORT/api/chat" \
  -H "Authorization: Bearer $KAGGLE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"stream\":false,\"options\":{\"num_predict\":1}}" \
  | head -c 500
echo
echo "== installed models =="
"$OLLAMA_BIN" list || true
echo "=================================================="

# 5. US-matrix extraction eval (run locally on this host, no tunnel).
# Each arm is sharded across the NGPU ollama instances and the shards run
# concurrently; parts are merged into <arm>.jsonl afterwards.
if [ "${RUN_US_MATRIX:-1}" = "1" ]; then
  BUNDLE="$REPO_DIR/kaggle_bundle"
  export US_MATRIX_GOLD_DIR="$BUNDLE"
  export US_MATRIX_CASES_DIR="$BUNDLE/cases"
  export US_MATRIX_RESULTS_DIR="${US_MATRIX_RESULTS_DIR:-$WORK/results_us_matrix}"
  export PYTHONPATH="$REPO_DIR:$BUNDLE"
  mkdir -p "$US_MATRIX_RESULTS_DIR"

  LIMIT_ARGS=()
  if [ "${US_MATRIX_LIMIT:-0}" != "0" ]; then LIMIT_ARGS=(--limit "$US_MATRIX_LIMIT"); fi
  NP_ARGS=()
  if [ -n "${LLM_NUM_PREDICT:-}" ]; then NP_ARGS=(--num-predict "$LLM_NUM_PREDICT"); fi
  IDS_ARGS=()
  US_MATRIX_IDS_FILE="${US_MATRIX_IDS_FILE:-$BUNDLE/cases_pilot30.txt}"
  if [ -n "${US_MATRIX_IDS_FILE:-}" ] && [ -f "$US_MATRIX_IDS_FILE" ]; then
    IDS_ARGS=(--ids "$(paste -sd, "$US_MATRIX_IDS_FILE")")
    echo "== case list: $US_MATRIX_IDS_FILE ($(wc -l < "$US_MATRIX_IDS_FILE") ids) =="
  fi
  THINK_ARGS=()
  if [ "${LLM_NO_THINK:-1}" = "1" ]; then THINK_ARGS=(--no-think); fi

  echo "== US-matrix eval -> $US_MATRIX_RESULTS_DIR (shards=$NGPU) =="
  for arm in ${US_MATRIX_ARMS:-llm_fulltext llm_graph_struct}; do
    echo "=== arm: $arm ==="
    OV_ARGS=()
    case "$arm" in
      llm_graph)        OV_ARGS=(--overlays "$BUNDLE/overlays") ;;
      llm_graph_struct) OV_ARGS=(--overlays "$BUNDLE/overlays_struct") ;;
    esac
    pids=()
    for i in $(seq 0 $((NGPU-1))); do
      port=$((OLLAMA_BASE_PORT+i))
      out="$US_MATRIX_RESULTS_DIR/$arm.part$i.jsonl"
      echo "  shard $i/$NGPU -> :$port -> $out"
      python3 -u -m scripts.eval.run_eval_us_matrix --arm "$arm" \
        --base-url "http://localhost:$port" \
        --model "$LLM_MODEL" --num-ctx "${LLM_CTX:-24576}" \
        --shard "$i/$NGPU" --out "$out" \
        ${IDS_ARGS[@]+"${IDS_ARGS[@]}"} ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
        ${OV_ARGS[@]+"${OV_ARGS[@]}"} ${NP_ARGS[@]+"${NP_ARGS[@]}"} \
        ${THINK_ARGS[@]+"${THINK_ARGS[@]}"} \
        > "$US_MATRIX_RESULTS_DIR/$arm.part$i.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || echo "  (a shard exited nonzero; see $arm.part*.log)"; done

    # merge shard parts -> <arm>.jsonl (dedupe by case_id)
    python3 - "$US_MATRIX_RESULTS_DIR" "$arm" "$NGPU" << 'PYEOF'
import json, sys
from pathlib import Path
d, arm, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
seen = {}
for i in range(n):
    p = d / f"{arm}.part{i}.jsonl"
    if not p.exists():
        continue
    for ln in p.read_text(errors="replace").splitlines():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("case_id") is not None:
            seen[str(r["case_id"])] = r
(d / f"{arm}.jsonl").write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in seen.values()))
print(f"  merged {len(seen)} cases -> {d / (arm + '.jsonl')}")
PYEOF
  done

  echo "== results =="
  ls -la "$US_MATRIX_RESULTS_DIR"
  echo "== download these: llm_fulltext.jsonl and llm_graph_struct.jsonl =="
fi
