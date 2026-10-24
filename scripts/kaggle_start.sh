#!/usr/bin/env bash
# One-shot bootstrap: Kaggle GPU notebook → BaryGraph job API + Ollama proxy
# Paste entire file into a %%bash cell on Kaggle (GPU T4 x2).

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/oleksiy-perepelytsya/bary-graph}"
BRANCH="${BRANCH:-academic-evaluation}"
REPO_DIR="${REPO_DIR:-/kaggle/working/bary-graph}"
WORK="${KAGGLE_WORK_DIR:-$REPO_DIR/kaggle_work}"

# 0. packages
if ! command -v git >/dev/null || ! command -v zstd >/dev/null; then
  echo "== installing deps =="
  apt-get update -qq && apt-get install -y -qq git zstd curl
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
mkdir -p "$WORK" "$WORK/parsed" "$WORK/pipeline_state"

export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(python3 -c 'import secrets;print(secrets.token_hex(16))')}"
export KAGGLE_API_PORT="${KAGGLE_API_PORT:-8765}"
export KAGGLE_WORK_DIR="$WORK"
export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"

echo "== token: $KAGGLE_API_TOKEN =="

# 2. ollama
if ! command -v ollama >/dev/null; then
  echo "== installing ollama =="
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
  echo "== starting ollama :11434 =="
  CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11434 nohup ollama serve > "$WORK/ollama.log" 2>&1 &
fi
for _ in $(seq 1 60); do
  curl -sf http://localhost:11434/api/version >/dev/null && break || sleep 2
done
ollama pull qwen3-embedding:0.6b

# 3. job API + proxy
if ! curl -sf -H "Authorization: Bearer $KAGGLE_API_TOKEN" "http://localhost:$KAGGLE_API_PORT/v1/health" >/dev/null 2>&1; then
  echo "== starting kaggle embed server+proxy on :$KAGGLE_API_PORT =="
  nohup python3 -m scripts.kaggle_embed_server > "$WORK/api.log" 2>&1 &
  sleep 3
fi

# 4. data/parsed - skip s01/kaikki if parsed already present
if [ -f "$WORK/parsed/senses.jsonl" ] && [ -f "$WORK/parsed/words.jsonl" ]; then
  echo "== parsed files exist in $WORK/parsed, skipping s01_parse and kaikki download =="
else
  KAIKKI="${KAIKKI_PATH:-$WORK/kaikki.jsonl}"
  if [ ! -f "$KAIKKI" ]; then
    for cand in /kaggle/input/*/kaikki*.jsonl /kaggle/input/kaikki*/kaikki*.jsonl; do
      if [ -f "$cand" ]; then
        echo "== using existing kaikki: $cand =="
        KAIKKI="$cand"
        break
      fi
    done
  fi
  if [ ! -f "$KAIKKI" ]; then
    echo "== downloading kaikki-en (only if needed) to $WORK/kaikki.jsonl =="
    curl -fL -o "$WORK/kaikki.jsonl" https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl
    KAIKKI="$WORK/kaikki.jsonl"
  fi
  echo "== s01_parse using $KAIKKI =="
  KAIKKI_PATH="$KAIKKI" PARSED_DIR="$WORK/parsed" PIPELINE_STATE_DIR="$WORK/pipeline_state" \
    python3 -m scripts.s01_parse
fi

# 5. tunnel (main proxy JSON)
if ! command -v cloudflared >/dev/null; then
  curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  dpkg -i /tmp/cloudflared.deb || apt-get install -f -y -qq
fi
pkill -f "cloudflared tunnel.*$KAGGLE_API_PORT" 2>/dev/null || true
nohup cloudflared tunnel --no-autoupdate --url "http://localhost:$KAGGLE_API_PORT" > "$WORK/cloudflared.log" 2>&1 &
sleep 8

TUNNEL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$WORK/cloudflared.log" | head -1 || true)

# 6. f16 shim (float16 wire for slow uplink) — preferred for s04
F16_PORT="${F16_PORT:-8777}"
F16_TUNNEL=""
if [ -f "$REPO_DIR/scripts/kaggle_f16_shim.py" ]; then
  if ! curl -sf -m 3 -H "Authorization: Bearer $KAGGLE_API_TOKEN" "http://127.0.0.1:$F16_PORT/v1/health" >/dev/null 2>&1; then
    echo "== starting f16 shim on :$F16_PORT =="
    nohup python3 -m scripts.kaggle_f16_shim --port "$F16_PORT" --token "$KAGGLE_API_TOKEN" --backend "http://localhost:11434" > "$WORK/f16_shim.log" 2>&1 &
    sleep 5
  fi
  pkill -f "cloudflared tunnel.*$F16_PORT" 2>/dev/null || true
  nohup cloudflared tunnel --no-autoupdate --url "http://localhost:$F16_PORT" > "$WORK/cloudflared_f16.log" 2>&1 &
  sleep 12
  F16_TUNNEL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$WORK/cloudflared_f16.log" | tail -1 || true)
fi

echo "=================================================="
echo "  token:  $KAGGLE_API_TOKEN"
echo "  tunnel (main JSON, :$KAGGLE_API_PORT): $TUNNEL"
echo "  API:    ${TUNNEL:-http://localhost:$KAGGLE_API_PORT}/v1/health"
echo "  OLLAMA_PROXY: ${TUNNEL:-http://localhost:$KAGGLE_API_PORT}/api/embed"
if [ -n "$F16_TUNNEL" ]; then
  echo "--------------------------------------------------"
  echo "  F16_TUNNEL (:8777, float16 octet-stream): $F16_TUNNEL"
  echo "  F16 health: ${F16_TUNNEL}/v1/health"
  echo "  EMBED_F16=1 OLLAMA_URL=$F16_TUNNEL KAGGLE_API_TOKEN=$KAGGLE_API_TOKEN"
  echo "    python3 -m scripts.s04_l15_edges --force --batch-size 512 --embed-concurrency 4"
fi
echo "=================================================="
