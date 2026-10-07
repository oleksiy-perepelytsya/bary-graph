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
mkdir -p "$WORK" "$WORK/parsed" "$WORK/pipeline_state" "$WORK/a/parsed" "$WORK/a/state" "$WORK/b/parsed" "$WORK/b/state"

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

# 4. data
KAIKKI="$WORK/kaikki.jsonl"
if [ ! -f "$KAIKKI" ]; then
  echo "== downloading kaikki-en =="
  curl -fL -o "$KAIKKI" https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl
fi
if [ ! -f "$WORK/parsed/senses.jsonl" ]; then
  echo "== s01_parse =="
  KAIKKI_PATH="$KAIKKI" PARSED_DIR="$WORK/parsed" PIPELINE_STATE_DIR="$WORK/pipeline_state" \
    python3 -m scripts.s01_parse
fi

# 5. tunnel
if ! command -v cloudflared >/dev/null; then
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
echo "  OLLAMA_PROXY (for s04 etc): ${TUNNEL:-http://localhost:$KAGGLE_API_PORT}/api/embed"
echo "=================================================="
