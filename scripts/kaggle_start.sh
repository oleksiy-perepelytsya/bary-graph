#!/usr/bin/env bash
# Bootstrap the embed job API on a fresh Kaggle GPU notebook.
#
#   bash scripts/kaggle_start.sh
#
# What it does:
#   1. installs ollama (CUDA) if missing, starts `ollama serve`
#   2. pulls the poc embed model (qwen3-embedding:0.6b)
#   3. starts scripts/kaggle_embed_server.py on :8765
#   4. starts cloudflared quick-tunnel → prints the public https URL
#
# The local driver then uses:
#   python3 -m scripts.kaggle_remote --base <https://….trycloudflare.com> \
#       --token <token> upload data/parsed/senses.jsonl

set -euo pipefail
cd "$(dirname "$0")/.."

export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(python3 -c 'import secrets;print(secrets.token_hex(16))')}"
export KAGGLE_API_PORT="${KAGGLE_API_PORT:-8765}"
export KAGGLE_WORK_DIR="${KAGGLE_WORK_DIR:-$PWD/kaggle_work}"
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

echo "== token: $KAGGLE_API_TOKEN (save this for the local driver) =="

# --- 1. ollama --------------------------------------------------------------
if ! command -v ollama >/dev/null; then
  echo "== installing ollama =="
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -sf "$OLLAMA_URL/api/version" >/dev/null; then
  echo "== starting ollama serve =="
  nohup ollama serve > "$KAGGLE_WORK_DIR/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do curl -sf "$OLLAMA_URL/api/version" && break || sleep 2; done
fi
ollama pull qwen3-embedding:0.6b

# --- 2. job api ---------------------------------------------------------------
if ! curl -sf -H "Authorization: Bearer $KAGGLE_API_TOKEN" "http://localhost:$KAGGLE_API_PORT/v1/health" >/dev/null 2>&1; then
  echo "== starting embed server on :$KAGGLE_API_PORT =="
  nohup python3 -m scripts.kaggle_embed_server > "$KAGGLE_WORK_DIR/api.log" 2>&1 &
  sleep 3
fi
curl -sf -H "Authorization: Bearer $KAGGLE_API_TOKEN" "http://localhost:$KAGGLE_API_PORT/v1/health" || true
echo

# --- 3. cloudflared -----------------------------------------------------------
if ! command -v cloudflared >/dev/null; then
  echo "== installing cloudflared =="
  curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  sudo dpkg -i /tmp/cloudflared.deb || dpkg -i /tmp/cloudflared.deb || true
fi
echo "== starting cloudflared quick tunnel (copy the https URL) =="
nohup cloudflared tunnel --no-autoupdate --url "http://localhost:$KAGGLE_API_PORT" \
  > "$KAGGLE_WORK_DIR/cloudflared.log" 2>&1 &
sleep 8
grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$KAGGLE_WORK_DIR/cloudflared.log" | head -1 || true
