#!/usr/bin/env bash
# One-shot bootstrap: Kaggle GPU notebook → BaryGraph job API + Ollama proxy
# (tensegrity variant: ollama + qwen0.6 + tev1 + proxy + tunnel, no kaikki/s01;
#  the laptop drives the tensegrity stages and only models ride the tunnel.)
# Typical use from a %%bash cell:
#   curl -fsSL https://raw.githubusercontent.com/oleksiy-perepelytsya/bary-graph/tensegrity-pipeline/scripts/kaggle_start.sh -o /tmp/s.sh
#   bash /tmp/s.sh

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
# tev1 is the decision model for edge roles (tensegrity t04..t09); it is served
# through Ollama's Jev endpoint, which needs ollama >= 0.35.
ollama pull tev1:0.8b
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
# (no kaikki/s01 step: the tensegrity stages run on the laptop; t01 is already
#  done there and only embed/tev1 model calls go through this tunnel)
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
echo "  embed (tensegrity t02): OLLAMA_URL=${TUNNEL:-http://localhost:$KAGGLE_API_PORT}"
echo "  tev1   (tensegrity t04+): TEV1_URL=${TUNNEL:-http://localhost:$KAGGLE_API_PORT} (via /v1/systemone)"
echo "=================================================="
