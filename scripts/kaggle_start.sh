#!/usr/bin/env bash
# One-shot bootstrap: Kaggle GPU notebook → BaryGraph embed job API.
#
# Paste the whole thing into a %%bash cell. It:
#   0. installs git + zstd (once)
#   1. clones (or git-pulls) the repo, branch academic-evaluation, to /kaggle/working/bary-graph
#   2. installs ollama (CUDA) if missing; starts `ollama serve`
#   3. pulls qwen3-embedding:0.6b
#   4. starts scripts/kaggle_embed_server.py on :8765
#   5. starts cloudflared quick-tunnel → prints the public https URL
#
# After it prints:
#   token: <save for scripts.kaggle_remote>
#   https://<random>.trycloudflare.com

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/oleksiy-perepelytsya/bary-graph}"
BRANCH="${BRANCH:-academic-evaluation}"
REPO_DIR="${REPO_DIR:-/kaggle/working/bary-graph}"

# --- 0. system packages -----------------------------------------------------
if ! command -v git >/dev/null || ! command -v zstd >/dev/null; then
  echo "== installing git + zstd =="
  apt-get update -qq && apt-get install -y -qq git zstd curl
fi

# --- 1. repo ------------------------------------------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "== cloning $REPO_URL @ $BRANCH =="
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
  echo "== pulling $BRANCH =="
  git -C "$REPO_DIR" fetch origin "$BRANCH" && git -C "$REPO_DIR" checkout "$BRANCH" && git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
fi
cd "$REPO_DIR"
mkdir -p "$REPO_DIR/kaggle_work"

export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(python3 -c 'import secrets;print(secrets.token_hex(16))')}"
export KAGGLE_API_PORT="${KAGGLE_API_PORT:-8765}"
export KAGGLE_WORK_DIR="${KAGGLE_WORK_DIR:-$REPO_DIR/kaggle_work}"
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

echo "== token: $KAGGLE_API_TOKEN (save this for the local driver) =="

# --- 2. ollama --------------------------------------------------------------
if ! command -v ollama >/dev/null; then
  echo "== installing ollama =="
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -sf "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
  echo "== starting ollama serve =="
  nohup ollama serve > "$KAGGLE_WORK_DIR/ollama.log" 2>&1 &
  for _ in $(seq 1 30); do curl -sf "$OLLAMA_URL/api/version" >/dev/null && break || sleep 2; done
fi
ollama pull qwen3-embedding:0.6b

# --- 3. job api ---------------------------------------------------------------
if ! curl -sf -H "Authorization: Bearer $KAGGLE_API_TOKEN" "http://localhost:$KAGGLE_API_PORT/v1/health" >/dev/null 2>&1; then
  echo "== starting embed server on :$KAGGLE_API_PORT =="
  nohup python3 -m scripts.kaggle_embed_server > "$KAGGLE_WORK_DIR/api.log" 2>&1 &
  sleep 3
fi
curl -sf -H "Authorization: Bearer $KAGGLE_API_TOKEN" "http://localhost:$KAGGLE_API_PORT/v1/health" || true
echo

# --- 4. cloudflared -----------------------------------------------------------
if ! command -v cloudflared >/dev/null; then
  echo "== installing cloudflared =="
  curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  dpkg -i /tmp/cloudflared.deb || apt-get install -f -y -qq
fi
echo "== starting cloudflared quick tunnel =="
pkill -f "cloudflared tunnel" 2>/dev/null || true
nohup cloudflared tunnel --no-autoupdate --url "http://localhost:$KAGGLE_API_PORT" \
  > "$KAGGLE_WORK_DIR/cloudflared.log" 2>&1 &
sleep 8
echo
echo "=================================================="
echo "  token:  $KAGGLE_API_TOKEN"
echo "  URL:    $(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$KAGGLE_WORK_DIR/cloudflared.log" | head -1)"
echo "=================================================="
