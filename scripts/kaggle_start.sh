#!/usr/bin/env bash
# One-shot bootstrap: Kaggle GPU notebook → BaryGraph embed jobs (2× GPU).
#
# Paste the whole thing into a %%bash cell. Steps:
#   0. install git + zstd + curl
#   1. clone/pull branch academic-evaluation into /kaggle/working/bary-graph
#   2. ollama serve pinned to each GPU (CUDA_VISIBLE_DEVICES 0 and 1,
#      :11434 and :11435), pull qwen3-embedding:0.6b
#   3. start scripts/kaggle_embed_server.py on :8765
#   4. download kaikki-en English dump if missing
#   5. run s01_parse inline (blocking) if parsed output missing
#   6. split senses 2 ways (a = first half, b = second)
#   7. submit both s02_embed jobs via the local API (GPU 0 → a, GPU 1 → b)
#   8. cloudflared quick tunnel → print URL (watch progress at the URL + /v1/jobs)
#
# The local driver then uses:
#   python3 -m scripts.kaggle_remote --base <https://….trycloudflare.com> \
#       --token <token> poll <job_id> / download …

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/oleksiy-perepelytsya/bary-graph}"
BRANCH="${BRANCH:-academic-evaluation}"
REPO_DIR="${REPO_DIR:-/kaggle/working/bary-graph}"
WORK="${KAGGLE_WORK_DIR:-$REPO_DIR/kaggle_work}"

# --- 0. system packages ------------------------------------------------------
if ! command -v git >/dev/null || ! command -v zstd >/dev/null; then
  echo "== installing git + zstd =="
  apt-get update -qq && apt-get install -y -qq git zstd curl
fi

# --- 1. repo -----------------------------------------------------------------
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "== cloning $REPO_URL @ $BRANCH =="
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch origin "$BRANCH" && git -C "$REPO_DIR" checkout "$BRANCH" && git -C "$REPO_DIR" pull --ff-only origin "$BRANCH" || true
fi
cd "$REPO_DIR"
mkdir -p "$WORK"

export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-$(python3 -c 'import secrets;print(secrets.token_hex(16))')}"
export KAGGLE_API_PORT="${KAGGLE_API_PORT:-8765}"
export KAGGLE_WORK_DIR="$WORK"
export OLLAMA_URL="http://localhost:11434"
export EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-256}"

echo "== token: $KAGGLE_API_TOKEN =="

# --- 2. ollama, one instance per GPU ----------------------------------------
if ! command -v ollama >/dev/null; then
  echo "== installing ollama =="
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
  echo "== ollama gpu0 :11434 =="
  CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11434 nohup ollama serve > "$WORK/ollama0.log" 2>&1 &
fi
if ! curl -sf http://localhost:11435/api/version >/dev/null 2>&1; then
  echo "== ollama gpu1 :11435 =="
  CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11435 nohup ollama serve > "$WORK/ollama1.log" 2>&1 &
fi
for _ in $(seq 1 30); do
  curl -sf http://localhost:11434/api/version >/dev/null && curl -sf http://localhost:11435/api/version >/dev/null && break || sleep 2
done
ollama pull qwen3-embedding:0.6b

# --- 3. embed job API --------------------------------------------------------
if ! curl -sf -H "Authorization: Bearer $KAGGLE_API_TOKEN" "http://localhost:$KAGGLE_API_PORT/v1/health" >/dev/null 2>&1; then
  echo "== starting embed server on :$KAGGLE_API_PORT =="
  nohup python3 -m scripts.kaggle_embed_server > "$WORK/api.log" 2>&1 &
  sleep 3
fi

# --- 4. data -----------------------------------------------------------------
KAIKKI="$WORK/kaikki.jsonl"
if [ ! -f "$KAIKKI" ]; then
  echo "== downloading kaikki English dump (~3.3GB) =="
  curl -fL -o "$KAIKKI" https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl
fi

# --- 5. s01_parse (inline, blocking) -----------------------------------------
if [ ! -f "$WORK/parsed/senses.jsonl" ]; then
  echo "== s01_parse =="
  KAIKKI_PATH="$KAIKKI" PARSED_DIR="$WORK/parsed" PIPELINE_STATE_DIR="$WORK/pipeline_state" \
    python3 -m scripts.s01_parse
fi

# --- 6. split into two halves -------------------------------------------------
if [ ! -f "$WORK/a/parsed/senses.jsonl" ] || [ ! -f "$WORK/b/parsed/senses.jsonl" ]; then
  echo "== splitting =="
  mkdir -p "$WORK/a/parsed" "$WORK/a/state" "$WORK/b/parsed" "$WORK/b/state"
  split -n l/2 -d "$WORK/parsed/senses.jsonl" /tmp/half
  mv /tmp/half00 "$WORK/a/parsed/senses.jsonl"
  mv /tmp/half01 "$WORK/b/parsed/senses.jsonl"
fi
wc -l "$WORK/a/parsed/senses.jsonl" "$WORK/b/parsed/senses.jsonl"

# --- 7. launch both s02 halves ------------------------------------------------
echo "== launching s02 jobs =="
curl -sf -X POST -H "Authorization: Bearer $KAGGLE_API_TOKEN" -H "Content-Type: application/json" \
  -d "{\"stage\":\"s02_embed\",\"args\":[\"--force\"],\"env\":{\"PARSED_DIR\":\"$WORK/a/parsed\",\"PIPELINE_STATE_DIR\":\"$WORK/a/state\",\"OLLAMA_URL\":\"http://localhost:11434\",\"EMBED_BATCH_SIZE\":\"$EMBED_BATCH_SIZE\"}}" \
  "http://localhost:$KAGGLE_API_PORT/v1/run" || true
curl -sf -X POST -H "Authorization: Bearer $KAGGLE_API_TOKEN" -H "Content-Type: application/json" \
  -d "{\"stage\":\"s02_embed\",\"args\":[\"--force\"],\"env\":{\"PARSED_DIR\":\"$WORK/b/parsed\",\"PIPELINE_STATE_DIR\":\"$WORK/b/state\",\"OLLAMA_URL\":\"http://localhost:11435\",\"EMBED_BATCH_SIZE\":\"$EMBED_BATCH_SIZE\"}}" \
  "http://localhost:$KAGGLE_API_PORT/v1/run" || true
echo

# --- 8. cloudflared -----------------------------------------------------------
if ! command -v cloudflared >/dev/null; then
  curl -fsSL -o /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  dpkg -i /tmp/cloudflared.deb || apt-get install -f -y -qq
fi
pkill -f "cloudflared tunnel" 2>/dev/null || true
nohup cloudflared tunnel --no-autoupdate --url "http://localhost:$KAGGLE_API_PORT" \
  > "$WORK/cloudflared.log" 2>&1 &
sleep 8
echo
echo "=================================================="
echo "  token:  $KAGGLE_API_TOKEN"
echo "  URL:    $(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$WORK/cloudflared.log" | head -1)"
echo "  poll:   \$URL/v1/jobs"
echo "=================================================="
