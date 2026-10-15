# Kaggle GPU embedding runner

The local box can't run embedding-heavy pipeline stages (s02_embed, s04's
type-text phase) over the full kaikki corpus on CPU. Kaggle gives free GPU
(T4/P100), so the same stages can run there behind a small HTTP job API,
reached through a cloudflared tunnel **created by the Kaggle notebook itself**.

> This is **not** the host's MCP endpoint. The barygraph MCP server is published
> on a stable ngrok dev domain — see RUNBOOK.md → "MCP + TUNNEL OPERATIONS". The
> Kaggle tunnel below is a separate, per-run random URL by design (the notebook
> is ephemeral), which is why it stays `https://<random>.trycloudflare.com`.

## On the Kaggle notebook

```bash
!git clone <repo> /kaggle/working/bary-vector && cd /kaggle/working/bary-vector
!bash scripts/kaggle_start.sh
```

This installs ollama (CUDA), starts `ollama serve`, pulls
`qwen3-embedding:0.6b` **and `tev1:0.8b`**, starts the job API on :8765, and
prints:

- a **token** (save it), and
- a **https://<random>.trycloudflare.com** URL.

The notebook also prints a WARN line if `/v1/systemone` (the tev1 decision
model) is not reachable — a missing tev1 is not fatal on a Kaggle run, since
`scripts/kaggle_start.sh` only needs it when the local machine drives
tensegrity pairing stages through the tunnel (below).

(For a stable domain instead of a random one, create a named tunnel +
Cloudflare DNS record and replace the `cloudflared tunnel --url ...` line.)

## From the local machine

```bash
BASE=https://<random>.trycloudflare.com
TOK=<token>

# stage inputs live under kaggle_work/parsed
python3 -m scripts.kaggle_remote --base $BASE --token $TOK \
    upload data/parsed/senses.jsonl --as parsed/senses.jsonl

# run the embed stage on the Kaggle GPU
python3 -m scripts.kaggle_remote --base $BASE --token $TOK \
    run s02_embed

python3 -m scripts.kaggle_remote --base $BASE --token $TOK poll <job_id>

python3 -m scripts.kaggle_remote --base $BASE --token $TOK \
    download parsed/senses_embedded.jsonl -o data/parsed/senses_embedded.jsonl
```

Other stages that need embeddings (e.g. `s04_l15_edges`) can be added to
`STAGES` in `scripts/kaggle_embed_server.py` the same way. The API only ever
touches files under `kaggle_work/` and runs whitelisted stage modules.

Security note: the tunnel URL is a bearer-token gate. Use a long random
token (`kaggle_start.sh` generates one by default) and treat the URL like a
password — no TLS client auth beyond that.

## Driving tev1 + embeddings through the tunnel (tensegrity)

The job API also **proxies ollama paths** on the same base URL: any `/api/*`
request (embeddings) and `/v1/systemone` (the tev1 decision model) are
forwarded verbatim to the notebook's ollama; everything else hits the job
API (`/run`, `/poll`, `/upload`, ...). That lets the local machine keep the
Mongo-driven stages where Mongo lives (the laptop) while GPU-serving the two
slow model calls:

```bash
BASE=https://<random>.trycloudflare.com
# local .env.tensegrity, pointed at the Kaggle GPU:
OLLAMA_URL=$BASE        # t02 embed (via /api/embed)
TEV1_URL=$BASE          # t07/t08 role decisions (via /v1/systemone)
```

Route contract is pinned by `tests/unit/test_kaggle_proxy.py` (`/api/*` and
`/v1/systemone` forward, `/` and `/v1/chat` are not ollama paths). If the
notebook did not pull `tev1:0.8b`, systemone calls fail and tensegrity falls
back to `role=None` / `role_source="unclassified"` — same contract as an
unreachable local tev1.

## ⚠️ Session persistence

The notebook's filesystem (`/kaggle/working`) is **ephemeral** — closing the
tab, losing network, or a timeout deletes `kaggle_work/`, including
checkpoints and any finished-but-not-downloaded artifacts. Rules:

1. After each s02 half-job reaches `status=done`, immediately
   `scripts.kaggle_remote ... download` its output.
2. Do NOT rely on the Kaggle session surviving overnight; assume it can die.
3. The retry path: new session → `kaggle_start.sh` → re-split input →
   relaunch the job halves; each half resumes from its own checkpoint only
   if the state dir survived (i.e. the session didn't die).
