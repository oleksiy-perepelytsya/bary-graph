# Kaggle GPU embedding runner

The local box can't run embedding-heavy pipeline stages (s02_embed, s04's
type-text phase) over the full kaikki corpus on CPU. Kaggle gives free GPU
(T4/P100), so the same stages can run there behind a small HTTP job API,
reached through a cloudflared tunnel.

## On the Kaggle notebook

```bash
!git clone <repo> /kaggle/working/bary-vector && cd /kaggle/working/bary-vector
!bash scripts/kaggle_start.sh
```

This installs ollama (CUDA), starts `ollama serve`, pulls
`qwen3-embedding:0.6b`, starts the job API on :8765, and prints:

- a **token** (save it), and
- a **https://<random>.trycloudflare.com** URL.

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

## Float16 embed transport (slow uplink)

Ollama returns embeddings as JSON decimal floats — ~12.7 KB per 1024-d vector
decoded, ~4.9 KB after Cloudflare gzip. On a slow home uplink that saturates
the link while the T4 sits idle (measured: 140 KB/s rx, ~12 pairs/s in s04).
`scripts/kaggle_f16_shim.py` sits between the tunnel and Ollama, converts
`/api/embed` responses to raw float16 and returns `application/octet-stream`
(2 KB/vector, ~2.4x more vectors per second at the same bandwidth). Every
other path is proxied to Ollama verbatim, so it can also serve as
`OLLAMA_BACKENDS` for the main proxy if one tunnel is preferred.

On Kaggle (prints token + fresh tunnel URL):

```bash
cd /kaggle/working/bary-graph
nohup python3 -m scripts.kaggle_f16_shim --tunnel > f16_shim.log 2>&1 &
sleep 20 && cat f16_shim.log
```

Locally:

```bash
EMBED_F16=1 OLLAMA_URL=https://<tunnel> KAGGLE_API_TOKEN=<token> \
    python3 -m scripts.s04_l15_edges --force --batch-size 512 --embed-concurrency 4
```

`lib/embed.py` only decodes the binary payload when `EMBED_F16=1`; the JSON
path is unchanged otherwise. float16 costs ~3e-5 absolute error per component
(cosine 1.0000 on the round-trip test), far below any q threshold.

Note the T4 must be free: if a chat model (e.g. `qwen3:14b`) is resident,
unload it first (`POST /api/generate {"model": ..., "keep_alive": 0}`) or
Ollama will run the embedder on CPU.
