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
