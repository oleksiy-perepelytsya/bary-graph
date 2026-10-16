# US-matrix benchmark — GPU side (Kaggle)

Runs two of the three arms of the US-matrix extraction benchmark against the
**local** ollama on this machine. No tunnel, no Cloudflare timeout.

The third arm (`graph_regex`, no model) runs on the laptop and is already scored;
it needs the tensegrity graph and the gold set, neither of which is here.

## What is in here

| path | what |
|---|---|
| `run_eval_us_matrix.py`, `us_matrix_common.py` | the harness (same files as the repo) |
| `cases/` | the 100 opinion texts (the same ones in `cases/` of the task) |
| `TASK.md`, `cases.csv` | the task spec and the metadata table given to testers |
| `overlays/` | precomputed graph overlays, one `<case_id>.json` per opinion |
| `run_kaggle.sh` | driver |

**No gold data is in this bundle** (gold stays on the laptop, per the
withholding rule). This bundle only *produces* candidate matrices.

## Run it

```bash
# sanity check: prompt sizes only, no model calls
python3 -u run_eval_us_matrix.py --arm llm_graph --overlays "$PWD/overlays" --dry-run

# small first pass (10 opinions, both arms) — check timing and that it works
LIMIT=10 bash run_kaggle.sh

# full run (100 opinions, both arms)
LIMIT=0 bash run_kaggle.sh
```

Options: `MODEL=qwen3:14b CTX=24576 BASE=http://localhost:11434 ARMS="llm_graph" bash run_kaggle.sh`

To run a single arm or a specific set of opinions:

```bash
python3 -u run_eval_us_matrix.py --arm llm_fulltext --model qwen3:14b \
    --num-ctx 24576 --ids 7211461,26133
```

## What it does per opinion

- `llm_fulltext`: TASK.md + the cases.csv row + the whole opinion text → LLM → matrix JSON.
- `llm_graph`: the same, plus a **graph overlay** — the opinion's section outline
  (L14), the role-tagged relations between its own sentences, and the passages it
  shares with other opinions in the corpus. Evidence quotes in this arm are
  snapped back to exact graph sentences, so they are verbatim by construction.

Both arms get `cases.csv`, because the task hands it to testers.

## Output

- `results/<arm>.jsonl` — one candidate matrix per line (**bring these back**)
- `results/<arm>_cases/<case_id>.json` — the same, one file per case

## Timing

Generation measured on this host: **~12-15 tok/s**, prompt eval ~555 tok/s.
A full matrix is roughly 1-3k output tokens, so expect **~2-4 min per opinion**:
about 30-40 min for 100 opinions in each arm, plus the one-off ~2 min warm-up.
Use `LIMIT=10` first to confirm.

## Notes

- The harness streams responses and pins `keep_alive=30m`, so the model stays
  resident (a cold context re-allocation costs ~2 min on this host).
- `--num-ctx 24576` is used because the model + KV must fit in GPU memory. The
  harness trims the *overlay* if a prompt would overflow; the opinion text is
  never truncated.
- If `run_kaggle.sh` reports HTTP 524 you are accidentally going through a
  tunnel — make sure `BASE=http://localhost:11434`.
