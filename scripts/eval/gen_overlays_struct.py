"""Precompute *cleaned structural* graph overlays for the US-matrix corpus.

Writes ``<out>/<case_id>.json`` with the same shape as the raw overlay bundle
(``{"overlay": str, "sentences": [str, ...]}``) so the GPU host can run the
``llm_graph_struct`` arm with ``--overlays`` and no Mongo access.

    python -m scripts.eval.gen_overlays_struct --out kaggle_bundle/overlays_struct
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from scripts.eval.run_eval_us_matrix import build_overlay_struct
from scripts.eval.us_matrix_common import connect, load_case_ids

log = logging.getLogger("gen_overlays_struct")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="kaggle_bundle/overlays_struct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    coll, _ = connect()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ids = load_case_ids()
    if args.limit:
        ids = ids[: args.limit]

    sizes = []
    for cid in ids:
        overlay, sents = build_overlay_struct(coll, cid)
        (out / f"{cid}.json").write_text(
            json.dumps({"overlay": overlay, "sentences": sents}, ensure_ascii=False))
        sizes.append(len(overlay))
        log.info("%s: overlay %d chars, %d sentences", cid, len(overlay), len(sents))
    log.info("wrote %d overlays to %s (avg %d chars)",
             len(ids), out, sum(sizes) // max(1, len(sizes)))


if __name__ == "__main__":
    main()
