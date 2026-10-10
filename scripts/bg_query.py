#!/usr/bin/env python3
"""Lightweight BaryGraph CLI to avoid MCP tool calls.

Usage:
  python3 scripts/bg_query.py vs "query" -k 5
  python3 scripts/bg_query.py assoc "cold hot distillation" -k 3
  python3 scripts/bg_query.py enrich "association as substrate of thought" -p 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main() -> int:
  p = argparse.ArgumentParser(prog='bg_query')
  sub = p.add_subparsers(dest='cmd', required=True)

  vs = sub.add_parser('vs', help='vector search (minimal)')
  vs.add_argument('query')
  vs.add_argument('-k', '--limit', type=int, default=5)
  vs.add_argument('--doc-type', default='any')

  assoc = sub.add_parser('assoc', help='associative_search via lib.assoc_search')
  assoc.add_argument('query')
  assoc.add_argument('-k', '--result-top-k', type=int, default=3)
  assoc.add_argument('--max-hops', type=int, default=3)

  enrich = sub.add_parser('enrich', help='enrich_request-lite via direct search')
  enrich.add_argument('query')
  enrich.add_argument('-p', '--probes', type=int, default=2)
  enrich.add_argument('-k', '--top-k', type=int, default=2)

  args = p.parse_args()

  try:
    from lib.config import Settings
    from lib.db import get_collection, vector_search
    from lib.embed import get_embedder
    s = Settings.load()
    coll = get_collection(s)
    embedder = get_embedder(s)
  except Exception as e:
    print(f"init err: {e}", file=sys.stderr)
    return 1

  if args.cmd == 'vs':
    qv = embedder.embed(args.query)
    qvl = qv[0].tolist() if hasattr(qv, 'ndim') and qv.ndim > 1 else qv.tolist()
    hits = vector_search(coll, qvl, limit=args.limit)
    for h in hits:
      h.pop('vector', None)
    print(json.dumps(hits, default=str, indent=2))
    return 0

  if args.cmd == 'assoc':
    from lib.assoc_search import AssocConfig, run_search
    cfg = AssocConfig(
      seed_top_k=max(10,args.result_top_k*2), bridge_top_k=8, result_top_k=args.result_top_k,
      max_hops=args.max_hops, target_levels=[12,11,10], min_convergence=1,
      beam_decay=0.75, novelty_weight=0.2, convergence_weight=0.4,
      q_weight_leaf=0.5, q_weight_high=0.05, return_paths=True, include_dois=False,
    )
    res = run_search(coll, embedder, args.query, cfg, bridge_coll=None)
    print(json.dumps(res, default=str, indent=2))
    return 0

  if args.cmd == 'enrich':
    # very light: embed query as single probe
    qv = embedder.embed(args.query)
    qvl = qv[0].tolist() if hasattr(qv, 'ndim') and qv.ndim > 1 else qv.tolist()
    hits = vector_search(coll, qvl, limit=args.top_k)
    out = []
    for h in hits:
      d = dict(h)
      d.pop('vector', None)
      out.append(d)
    print(json.dumps(out, default=str, indent=2))
    return 0

  return 1

if __name__=='__main__':
  raise SystemExit(main())
