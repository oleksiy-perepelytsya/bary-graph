#!/usr/bin/env python3
"""not_sure.py — uncertainty oracle via a decision model (Jev API on ollama).

Use it when the agent is unsure: hand it a STATE (the text to judge) and one or
more QUESTIONS; a decision model returns a choice and probabilities for each
allowed answer. This is a *prompt for a decision*, never a fact or a coercion —
weigh it like any other estimation.

The wire format is TypeSafe's Jev API, served by ollama at POST /v1/systemone.
A "decision model" (e.g. Bespoke Labs' `nimble`) reads the state once per
question and scores the allowed answer tokens directly — fast, no reasoning
step.

Two ways to ask
---------------
  1. Quick single question (convenience):
       not_sure.py --state "..." --ask "Which option fits?" --options a b c
     (add --criteria 'a=desc,b=desc' for choice descriptions)

  2. Full control (free parameters): pass the questions object verbatim:
       not_sure.py --state-file cand.json --questions-json '{
         "verdict": {"type":"choice",
                     "instructions":"Is this a genuine cross-domain doubling?",
                     "criteria":{"yes":"two senses share a form","no":"coincidence"}},
         "bridge":  {"type":"choice",
                     "instructions":"Which bridge best connects them?",
                     "criteria":{"cm-a":"...","cm-b":"..."}}}'

STATE may be inline text, a file path, or '-' for stdin.
Model: --model, else $JEV_MODEL, else 'nimble'.
Host:  --host, else $JEV_HOST, else http://127.0.0.1:11435.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_HOST = os.environ.get("JEV_HOST") or "http://127.0.0.1:11435"
DEFAULT_MODEL = os.environ.get("JEV_MODEL") or "nimble"


def read_state(arg: str | None) -> str:
    if not arg:
        return ""
    if arg == "-":
        return sys.stdin.read()
    if os.path.exists(arg):
        with open(arg, encoding="utf-8") as fh:
            return fh.read()
    return arg


def build_questions(args: argparse.Namespace) -> dict:
    if args.questions_json:
        raw = args.questions_json
        if os.path.exists(raw):
            raw = open(raw, encoding="utf-8").read()
        q = json.loads(raw)
        if not isinstance(q, dict):
            raise SystemExit("--questions-json must be a JSON object")
        return q
    if not args.ask:
        raise SystemExit("need --ask or --questions-json")
    criteria = {}
    for pair in (args.criteria or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" in pair:
            k, v = pair.split("=", 1)
            criteria[k.strip()] = v.strip()
    if args.options and not criteria:
        criteria = {o: "" for o in args.options}
    q = {"type": "choice", "instructions": args.ask, "criteria": criteria}
    return {args.field: q}


def call_systemone(host: str, model: str, state: str, questions: dict,
                   timeout: float) -> dict:
    url = host.rstrip("/") + "/v1/systemone"
    body = json.dumps({"model": model, "state": state,
                       "questions": questions}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace").strip()
        raise SystemExit(
            f"systemone HTTP {exc.code} (model {model!r}): {detail}\n"
            f"Is a decision model deployed on {host}?  Try:  ollama pull {model}"
        )
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {host}: {exc.reason}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Uncertainty oracle (Jev API).")
    ap.add_argument("--state", help="state text | file path | '-' for stdin")
    ap.add_argument("--state-file", help="read state from this file")
    ap.add_argument("--ask", help="single-question instructions (convenience)")
    ap.add_argument("--options", nargs="*", default=[], help="allowed choices")
    ap.add_argument("--criteria", help="'a=desc,b=desc' choice descriptions")
    ap.add_argument("--field", default="answer", help="question name")
    ap.add_argument("--questions-json",
                    help="full questions object (JSON string or file path)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--raw", action="store_true", help="print raw API response")
    args = ap.parse_args()

    state = read_state(args.state_file or args.state)
    questions = build_questions(args)
    resp = call_systemone(args.host, args.model, state, questions, args.timeout)

    if args.raw:
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        return 0

    out = {"model": resp.get("model", args.model),
           "questions": questions, "answers": resp.get("answers", resp)}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
