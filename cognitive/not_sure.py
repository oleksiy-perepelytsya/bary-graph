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
Model: --model, else $JEV_MODEL, else 'tev1:0.8b' (Together AI's tiny decision
model; `nimble` from Bespoke Labs is a larger drop-in).
Host:  --host, else $JEV_HOST, else http://127.0.0.1:11435.

Output modes
------------
  (default)  full JSON: {"model", "questions", "answers": {...}}
  --short    one line per question: ``field=choice  p(choice)=0.65  conf=0.07``
  --raw      the untouched API response

Importable API (for the SMB loop)
---------------------------------
    from cognitive.not_sure import judge, ask
    v = judge(state, "Is this a productive one-word-two-lives adjacency?",
              ["yes", "no"], criteria={"yes": "...", "no": "..."})
    # -> {"type": "choice", "choice": "yes",
    #     "probabilities": {"yes": 0.65, "no": 0.35}, "confidence": 0.07}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_HOST = os.environ.get("JEV_HOST") or "http://127.0.0.1:11435"
DEFAULT_MODEL = os.environ.get("JEV_MODEL") or "tev1:0.8b"


def read_state(arg: str | None) -> str:
    if not arg:
        return ""
    if arg == "-":
        return sys.stdin.read()
    if os.path.exists(arg):
        with open(arg, encoding="utf-8") as fh:
            return fh.read()
    return arg


def _criteria_from(spec: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in (spec or "").split(","):
        pair = pair.strip()
        if pair and "=" in pair:
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def make_question(instruction: str, options: list[str] | None = None,
                  criteria: dict[str, str] | None = None) -> dict:
    """Build one Jev choice question. Criteria win over bare option names."""
    crit = dict(criteria or {})
    if not crit and options:
        crit = {o: "" for o in options}
    return {"type": "choice", "instructions": instruction, "criteria": crit}


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
    return {args.field: make_question(args.ask, args.options,
                                      _criteria_from(args.criteria))}


def _request(host: str, path: str, method: str = "GET",
             payload: dict | None = None, timeout: float = 300.0) -> bytes:
    url = host.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def list_models(host: str = DEFAULT_HOST, timeout: float = 15.0) -> list[str]:
    """Names served by the ollama host (GET /api/tags)."""
    try:
        data = json.loads(_request(host, "/api/tags", timeout=timeout))
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"cannot reach {host}: {exc}")
    return [m.get("name", "") for m in data.get("models", [])]


def check(host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL) -> bool:
    names = list_models(host)
    ok = model in names
    marker = "OK " if ok else "MISSING "
    print(f"{marker}{model} @ {host}")
    for n in names:
        print(f"    - {n}")
    return ok


def call_systemone(host: str, model: str, state: str, questions: dict,
                   timeout: float, retries: int = 2) -> dict:
    """POST /v1/systemone. Retries transport hiccups; HTTP errors are fatal."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = _request(host, "/v1/systemone", "POST",
                           {"model": model, "state": state, "questions": questions},
                           timeout)
            return json.loads(raw.decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace").strip()
            raise SystemExit(
                f"systemone HTTP {exc.code} (model {model!r}): {detail}\n"
                f"Is a decision model deployed on {host}?  Try:  ollama pull {model}"
            )
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            reason = getattr(exc, "reason", exc)
            if attempt < retries:
                wait = 2 * (2 ** attempt)
                print(f"[not_sure] transient error ({reason}); retry in {wait}s",
                      file=sys.stderr)
                time.sleep(wait)
                continue
            raise SystemExit(f"cannot reach {host}: {reason}")
    raise SystemExit(f"cannot reach {host}: {last}")


# --- importable API ---------------------------------------------------------

def ask(state: str, questions: dict, *, host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL, timeout: float = 300.0) -> dict:
    """Raw parsed systemone response, for a given questions object."""
    return call_systemone(host, model, state, questions, timeout)


def judge(state: str, instruction: str, options: list[str] | None = None,
          criteria: dict[str, str] | None = None, *, field: str = "answer",
          host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL,
          timeout: float = 300.0) -> dict:
    """Convenience: one choice question -> {choice, probabilities, confidence}."""
    resp = ask(state, {field: make_question(instruction, options, criteria)},
               host=host, model=model, timeout=timeout)
    answers = resp.get("answers") if isinstance(resp, dict) else None
    if not isinstance(answers, dict):
        return {}
    return answers.get(field, {})


def _short_line(field: str, ans: dict) -> str:
    choice = ans.get("choice")
    probs = ans.get("probabilities") or {}
    p = probs.get(choice) if isinstance(probs, dict) else None
    conf = ans.get("confidence")
    bits = [f"{field}={choice}"]
    if p is not None:
        bits.append(f"p({choice})={p:.3f}")
    if conf is not None:
        bits.append(f"conf={conf:.3f}")
    return "  ".join(bits)


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
    ap.add_argument("--retries", type=int, default=2,
                    help="transport retries (default 2)")
    ap.add_argument("--check", action="store_true",
                    help="verify the model is served, then exit")
    ap.add_argument("--short", action="store_true",
                    help="compact one-line-per-question output")
    ap.add_argument("--raw", action="store_true", help="print raw API response")
    args = ap.parse_args()

    if args.check:
        return 0 if check(args.host, args.model) else 1

    state = read_state(args.state_file or args.state)
    questions = build_questions(args)
    resp = call_systemone(args.host, args.model, state, questions,
                          args.timeout, args.retries)

    if args.raw:
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        return 0

    answers = resp.get("answers") if isinstance(resp, dict) else None
    if args.short and isinstance(answers, dict):
        for field, ans in answers.items():
            if isinstance(ans, dict):
                print(_short_line(field, ans))
        return 0

    out = {"model": resp.get("model", args.model) if isinstance(resp, dict) else args.model,
           "questions": questions, "answers": resp.get("answers", resp)}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
