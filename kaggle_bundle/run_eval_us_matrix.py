"""Run the US-matrix extraction arms against the 100 gold opinions.

Arms (see :mod:`scripts.eval.us_matrix_common`):

  graph_regex   no model; graph sentence index + citation/statute regexes
  llm_fulltext  whole opinion text -> LLM -> matrix (baseline)
  llm_graph     opinion text + graph overlay -> LLM -> matrix

Candidates land in ``evaluation/results/us_matrix/<arm>.jsonl`` (plus one
``<arm>_cases/<case_id>.json`` per case) and are scored with the archive's own
``score.py``.

Usage:
    python -m scripts.eval.run_eval_us_matrix --arm graph_regex
    python -m scripts.eval.run_eval_us_matrix --arm llm_graph --limit 10 \
        --base-url https://<tunnel>.trycloudflare.com --token $KAGGLE_API_TOKEN \
        --model qwen3:14b --num-ctx 24576
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

try:  # repo layout
    from scripts.eval.us_matrix_common import (
        ENUMS,
        FIELDS,
        RESULTS_DIR,
        OllamaChat,
        complete_matrix,
        connect,
        csv_metadata,
        doc_edges,
        doc_sentences,
        doc_titles,
        empty_matrix,
        harvest_citations,
        harvest_statutes,
        load_case_ids,
        load_overlay,
        parse_json_loose,
        read_case_text,
        task_spec,
    )
except ImportError:  # flat bundle layout (GPU host)
    from us_matrix_common import (  # type: ignore[no-redef]
        ENUMS,
        FIELDS,
        RESULTS_DIR,
        OllamaChat,
        complete_matrix,
        connect,
        csv_metadata,
        doc_edges,
        doc_sentences,
        doc_titles,
        empty_matrix,
        harvest_citations,
        harvest_statutes,
        load_case_ids,
        load_overlay,
        parse_json_loose,
        read_case_text,
        task_spec,
    )

log = logging.getLogger("us_matrix")


# ---------------------------------------------------------------------------
# JSON schema handed to ollama's structured output
# ---------------------------------------------------------------------------

def matrix_schema() -> dict[str, Any]:
    """JSON schema for the 26-field matrix (enums from ENUMS)."""
    def s() -> dict:
        return {"type": "string"}

    def enum(k: str) -> dict:
        return {"type": "string", "enum": ENUMS[k]}

    return {
        "type": "object",
        "properties": {
            "case_id": s(),
            "case_name": s(),
            "court": s(),
            "court_level": enum("court_level"),
            "jurisdiction": s(),
            "date_decided": s(),
            "docket_number": s(),
            "procedure_type": enum("procedure_type"),
            "area_of_law": enum("area_of_law"),
            "judges": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "name": s(),
                    "role": {"type": "string",
                             "enum": ["author", "panel", "concurring", "dissenting"]},
                },
                "required": ["name", "role"]}},
            "parties": {"type": "object", "properties": {
                "petitioner_or_appellant": s(),
                "respondent_or_appellee": s(),
                "lower_court_party_roles": s()},
                "required": ["petitioner_or_appellant", "respondent_or_appellee",
                             "lower_court_party_roles"]},
            "lower_court": {"type": "object",
                            "properties": {"court": s(), "ruling": s()},
                            "required": ["court", "ruling"]},
            "question_presented": s(),
            "facts": {"type": "array", "items": s()},
            "arguments": {"type": "array", "items": {
                "type": "object",
                "properties": {"party": s(), "argument": s(),
                               "succeeded": {"type": "boolean"},
                               "evidence_quote": s(), "legal_basis": s()},
                "required": ["party", "argument", "succeeded", "evidence_quote", "legal_basis"]}},
            "legal_issues": {"type": "array", "items": {
                "type": "object",
                "properties": {"issue": s(), "holding_on_issue": s(),
                               "decided_for": enum("winning_party")},
                "required": ["issue", "holding_on_issue", "decided_for"]}},
            "standard_of_review": enum("standard_of_review"),
            "reasoning": {"type": "array", "items": s()},
            "holding": s(),
            "citations": {"type": "array", "items": {
                "type": "object",
                "properties": {"cited_case": s(), "citation": s(),
                               "treatment": {"type": "string",
                                             "enum": ["followed", "distinguished", "overruled",
                                                      "criticized", "cited"]}},
                "required": ["cited_case", "citation", "treatment"]}},
            "statutes_applied": {"type": "array", "items": s()},
            "disposition": enum("disposition"),
            "winning_party": enum("winning_party"),
            "separate_opinions": {"type": "array", "items": {
                "type": "object",
                "properties": {"type": {"type": "string", "enum": ["concurrence", "dissent"]},
                               "author": s(), "gist": s()},
                "required": ["type", "author", "gist"]}},
            "costs_fees": {"type": "object",
                           "properties": {"who_pays": s(),
                                          "amount_usd": {"type": ["number", "null"]}},
                           "required": ["who_pays", "amount_usd"]},
            "decisive_factors": {"type": "array", "items": s()},
        },
        "required": FIELDS,
    }


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

SYSTEM = (
    "You extract structured case matrices from US court opinions. "
    "Work only from the supplied text; never invent facts, names, citations or quotes. "
    "Every evidence_quote must be a literal substring of the opinion, copied exactly. "
    'Use "" for a field you cannot determine. Output JSON only.'
)


def metadata_block(case_id: str, meta: dict[str, dict[str, str]]) -> str:
    """The cases.csv row, in the template's own vocabulary.

    ``cases.csv`` ships with the task (see USAGE_EN.md), so every arm gets it;
    the court_level is pre-mapped into the template enum because the CSV uses
    CourtListener's (``fed_appellate`` vs ``federal_appellate``).
    """
    m = meta.get(case_id) or {}
    if not m:
        return ""
    return (
        "## Court metadata (from cases.csv, supplied with the task)\n"
        f"- court: {m.get('court', '')}\n"
        f"- court_level: {m.get('court_level', '')}\n"
        f"- jurisdiction: {m.get('jurisdiction', '')}\n"
        f"- case_name: {m.get('case_name', '')}\n"
        f"- date_filed: {m.get('date_decided', '')}\n"
        f"- docket_number: {m.get('docket_number', '')}\n"
    )


def _clip(s: str, n: int = 220) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "..."


def build_overlay(coll, case_id: str) -> tuple[str, list[str]]:
    """Graph overlay text for one opinion + its verbatim sentences (for snapping)."""
    sents = doc_sentences(coll, case_id)
    titles = doc_titles(coll, case_id)
    texts = {s["_id"]: (s.get("properties", {}).get("text") or "") for s in sents}
    lines: list[str] = ["## Graph overlay (derived from the tensegrity index)"]

    if titles:
        lines.append("\n### Section outline")
        for t in titles:
            p = t.get("properties", {})
            lines.append(f"- [{p.get('section_idx', '?')}] {p.get('text', '')}")

    edges = doc_edges(coll, list(texts.keys()))
    intra, cross = [], []
    for e in edges:
        a, b = e.get("cm1_id"), e.get("cm2_id")
        role = e.get("role") or "?"
        q = e.get("connection_strength") or 0.0
        if a in texts and b in texts:
            intra.append((role, q, a, b))
        elif a in texts or b in texts:
            cross.append((role, q, a, b))

    intra.sort(key=lambda x: -x[1])
    if intra:
        lines.append("\n### Relations between this opinion's own sentences (role, similarity)")
        for role, q, a, b in intra[:40]:
            lines.append(f"- {role} ({q:.2f}): {_clip(texts[a])} || {_clip(texts[b])}")

    if cross:
        cross.sort(key=lambda x: -x[1])
        far_ids = [c if c not in texts else d for _, _, c, d in cross[:60]]
        far = {s["_id"]: s for s in coll.find(
            {"_id": {"$in": far_ids}},
            {"properties.text": 1, "properties.doc_id": 1})}
        lines.append("\n### Parallel passages in other opinions of this corpus "
                     "(role, similarity, opinion_id)")
        for role, q, a, b in cross[:35]:
            other = b if a in texts else a
            o = far.get(other, {}).get("properties", {})
            lines.append(f"- {role} ({q:.2f}) [{o.get('doc_id', '?')}]: {_clip(o.get('text', ''))}")

    return "\n".join(lines), list(texts.values())


def messages_fulltext(case_id: str, spec: str, meta_txt: str = "") -> list[dict]:
    head = f"{meta_txt}\n" if meta_txt else ""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"{spec}\n\n---\n\n{head}"
            f"OPINION TEXT (opinion_id={case_id}). "
            "Fill every field of the matrix for this opinion and return the JSON object.\n\n"
            + read_case_text(case_id)
        )},
    ]


def messages_graph(case_id: str, spec: str, overlay: str, meta_txt: str = "",
                   budget_chars: int = 0) -> list[dict]:
    head = f"{meta_txt}\n" if meta_txt else ""
    text = read_case_text(case_id)
    if budget_chars:
        # Never truncate the opinion itself; trim the overlay (tail = cross-opinion
        # parallels first) so the prompt stays inside the model's context.
        room = budget_chars - len(spec) - len(head) - len(text) - 600
        if room < 0:
            room = 0
        if len(overlay) > room:
            overlay = (overlay[:room].rsplit("\n", 1)[0]
                       + "\n[overlay truncated to fit the context]")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"{spec}\n\n---\n\n{head}{overlay}\n\n---\n\n"
            f"OPINION TEXT (opinion_id={case_id}). Fill every field of the matrix for this "
            "opinion. Use the graph overlay above to locate the opinion's structure, its "
            "internal tensions (role `contradicts`) and any passages it shares with other "
            "opinions in the corpus. Return the JSON object.\n\n" + text
        )},
    ]


def snap_quotes(m: dict[str, Any], sents: list[str]) -> None:
    """Replace any non-verbatim evidence_quote with the sentence it came from.

    The graph holds every sentence of the opinion as a node, so a quote can always
    be resolved to an exact one - that is the guarantee being exercised here.
    """
    if not sents:
        return
    stripped = {s.strip() for s in sents if s}
    for arg in m.get("arguments") or []:
        if not isinstance(arg, dict):
            continue
        q = (arg.get("evidence_quote") or "").strip()
        if not q or q in stripped:
            continue
        best, score = "", 0.0
        head = q[:60].lower()
        for s in sents:
            if head and head in s.lower():
                best, score = s, 1.0
                break
            overlap = len(set(q.lower().split()) & set(s.lower().split())) / max(
                1, len(set(q.lower().split())))
            if overlap > score:
                best, score = s, overlap
        if best and score >= 0.5:
            arg["evidence_quote"] = best


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def arm_graph_regex(case_ids: list[str], out_jsonl: Path, out_dir: Path,
                    meta: dict[str, dict[str, str]]) -> None:
    """No model: cases.csv metadata + graph sentence index, no LLM.

    The reference to the corpus is the graph's own L15 nodes, which are literal
    sentences of each opinion, so citations/statutes are harvested from graph
    text and every evidence quote is verbatim by construction.
    """
    coll, _ = connect()
    with out_jsonl.open("w") as fh:
        for i, cid in enumerate(case_ids, 1):
            sents = [(s.get("properties", {}).get("text") or "")
                     for s in doc_sentences(coll, cid)]
            text = "\n".join(sents) if sents else read_case_text(cid)
            m = empty_matrix(cid)
            row = meta.get(cid) or {}
            for k in ("case_name", "court", "court_level", "jurisdiction",
                      "date_decided", "docket_number"):
                if row.get(k):
                    m[k] = row[k]
            cits = harvest_citations(text)
            stats = harvest_statutes(text)
            m["citations"] = [{"cited_case": "", "citation": c, "treatment": "cited"}
                              for c in cits]
            m["statutes_applied"] = stats
            picks = pick_evidence(sents)
            m["arguments"] = [{"party": "", "argument": "", "succeeded": False,
                               "evidence_quote": s, "legal_basis": ""} for s in picks]
            log.info("[graph_regex] %d/%d %s: %d citations, %d statutes, %d quotes",
                     i, len(case_ids), cid, len(cits), len(stats), len(picks))
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
            (out_dir / f"{cid}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2))


_HOLDING_HINTS = ("we hold", "we conclude", "we affirm", "we reverse", "we vacate",
                  "the court holds", "we agree", "we disagree", "it is ordered",
                  "judgment is affirmed", "judgment is reversed", "we find",
                  "accordingly", "for the foregoing reasons")


def pick_evidence(sents: list[str], n: int = 3) -> list[str]:
    """Up to ``n`` verbatim graph sentences that read like a holding."""
    good = [s for s in sents if len(s) > 60]
    scored = sorted(good, key=lambda s: (
        -sum(1 for h in _HOLDING_HINTS if h in s.lower()), -len(s)))
    return scored[:n]


def arm_llm(case_ids: list[str], out_jsonl: Path, out_dir: Path, *, arm: str,
            chat: OllamaChat, schema: dict, spec: str, snap: bool,
            meta: dict[str, dict[str, str]], overlays_dir: Path | None = None,
            dry_run: bool = False, prompt_budget: int = 0) -> None:
    coll = None
    if arm == "llm_graph" and overlays_dir is None:
        coll, _ = connect()
    t_start = time.monotonic()
    # Resume: append to the jsonl and skip case_ids already finished, so an
    # interrupted run continues instead of redoing hours of GPU work.
    done: set[str] = set()
    if out_jsonl.exists():
        for ln in out_jsonl.read_text(errors="replace").splitlines():
            try:
                done.add(json.loads(ln).get("case_id"))
            except Exception:
                pass
        done.discard(None)
    todo = [c for c in case_ids if c not in done]
    if done:
        log.info("[%s] resuming: %d already done, %d remaining", arm, len(done), len(todo))
    with out_jsonl.open("a") as fh:
        for i, cid in enumerate(todo, 1):
            sents: list[str] = []
            meta_txt = metadata_block(cid, meta)
            if arm == "llm_graph":
                if overlays_dir is not None:
                    overlay, sents = load_overlay(overlays_dir / f"{cid}.json")
                else:
                    overlay, sents = build_overlay(coll, cid)
                msgs = messages_graph(cid, spec, overlay, meta_txt, prompt_budget)
            else:
                msgs = messages_fulltext(cid, spec, meta_txt)
            if dry_run:
                chars = sum(len(m["content"]) for m in msgs)
                log.info("[%s] %d/%d %s prompt %d chars (~%d tok)",
                         arm, i, len(todo), cid, chars, chars // 4)
                continue
            t0 = time.monotonic()
            try:
                resp = chat.chat(msgs, schema=schema)
                raw = parse_json_loose(resp["content"])
                rmeta = resp["meta"]
            except Exception as exc:  # a failed case is a zero, not a crash
                log.warning("[%s] %s failed: %s", arm, cid, exc)
                raw, rmeta = {}, {}
            m = complete_matrix(cid, raw)
            m["case_id"] = cid
            if snap:
                snap_quotes(m, sents)
            dt = time.monotonic() - t0
            log.info("[%s] %d/%d %s in %.0fs (prompt %s tok, out %s tok)",
                     arm, i, len(todo), cid, dt, rmeta.get("prompt_eval_count"),
                     rmeta.get("eval_count"))
            m["_elapsed_s"] = round(dt, 1)
            m["_eval_count"] = rmeta.get("eval_count")
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
            fh.flush()
            (out_dir / f"{cid}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2))
    if chat.calls:
        log.info("[%s] done: %d cases in %.1f min; %d calls, %.1f s/call",
                 arm, len(case_ids), (time.monotonic() - t_start) / 60, chat.calls,
                 chat.seconds / chat.calls)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="US-matrix extraction benchmark arms")
    ap.add_argument("--arm", required=True,
                    choices=["graph_regex", "llm_fulltext", "llm_graph"])
    ap.add_argument("--limit", type=int, default=0, help="first N cases only")
    ap.add_argument("--ids", default="", help="comma-separated case ids")
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--token", default="")
    ap.add_argument("--num-ctx", type=int, default=24576)
    ap.add_argument("--num-predict", type=int, default=None,
                    help="cap generated tokens per case (0/None = unlimited)")
    ap.add_argument("--keep-alive", default="30m")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the throwaway warm-up call at the target num_ctx")
    ap.add_argument("--overlays", default="",
                    help="dir of precomputed <case_id>.json overlays; avoids Mongo")
    ap.add_argument("--dry-run", action="store_true",
                    help="build prompts and report their size without calling the model")
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--out", default="", help="output jsonl (default per arm)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ids = [x for x in args.ids.split(",") if x] or load_case_ids()
    if args.limit:
        ids = ids[: args.limit]
    out_jsonl = Path(args.out) if args.out else RESULTS_DIR / f"{args.arm}.jsonl"
    out_dir = RESULTS_DIR / f"{args.arm}_cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("arm=%s cases=%d model=%s ctx=%d", args.arm, len(ids), args.model, args.num_ctx)
    meta = csv_metadata()

    if args.arm == "graph_regex":
        arm_graph_regex(ids, out_jsonl, out_dir, meta)
        return

    chat = OllamaChat(args.base_url, args.model, token=args.token or None,
                      num_ctx=args.num_ctx, keep_alive=args.keep_alive,
                      timeout=args.timeout, num_predict=args.num_predict)
    if not args.no_warmup and not args.dry_run:
        wt = chat.warmup()
        log.info("warm-up at num_ctx=%d took %.1fs", args.num_ctx, wt)
    arm_llm(ids, out_jsonl, out_dir, arm=args.arm, chat=chat, schema=matrix_schema(),
            spec=task_spec(), snap=(args.arm == "llm_graph"), meta=meta,
            overlays_dir=Path(args.overlays) if args.overlays else None,
            dry_run=args.dry_run, prompt_budget=int(args.num_ctx * 3.2))


if __name__ == "__main__":
    main(sys.argv[1:])
