"""Shared plumbing for the US-matrix extraction benchmark (100 gold opinions).

Scores a method against the gold set in ``data/us_matrix/gold/us_matrix_gold_100``
via its own ``score.py``. Three arms are supported by
:mod:`scripts.eval.run_eval_us_matrix`:

``graph_regex``
    No model at all. The tensegrity graph is a *verbatim sentence index* of the
    corpus (every L15 node is a literal sentence of one opinion), so citations
    and statutes can be harvested from its sentences and every evidence quote is
    verbatim by construction.
``llm_fulltext``
    Baseline: the whole opinion text goes to the LLM.
``llm_graph``
    The same text plus a graph overlay: the L14 section outline, the role-tagged
    relations between the opinion's own sentences, and the cross-opinion
    parallels the graph found.

The graph DB is opened directly from ``MONGO_URI`` and hard-guarded to the
``tensegrity`` database — this harness must never touch ``barygraph_poc``.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _dir(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser() if value else default


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Overridable so the same harness runs from a flat bundle elsewhere (e.g. on the
# GPU host, where there is no Mongo and no repo checkout).
GOLD_DIR = _dir("US_MATRIX_GOLD_DIR",
                PROJECT_ROOT / "data" / "us_matrix" / "gold" / "us_matrix_gold_100")
CASES_DIR = _dir("US_MATRIX_CASES_DIR", PROJECT_ROOT / "data" / "us_matrix" / "cases")
RESULTS_DIR = _dir("US_MATRIX_RESULTS_DIR",
                   PROJECT_ROOT / "evaluation" / "results" / "us_matrix")

DB_NAME = "tensegrity"

# ---------------------------------------------------------------------------
# gold/task interface
# ---------------------------------------------------------------------------

# The 26 fields of matrix_template.json, in template order.
FIELDS = [
    "case_id", "case_name", "court", "court_level", "jurisdiction",
    "date_decided", "docket_number", "procedure_type", "area_of_law", "judges",
    "parties", "lower_court", "question_presented", "facts", "arguments",
    "legal_issues", "standard_of_review", "reasoning", "holding", "citations",
    "statutes_applied", "disposition", "winning_party", "separate_opinions",
    "costs_fees", "decisive_factors",
]

ENUMS: dict[str, list[str]] = {
    "court_level": ["supreme_court_us", "federal_appellate", "federal_district",
                    "state_supreme", "state_appellate", "state_trial", "other"],
    "procedure_type": ["appeal", "certiorari", "original_action", "motion_ruling",
                       "trial_judgment", "habeas", "petition_for_review", "other"],
    "area_of_law": ["civil", "criminal", "constitutional", "administrative", "family",
                    "employment", "tax", "immigration", "bankruptcy",
                    "intellectual_property", "other"],
    "standard_of_review": ["de_novo", "abuse_of_discretion", "clear_error",
                           "substantial_evidence", "plain_error", "other", "none"],
    "disposition": ["affirmed", "reversed", "vacated", "remanded",
                    "affirmed_in_part_reversed_in_part", "dismissed", "granted",
                    "denied", "other"],
    "winning_party": ["petitioner_or_appellant", "respondent_or_appellee", "mixed", ""],
}

# Empty value per field, used to make every delivered matrix complete (the task
# requires all 26 keys; unknown values are empty).
EMPTY: dict[str, Any] = {
    "case_id": "", "case_name": "", "court": "", "court_level": "", "jurisdiction": "",
    "date_decided": "", "docket_number": "", "procedure_type": "", "area_of_law": "",
    "question_presented": "", "holding": "", "disposition": "", "winning_party": "",
    "standard_of_review": "",
}
EMPTY_LIST = ["judges", "facts", "arguments", "legal_issues", "reasoning",
              "citations", "statutes_applied", "separate_opinions", "decisive_factors"]
EMPTY_OBJ = {"parties": {"petitioner_or_appellant": "", "respondent_or_appellee": "",
                         "lower_court_party_roles": ""},
             "lower_court": {"court": "", "ruling": ""},
             "costs_fees": {"who_pays": "", "amount_usd": None}}


def empty_matrix(case_id: str) -> dict[str, Any]:
    m: dict[str, Any] = {"case_id": case_id}
    m.update({k: (v if not isinstance(v, (list, dict)) else json.loads(json.dumps(v)))
              for k, v in EMPTY.items() if k != "case_id"})
    m.update({k: [] for k in EMPTY_LIST})
    m.update({k: json.loads(json.dumps(v)) for k, v in EMPTY_OBJ.items()})
    return m


def complete_matrix(case_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Force every one of the 26 fields to exist, with lists/dicts well typed."""
    m = empty_matrix(case_id)
    for k, v in (raw or {}).items():
        if k not in m:
            continue
        if k in EMPTY_LIST:
            m[k] = v if isinstance(v, list) else []
        elif k in EMPTY_OBJ:
            m[k] = v if isinstance(v, dict) else m[k]
        else:
            m[k] = "" if v is None else str(v)
    return m


def load_case_ids() -> list[str]:
    return sorted(p.stem for p in CASES_DIR.glob("*.txt"))


def read_case_text(case_id: str) -> str:
    return (CASES_DIR / f"{case_id}.txt").read_text(errors="ignore")


def gold_cases() -> dict[str, dict[str, Any]]:
    """Gold matrices — for the scorer/diagnostics only, never for the arms."""
    out: dict[str, dict[str, Any]] = {}
    for line in (GOLD_DIR / "gold.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[str(rec["case_id"])] = rec
    return out


def task_spec() -> str:
    return (GOLD_DIR / "TASK.md").read_text()


# ---------------------------------------------------------------------------
# cases.csv metadata
#
# ``cases.csv`` is handed to the tested party with TASK.md, cases/ and
# matrix_template.json (see USAGE_EN.md), so every arm may use it. It carries
# opinion_id, court_level, court, case_name, date_filed and docket_number — but
# its court_level uses CourtListener's vocabulary (``fed_appellate``), not the
# template's (``federal_appellate``), so a mapping is required.
# ---------------------------------------------------------------------------

COURT_LEVEL_MAP = {
    "scotus": "supreme_court_us",
    "fed_appellate": "federal_appellate",
    "fed_district": "federal_district",
    "state_supreme": "state_supreme",
    "state_appellate": "state_appellate",
    "state_trial": "state_trial",
}

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
    "Washington", "West Virginia", "Wisconsin", "Wyoming",
    "District of Columbia",
]


def csv_metadata() -> dict[str, dict[str, str]]:
    """``cases.csv`` keyed by opinion_id, with court_level already mapped."""
    import csv

    out: dict[str, dict[str, str]] = {}
    with (GOLD_DIR / "cases.csv").open() as fh:
        for row in csv.DictReader(fh):
            cid = str(row.get("opinion_id") or row.get("case_id") or "").strip()
            if not cid:
                continue
            court = (row.get("court") or "").strip()
            raw_level = (row.get("court_level") or "").strip()
            out[cid] = {
                "court": court,
                "case_name": (row.get("case_name") or "").strip(),
                "docket_number": (row.get("docket_number") or "").strip(),
                "date_decided": (row.get("date_filed") or "").strip(),
                "court_level": COURT_LEVEL_MAP.get(raw_level, raw_level),
                "jurisdiction": _jurisdiction(raw_level, court),
            }
    return out


def _jurisdiction(raw_level: str, court: str) -> str:
    """``federal`` for federal courts, else the state named in the court string.

    Gold annotates a few federal entries more precisely ("Federal (8th Cir.;
    appeal from D.N.D.)"); those are not reproducible from the CSV alone and are
    left to the model.
    """
    if raw_level.startswith("fed") or raw_level == "scotus":
        return "federal"
    for state in sorted(US_STATES, key=len, reverse=True):
        if state in court:
            return state
    return ""


# ---------------------------------------------------------------------------
# graph access (tensegrity)
# ---------------------------------------------------------------------------

def connect(uri: str | None = None):
    """Open the tensegrity collection. Refuses any other database.

    ``pymongo`` is imported here, not at module import, so the harness can run
    where no Mongo driver/instance exists (the LLM arms on the GPU host).
    """
    from pymongo import MongoClient

    uri = uri or os.environ.get("MONGO_URI", "mongodb://localhost:27117/?directConnection=true")
    db_name = os.environ.get("MONGO_DB", DB_NAME)
    if db_name != DB_NAME:
        sys.exit(f"us_matrix eval must run against {DB_NAME!r}, got {db_name!r}")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    return client[DB_NAME]["tensegrity"], client[DB_NAME]["doc_bridges"]


def id_order_key(node_id: Any) -> tuple[int, ...]:
    """Document-order key from the integers embedded in a graph node id.

    The ids are strings (``26133:p100:s0``), so sorting them directly puts
    ``p100`` before ``p2``; the numeric parts have to be parsed out.
    """
    nums = [int(x) for x in re.findall(r"\d+", str(node_id or ""))]
    return tuple(nums) or (1 << 30,)


def doc_sentences(coll, doc_id: str) -> list[dict[str, Any]]:
    """This opinion's L15 sentences, in document order."""
    sents = list(coll.find(
        {"doc_type": "node", "level": 15, "properties.doc_id": doc_id},
        {"properties.text": 1, "properties.sentence_id": 1, "properties.section_idx": 1},
    ))
    return sorted(sents, key=lambda s: id_order_key(
        s.get("properties", {}).get("sentence_id")))


def doc_titles(coll, doc_id: str) -> list[dict[str, Any]]:
    titles = list(coll.find(
        {"doc_type": "node", "level": 14, "properties.doc_id": doc_id},
        {"properties.text": 1, "properties.title_id": 1, "properties.section_idx": 1},
    ))
    return sorted(titles, key=lambda t: id_order_key(
        t.get("properties", {}).get("title_id")))


def _chunks(seq: list, n: int = 400) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def doc_edges(coll, sent_ids: list, *, max_edges: int = 4000) -> list[dict[str, Any]]:
    """Every L15 BaryEdge with at least one end among ``sent_ids``."""
    seen: dict[Any, dict[str, Any]] = {}
    proj = {"cm1_id": 1, "cm2_id": 1, "role": 1, "role_probability": 1,
            "connection_strength": 1}
    for chunk in _chunks(sent_ids):
        for key in ("cm1_id", "cm2_id"):
            for e in coll.find({"doc_type": "baryedge", "level": 15, key: {"$in": chunk}},
                               proj):
                seen[e["_id"]] = e
                if len(seen) >= max_edges:
                    return list(seen.values())
    return list(seen.values())


# ---------------------------------------------------------------------------
# citation / statute harvesting (arm A, no model)
# ---------------------------------------------------------------------------

# Reporter abbreviations seen across the corpus (federal + the state reporters the
# first pass missed: Ga., Mass., Md., NY2d/NY3d, AD2d/AD3d, Cal.App.3d/4th, WL, ...).
_REPORTER = (
    r"U\.\s?S\.|S\.\s?Ct\.|L\.\s?Ed\.\s?2d|"
    r"F\.\s?(?:2d|3d|4th|App'x|Supp\.\s?2d|Supp\.|R\.\s?D\.)|B\.\s?R\.|"
    r"N\.\s?(?:E|W|Y)\.\s?(?:2d|3d)|P\.\s?(?:2d|3d)|A\.\s?(?:2d|3d)|"
    r"A\.\s?D\.\s?(?:2d|3d)|So\.\s?(?:2d|3d)|S\.\s?(?:E|W)\.\s?(?:2d|3d)|"
    r"Cal\.\s?(?:Rptr\.\s?(?:2d|3d)|App\.\s?(?:3d|4th|2d))|Ill\.\s?(?:App\.\s?3d|2d)|"
    r"Ga\.(?:\s?App\.)?|Mass\.(?:\s?App\.\s?Ct\.)?|Md\.(?:\s?App\.)?|"
    r"Mich\.(?:\s?App\.)?|Ohio\s?St\.\s?3d|Pa\.(?:\s?Super\.)?|N\.J\.\s?Super\.|"
    r"Wis\.\s?2d|Minn\.|Iowa|Conn\.(?:\s?App\.)?|Va\.(?:\s?App\.)?|"
    r"N\.\s?C\.(?:\s?App\.)?|Kan\.(?:\s?App\.)?|Alaska|W\.\s?Va\.|S\.\s?C\.|Tenn\.|"
    r"Ariz\.(?:\s?App\.)?|Colo\.(?:\s?App\.)?|Or\.(?:\s?App\.)?|Wash\.(?:\s?App\.)?|"
    r"Ind\.(?:\s?App\.)?|Utah(?:\s?App\.)?|N\.\s?M\.(?:\s?App\.)?|Neb\.|Nev\.|Del\.|"
    r"Fla\.(?:\s?App\.)?|La\.(?:\s?App\.)?|Miss\.(?:\s?App\.)?|Ala\.(?:\s?App\.)?|"
    r"Ark\.(?:\s?App\.)?|Ky\.(?:\s?App\.)?|Okla\.|S\.\s?D\.|N\.\s?D\.|Mont\.|Idaho|"
    r"Wyo\.|Me\.|N\.\s?H\.|R\.\s?I\.|Vt\.|Haw\.|WL"
)
# One reporter cite with optional pincites: "556 U.S. 662, 678"
_ONE_CITE = rf"\d{{1,4}}\s+(?:{_REPORTER})\s*\d{{1,5}}(?:\s*,\s*\d{{1,5}})*"
# Parallel cites joined by commas: "438 Md. 451, 92 A.3d 400", plus "(2014)".
CITATION_RE = re.compile(
    rf"\b(?P<cite>{_ONE_CITE}(?:\s*,\s*{_ONE_CITE})*)(?:\s*\((?P<year>\d{{4}})\))?"
)

STATUTE_RES = [
    # "28 U.S.C. § 1001", "28 U.S.C. 2671 et seq." (gold uses both, § optional)
    re.compile(r"\b\d+\s+U\.S\.C\.\s*(?:§{1,2}\s*)?[\d]+[\dA-Za-z\.\-]*"
               r"(?:\([\w]+\))*(?:\s+et\s+seq\.?)?"),
    re.compile(r"\b\d+\s+C\.F\.R\.\s*(?:§{1,2}\s*)?[\d\.]+(?:\([\w]+\))*"),
    re.compile(r"\bU\.S\.S\.G\.\s*(?:§{1,2}\s*)?[\dA-Z][\dA-Za-z\.\-]*(?:\([\w]+\))*"),
    # "U.S. Const. amend. XIV", "U.S. Const. art. III, § 2, cl. 3"
    re.compile(r"\bU\.S\.\s*Const\.\s*(?:amend\.|art\.)\s*[IVXLC\d]+"
               r"(?:\s*,\s*§\s*\d+)?(?:\s*,\s*cl\.\s*\d+)?"),
    re.compile(r"\bFed\.\s*R\.\s*(?:Civ|Crim|Evid|App|Bankr)\.\s*P\.\s*"
               r"\d+(?:\([\w]+\))*"),
    # state codes and rules: "Cal. Penal Code § 288", "Md. Code Ann., Transp.
    # § 21-10A-04(a)(3)", "Md. Rule 8-304", "S.C. Code Ann. 48-39-10"
    re.compile(r"\b(?:[A-Z][a-zA-Z]{1,6}\.\s*){1,4}(?:Code(?:\s*Ann\.)?|Stat(?:\.|ute)?|"
               r"Rev\.\s*Stat\.|Comp\.\s*Laws|Gen\.\s*Laws|Penal\s*(?:Code|Law)|"
               r"Civ\.\s*Prac\.\s*(?:&\s*Rem\.\s*Code)?|Ann\.\s*Stat\.|Reg\.|Rule|"
               r"Declaration\s+of\s+Rights)\s*,?\s*(?:[A-Z][a-zA-Z]+\.?\s*)*"
               r"§?\s*[\d][\dA-Za-z\.\-]*(?:\([\w]+\))*"),
]

# Bare section fallback. score.py's ``st_m`` matches by normalised containment in
# either direction, so a section reference alone ("§ 21-10A-04(a)(3)") already
# matches the gold entry that carries it ("Md. Code Ann., Transp. § 21-10A-04(a)(3)").
SECTION_RE = re.compile(r"§{1,2}\s*[\d][\dA-Za-z\.\-]*(?:\([\w]+\))*(?:\([\w]+\))*")


def harvest_statutes(text: str, *, bare_sections: bool = True) -> list[str]:
    """Statute / rule / constitutional references in ``text``.

    ``bare_sections`` adds the section-only form for references whose code name
    the patterns above missed; it is on by default because ``st_m`` matches by
    containment and the gold set spells the same reference several ways.
    """
    out, seen = [], set()
    for rx in STATUTE_RES + ([SECTION_RE] if bare_sections else []):
        for m in rx.finditer(text):
            s = re.sub(r"\s+", " ", m.group(0)).strip().rstrip(",")
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def harvest_citations(text: str) -> list[str]:
    """Every reporter citation in ``text``, as the bare "volume reporter page" form.

    ``cit_m`` in score.py compares normalised strings for equality; the bare form
    matched gold best in testing (F1 0.402 vs 0.378 for bare+year and 0.28 for
    three-variant emission), and parallel cites ("438 Md. 451, 92 A.3d 400") stay
    one string because that is how the opinion and the gold set treat them.
    Pincites and years are dropped: gold is inconsistent about including them.
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in CITATION_RE.finditer(text):
        cite = re.sub(r"\s+", " ", m.group("cite")).strip()
        bare = re.sub(r"(?:\s*,\s*\d{1,5})+$", "", cite).strip()
        if bare and bare not in seen:
            seen.add(bare)
            out.append(bare)
    return out


# ---------------------------------------------------------------------------
# ollama chat (through the Kaggle tunnel or any OpenAI-less ollama host)
# ---------------------------------------------------------------------------

class OllamaChat:
    """Minimal ollama /api/chat client with retries and structured output.

    Responses are consumed as a **stream** (``stream: true``, NDJSON). That is not
    an optimisation: the public tunnel is fronted by Cloudflare, which returns 524
    when the origin sends nothing for ~100 s, so a blocking ``stream: false`` call
    for a long matrix generation always fails. Streaming also yields the token
    counts at the end of the run.
    """

    def __init__(self, base_url: str, model: str, *, token: str | None = None,
                 num_ctx: int = 32768, num_predict: int | None = None,
                 keep_alive: str = "30m", timeout: int = 1800, retries: int = 3):
        self.base = base_url.rstrip("/")
        self.model = model
        self.token = token
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.retries = retries
        self.calls = 0
        self.seconds = 0.0

    def warmup(self) -> float:
        """One throwaway token at the target context, before the timed loop.

        The first call at a new ``num_ctx`` pays a one-off ~2 min cost on this
        host (KV-cache allocation); without a warm-up it is charged to the first
        case and, worse, can exceed Cloudflare's ~100 s limit on the tunnel.
        """
        t0 = time.monotonic()
        self.chat([{"role": "user", "content": "ok"}], temperature=0.0)
        return time.monotonic() - t0

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    def chat(self, messages: list[dict], *, schema: dict | None = None,
             temperature: float = 0.0, think: bool = False) -> dict[str, Any]:
        options: dict[str, Any] = {"num_ctx": self.num_ctx, "temperature": temperature}
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        endpoints = [
            ("chat", "/api/chat"),
            ("generate", "/api/generate"),
            ("v1chat", "/v1/chat/completions"),
        ]
        last: Exception | None = None
        for attempt in range(self.retries):
            for ep_name, ep_path in endpoints:
                try:
                    if ep_name == "chat":
                        payload = {
                            "model": self.model,
                            "messages": messages,
                            "stream": True,
                            "think": think,
                            "keep_alive": self.keep_alive,
                            "options": options,
                        }
                        if schema is not None:
                            payload["format"] = schema
                        data = json.dumps(payload).encode()
                        headers = {"Content-Type": "application/json"}
                        if self.token:
                            headers["Authorization"] = f"Bearer {self.token}"
                        # try with auth, retry without on 401
                        def _do():
                            h = dict(headers)
                            r = urllib.request.Request(f"{self.base}{ep_path}", data=data, headers=h)
                            t0 = time.monotonic()
                            parts: list[str] = []
                            meta: dict[str, Any] = {}
                            with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                                for raw in resp:
                                    line = raw.strip()
                                    if not line:
                                        continue
                                    obj = json.loads(line)
                                    if obj.get("error"):
                                        raise RuntimeError(str(obj["error"]))
                                    chunk = (obj.get("message") or {}).get("content")
                                    if chunk:
                                        parts.append(chunk)
                                    if obj.get("done"):
                                        meta = {"eval_count": obj.get("eval_count"), "prompt_eval_count": obj.get("prompt_eval_count"), "total_duration_ns": obj.get("total_duration")}
                                        break
                            return parts, meta, time.monotonic() - t0
                        try:
                            parts, meta, dt = _do()
                        except urllib.error.HTTPError as ee:
                            if ee.code == 401:
                                tried = []
                                # try variants
                                variants = []
                                variants.append({})  # no auth
                                if self.token:
                                    variants.append({"Authorization": f"Bearer {self.token}"})
                                    variants.append({"Authorization": self.token})
                                    variants.append({"X-API-Key": self.token})
                                for v in variants:
                                    try:
                                        h = {"Content-Type": "application/json"}
                                        h.update(v)
                                        r = urllib.request.Request(f"{self.base}{ep_path}", data=data, headers=h)
                                        t0 = time.monotonic()
                                        parts, meta = [], {}
                                        with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                                            for raw in resp:
                                                line = raw.strip()
                                                if not line:
                                                    continue
                                                obj = json.loads(line)
                                                if obj.get("error"):
                                                    raise RuntimeError(str(obj["error"]))
                                                chunk = (obj.get("message") or {}).get("content")
                                                if chunk:
                                                    parts.append(chunk)
                                                if obj.get("done"):
                                                    meta = {"eval_count": obj.get("eval_count"), "prompt_eval_count": obj.get("prompt_eval_count"), "total_duration_ns": obj.get("total_duration")}
                                                    break
                                        dt = time.monotonic() - t0
                                        break
                                    except urllib.error.HTTPError as ee2:
                                        if ee2.code != 401:
                                            raise
                                        continue
                                else:
                                    raise  # no variant worked
                            else:
                                raise
                        if not parts and not meta:
                            raise RuntimeError("empty stream")
                        self.calls += 1
                        self.seconds += time.monotonic() - t0
                        return {"content": "".join(parts), "meta": meta}
                    elif ep_name == "generate":
                        payload = {
                            "model": self.model,
                            "prompt": self._messages_to_prompt(messages),
                            "stream": True,
                            "keep_alive": self.keep_alive,
                            "options": options,
                        }
                        if schema is not None:
                            payload["format"] = schema
                        data = json.dumps(payload).encode()
                        headers = {"Content-Type": "application/json"}
                        if self.token:
                            headers["Authorization"] = f"Bearer {self.token}"
                        def _do():
                            h = dict(headers)
                            r = urllib.request.Request(f"{self.base}{ep_path}", data=data, headers=h)
                            t0 = time.monotonic()
                            parts: list[str] = []
                            meta: dict[str, Any] = {}
                            with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                                for raw in resp:
                                    line = raw.strip()
                                    if not line:
                                        continue
                                    obj = json.loads(line)
                                    if obj.get("error"):
                                        raise RuntimeError(str(obj["error"]))
                                    chunk = obj.get("response")
                                    if chunk:
                                        parts.append(chunk)
                                    if obj.get("done"):
                                        meta = {"eval_count": obj.get("eval_count"), "prompt_eval_count": obj.get("prompt_eval_count"), "total_duration_ns": obj.get("total_duration")}
                                        break
                            return parts, meta, time.monotonic() - t0
                        try:
                            parts, meta, dt = _do()
                        except urllib.error.HTTPError as ee:
                            if ee.code == 401 and self.token:
                                h = {"Content-Type": "application/json"}
                                r = urllib.request.Request(f"{self.base}{ep_path}", data=data, headers=h)
                                t0 = time.monotonic()
                                parts, meta = [], {}
                                with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                                    for raw in resp:
                                        line = raw.strip()
                                        if not line:
                                            continue
                                        obj = json.loads(line)
                                        if obj.get("error"):
                                            raise RuntimeError(str(obj["error"]))
                                        chunk = obj.get("response")
                                        if chunk:
                                            parts.append(chunk)
                                        if obj.get("done"):
                                            meta = {"eval_count": obj.get("eval_count"), "prompt_eval_count": obj.get("prompt_eval_count"), "total_duration_ns": obj.get("total_duration")}
                                            break
                                dt = time.monotonic() - t0
                            else:
                                raise
                        if not parts and not meta:
                            raise RuntimeError("empty stream")
                        self.calls += 1
                        self.seconds += dt
                        return {"content": "".join(parts), "meta": meta}
                    elif ep_name == "v1chat":
                        payload = {
                            "model": self.model,
                            "messages": messages,
                            "stream": True,
                            "temperature": temperature,
                        }
                        if self.num_predict is not None:
                            payload["max_tokens"] = self.num_predict
                        data = json.dumps(payload).encode()
                        headers = {"Content-Type": "application/json"}
                        if self.token:
                            headers["Authorization"] = f"Bearer {self.token}"
                        def _do():
                            h = dict(headers)
                            r = urllib.request.Request(f"{self.base}{ep_path}", data=data, headers=h)
                            t0 = time.monotonic()
                            parts: list[str] = []
                            meta: dict[str, Any] = {}
                            with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                                for raw in resp:
                                    line = raw.strip()
                                    if not line:
                                        continue
                                    if line.startswith(b"data: "):
                                        line = line[6:]
                                    if line == b"[DONE]":
                                        break
                                    try:
                                        obj = json.loads(line)
                                    except Exception:
                                        continue
                                    if obj.get("error"):
                                        raise RuntimeError(str(obj["error"]))
                                    choice = (obj.get("choices") or [{}])[0]
                                    delta = choice.get("delta") or {}
                                    chunk = delta.get("content") or ""
                                    if chunk:
                                        parts.append(chunk)
                                    if obj.get("usage"):
                                        meta = {"eval_count": obj.get("usage", {}).get("completion_tokens"), "prompt_eval_count": obj.get("usage", {}).get("prompt_tokens")}
                            return parts, meta, time.monotonic() - t0
                        try:
                            parts, meta, dt = _do()
                        except urllib.error.HTTPError as ee:
                            if ee.code == 401 and self.token:
                                h = {"Content-Type": "application/json"}
                                r = urllib.request.Request(f"{self.base}{ep_path}", data=data, headers=h)
                                t0 = time.monotonic()
                                parts, meta = [], {}
                                with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                                    for raw in resp:
                                        line = raw.strip()
                                        if not line:
                                            continue
                                        if line.startswith(b"data: "):
                                            line = line[6:]
                                        if line == b"[DONE]":
                                            break
                                        try:
                                            obj = json.loads(line)
                                        except Exception:
                                            continue
                                        if obj.get("error"):
                                            raise RuntimeError(str(obj["error"]))
                                        choice = (obj.get("choices") or [{}])[0]
                                        delta = choice.get("delta") or {}
                                        chunk = delta.get("content") or ""
                                        if chunk:
                                            parts.append(chunk)
                                        if obj.get("usage"):
                                            meta = {"eval_count": obj.get("usage", {}).get("completion_tokens"), "prompt_eval_count": obj.get("usage", {}).get("prompt_tokens")}
                                dt = time.monotonic() - t0
                            else:
                                raise
                        if not parts and not meta:
                            raise RuntimeError("empty stream")
                        self.calls += 1
                        self.seconds += dt
                        return {"content": "".join(parts), "meta": meta}
                except urllib.error.HTTPError as e:
                    body = ""
                    try:
                        body = e.read().decode(errors="replace")[:600]
                    except Exception:
                        pass
                    last = RuntimeError(f"{e} — response body: {body!r}")
                    # Try other endpoints on 404/401/403 (auth/path variants)
                    if e.code in (404, 401, 403):
                        continue
                    break
                except Exception as exc:
                    last = exc
                    break
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"ollama chat failed after {self.retries} attempts: {last}\n"
            f"  base={self.base} model={self.model} token={'set' if self.token else 'unset'}\n"
            f"  {self._diagnose()}"
        )

    def _diagnose(self) -> str:
        """Probe the server so failures say *why* (auth vs missing model)."""
        hdr = {"Content-Type": "application/json"}
        if self.token:
            hdr["Authorization"] = f"Bearer {self.token}"
        out = []
        for path in ("/api/version", "/api/tags"):
            try:
                req = urllib.request.Request(f"{self.base}{path}", headers=hdr)
                with urllib.request.urlopen(req, timeout=15) as r:
                    out.append(f"{path} -> {r.read(400).decode(errors='replace')}")
            except urllib.error.HTTPError as he:
                b = ""
                try:
                    b = he.read().decode(errors="replace")[:300]
                except Exception:
                    pass
                out.append(f"{path} -> HTTP {he.code} {b!r}")
            except Exception as ee:
                out.append(f"{path} -> {ee}")
        return " | ".join(out)


def parse_json_loose(text: str) -> dict[str, Any]:
    """Best-effort JSON object out of a model reply."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# precomputed overlays (lets the graph arm run without Mongo, e.g. on the GPU host)
# ---------------------------------------------------------------------------

def save_overlay(path: Path, overlay: str, sentences: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"overlay": overlay, "sentences": sentences},
                               ensure_ascii=False))


def load_overlay(path: Path) -> tuple[str, list[str]]:
    data = json.loads(path.read_text())
    return data.get("overlay", ""), data.get("sentences", [])
