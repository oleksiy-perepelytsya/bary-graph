"""Fetch academic papers onto the цех shelf (arXiv API → papers collection).

Runs on cron (daily is plenty); agents then choose from the shelf by will via
the list_papers/claim_paper MCP tools. Categories default to a deliberate
domain spread — cross-domain SMBs need far-apart raw material, so the shelf
must not be an NLP-only buffet.

arXiv API etiquette: sequential requests with a delay between categories
(the API asks for ~3s). Content fields are inserted-if-absent, so re-runs
never rewrite a paper already sitting on the shelf.
"""

from __future__ import annotations

import argparse
import os
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from urllib.parse import urlencode

import httpx

from lib import papers
from lib.config import Settings
from lib.log import get_logger, setup_logging

DEFAULT_CATEGORIES = [
    "cs.CL",            # language / NLP
    "cs.AI",
    "stat.ML",
    "q-bio.NC",         # neuroscience
    "q-bio.MN",         # molecular networks
    "cond-mat.stat-mech",
    "physics.soc-ph",   # social physics
    "physics.hist-ph",  # history & philosophy of physics
    "astro-ph.IM",
    "math.PR",
    "econ.GN",          # general economics
    "cs.DL",            # digital libraries / scholarly communication
]

_ARXIV_API = "https://export.arxiv.org/api/query"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"

_WS_RE = re.compile(r"\s+")
_VERSION_RE = re.compile(r"v\d+$")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fetch_papers", description="Shelve arXiv metadata for volitional agent reading"
    )
    p.add_argument(
        "--categories", nargs="+", default=None,
        help="arXiv categories (default: $ARXIV_CATEGORIES csv or built-in spread)",
    )
    p.add_argument("--max-per-category", type=int, default=20)
    p.add_argument("--delay", type=float, default=3.0,
                   help="seconds between arXiv API requests")
    return p


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def _arxiv_id_from_url(url: str) -> str:
    return _VERSION_RE.sub("", url.rsplit("/abs/", 1)[-1])


def fetch_category(client: httpx.Client, category: str, max_results: int) -> list[dict]:
    params = urlencode({
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    resp = client.get(f"{_ARXIV_API}?{params}", timeout=30)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)

    docs = []
    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        abs_url = entry.findtext(f"{{{_ATOM_NS}}}id") or ""
        if "/abs/" not in abs_url:
            continue
        authors = [
            _clean(a.findtext(f"{{{_ATOM_NS}}}name") or "")
            for a in entry.findall(f"{{{_ATOM_NS}}}author")
        ]
        cats = [c.get("term") for c in entry.findall(f"{{{_ATOM_NS}}}category") if c.get("term")]
        primary = entry.find(f"{{{_ARXIV_NS}}}primary_category")
        doi_raw = (entry.findtext(f"{{{_ARXIV_NS}}}doi") or "").strip() or None
        docs.append(papers.make_paper_doc(
            arxiv_id=_arxiv_id_from_url(abs_url),
            title=_clean(entry.findtext(f"{{{_ATOM_NS}}}title") or ""),
            abstract=_clean(entry.findtext(f"{{{_ATOM_NS}}}summary") or ""),
            authors=authors,
            primary_category=(primary.get("term") if primary is not None else category),
            categories=cats,
            published=entry.findtext(f"{{{_ATOM_NS}}}published") or "",
            updated=entry.findtext(f"{{{_ATOM_NS}}}updated") or "",
            link=abs_url,
            doi=doi_raw,
        ))
    return docs


def run(argv: Sequence[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    settings = Settings.load()
    setup_logging(settings.log_level)
    log = get_logger("fetch_papers")

    env_cats = os.environ.get("ARXIV_CATEGORIES", "")
    categories = args.categories or (
        [c.strip() for c in env_cats.split(",") if c.strip()] or DEFAULT_CATEGORIES
    )

    coll = papers.get_collection(settings)
    papers.ensure_indexes(coll)

    n_new_total = n_known_total = 0
    with httpx.Client(headers={"User-Agent": "baryvector-cekh-shelf/0.1"}) as client:
        for i, cat in enumerate(categories):
            docs = fetch_category(client, cat, args.max_per_category)
            n_new, n_known = papers.upsert_many(coll, docs)
            n_new_total += n_new
            n_known_total += n_known
            log.info("cat:%s fetched=%d new=%d known=%d", cat, len(docs), n_new, n_known)
            if i < len(categories) - 1:
                time.sleep(args.delay)

    log.info(
        "shelf updated: %d new papers, %d already shelved, %d available now",
        n_new_total, n_known_total, papers.available_count(coll),
    )


if __name__ == "__main__":
    run()
