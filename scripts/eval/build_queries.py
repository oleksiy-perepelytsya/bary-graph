"""Build the deterministic 30-query suite (5 queries x 6 problems) from
papers_combined.parquet into evaluation/queries.json.
"""

from __future__ import annotations

import json

from scripts.eval.academic_common import QUERIES_PATH, load_rows

TEMPLATE_1 = "Approaches and techniques relevant to: {objective}"
TEMPLATE_2 = "Research on: {problem}"


def build() -> None:
    rows = load_rows()
    by_uc: dict[str, dict] = {}
    for r in rows:
        uc = r.get("use_case_key")
        if uc and uc not in by_uc:
            by_uc[uc] = r

    queries: list[dict] = []
    for uc in sorted(by_uc):
        r = by_uc[uc]
        problem = (r.get("problem_statement") or "").strip()
        objective = (r.get("objective") or "").strip()
        must = [t for t in (r.get("terms_must_include") or []) if t]
        exclude = [t for t in (r.get("terms_exclude") or []) if t]
        qs = [
            {"qid": "statement", "text": problem},
            {"qid": "objective", "text": objective},
            {"qid": "must_terms", "text": ", ".join(must)},
            {"qid": "rephrase_1", "text": TEMPLATE_1.format(objective=objective)},
            {"qid": "rephrase_2", "text": TEMPLATE_2.format(problem=problem)},
        ]
        queries.append(
            {
                "problem": uc,
                "use_case_name": r.get("use_case_name"),
                "terms_must_include": must,
                "terms_exclude": exclude,
                "queries": qs,
            }
        )

    QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUERIES_PATH.write_text(json.dumps(queries, indent=2, ensure_ascii=False))
    print(f"wrote {len(queries)} problems x 5 queries -> {QUERIES_PATH}")


if __name__ == "__main__":
    build()
