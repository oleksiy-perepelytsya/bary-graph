"""t06_l14_edges — pair L14 titles and build role-tagged BaryEdges.

Mirrors s06's structure, but pairing is embedding-driven (titles have no
kaikki-style relation vocabulary) rather than relation-tier driven. Requires
t05 to have derived the L14 title vectors.

Safeguard: refuses to run if any L14 BaryEdge already exists.
"""

from __future__ import annotations

from collections.abc import Sequence

from lib.embed import get_embedder
from tensegrity_pipeline._base import bootstrap, finish
from tensegrity_pipeline.db import get_collection
from tensegrity_pipeline.pairing import form_level_edges
from tensegrity_pipeline.tev1 import get_decision_model

STAGE = "06_l14_edges"
LEVEL = 14


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    log.info("start dry_run=%s", args.dry_run)

    if not args.dry_run and not args.force:
        if coll.count_documents({"doc_type": "baryedge", "level": LEVEL}, limit=1):
            raise RuntimeError(
                "L14 baryedges already present — re-running would double-parent "
                "titles. Drop them and use --reset, or pass --force."
            )

    embedder = get_embedder(settings)
    model = get_decision_model(settings)

    n = form_level_edges(
        coll, settings, embedder, model, LEVEL,
        limit=args.limit, dry_run=args.dry_run, log=log,
    )

    cp.processed = n
    cp.total = n
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
