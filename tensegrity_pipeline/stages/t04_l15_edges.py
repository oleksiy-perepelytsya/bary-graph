"""t04_l15_edges — pair L15 sentences and build role-tagged BaryEdges.

Mirrors s04's pairing core: embedding-cosine ANN → greedy unique matching →
signed-role barycentric vector. Each pair is additionally classified by tev1
into ``complement`` / ``contradict``. Both participant sentences are stamped
with the new BaryEdge's ``parent_edge_id`` (unique-parent).

Safeguard: refuses to run if any L15 BaryEdge already exists.
"""

from __future__ import annotations

from collections.abc import Sequence

from lib.embed import get_embedder
from tensegrity_pipeline._base import bootstrap, finish
from tensegrity_pipeline.db import get_collection
from tensegrity_pipeline.pairing import form_level_edges
from tensegrity_pipeline.tev1 import get_decision_model

STAGE = "04_l15_edges"
LEVEL = 15


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    log.info("start dry_run=%s", args.dry_run)

    if not args.dry_run and not args.force:
        if coll.count_documents({"doc_type": "baryedge", "level": LEVEL}, limit=1):
            raise RuntimeError(
                "L15 baryedges already present — re-running would double-parent "
                "sentences. Drop them and use --reset, or pass --force."
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
