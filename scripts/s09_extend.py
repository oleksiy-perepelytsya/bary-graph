"""Form MetaBary from orphan BaryEdges iteratively until convergence.

Picks up where s08 left off.  Runs the same _form_level logic on the
remaining unparented BEs, repeating full descending passes until a pass
produces zero new triads.

No embedding calls.  All reads are from existing BE vectors; the only
writes are new MetaBary docs and parent_edge_id updates on existing BEs.
Safe on a CPU-only instance: the bridge HNSW index builds with hnswlib
(CPU) and the working vectors fit comfortably in 64 GB RAM.

Resumable: MongoDB parent_edge_id=None is the ground truth for which BEs
are still unparented.  The checkpoint records cumulative triads formed for
observability but does not gate which BEs are re-processed on resume.
"""

from __future__ import annotations

from collections.abc import Sequence

from lib import checkpoint as cp_mod
from lib.db import get_collection
from scripts._base import bootstrap, finish
from scripts.s08_metabary import _form_level

_log = __import__("logging").getLogger(__name__)

STAGE = "09_extend"


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    thr = settings.meta_bary_cos_threshold
    alpha = settings.level_factor_alpha

    log.info(
        "start processed=%d dry_run=%s cos_threshold=%.2f alpha=%.2f",
        cp.processed, args.dry_run, thr, alpha,
    )

    total = cp.processed
    outer_pass = 0

    while True:
        outer_pass += 1
        pass_total = 0
        child_level = 15

        while child_level - 2 >= 1:
            bridge_level = child_level - 1
            n = _form_level(coll, child_level, bridge_level, thr, alpha, args.dry_run)
            log.info(
                "pass %d L%d MetaBary: children@L%d bridges@L%d → %d triads",
                outer_pass, child_level - 2, child_level, bridge_level, n,
            )
            pass_total += n
            if n == 0:
                break
            child_level -= 1

        total += pass_total
        log.info("pass %d complete: %d new MBs (cumulative=%d)", outer_pass, pass_total, total)

        # Persist progress after each pass without marking the stage done,
        # so a crash mid-convergence resumes from MongoDB ground truth.
        cp.processed = total
        if not args.dry_run:
            cp_mod.save(cp, settings)

        if pass_total == 0:
            break

    log.info("converged after %d passes, %d total new MBs", outer_pass, total)
    cp.total = total
    if not args.dry_run:
        finish(cp, settings, log)


if __name__ == "__main__":
    run()
