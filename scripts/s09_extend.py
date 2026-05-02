"""Form MetaBary from orphan BaryEdges iteratively until convergence.

Picks up where s08 left off.  For each child level, sweeps the cosine
threshold from the base value (meta_bary_cos_threshold, default 0.9) down
to a per-level floor of ``0.9 - level × 0.01``, logging yield at each step:

  L15 floor = 0.75  (15 rounds of 0.01 relaxation)
  L14 floor = 0.76
  L13 floor = 0.77
  … and so on upward

Repeats full descending passes until a complete pass produces zero new
triads across all levels and all threshold rounds.

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

_BASE_THR = 0.9


def _min_threshold(level: int) -> float:
    """Per-level floor: 0.9 - level × 0.01  (L15→0.75, L14→0.76, …)."""
    return round(_BASE_THR - level * 0.01, 2)


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    base_thr = settings.meta_bary_cos_threshold
    alpha = settings.level_factor_alpha

    log.info(
        "start processed=%d dry_run=%s cos_threshold=%.2f alpha=%.2f",
        cp.processed, args.dry_run, base_thr, alpha,
    )

    total = cp.processed
    outer_pass = 0

    while True:
        outer_pass += 1
        pass_total = 0
        child_level = 15

        while child_level - 2 >= 1:
            bridge_level = child_level - 1
            min_thr = _min_threshold(child_level)
            level_total = 0

            current_thr = base_thr
            while current_thr >= min_thr:
                n = _form_level(
                    coll, child_level, bridge_level, current_thr, alpha, args.dry_run
                )
                log.info(
                    "pass %d L%d thr=%.2f: children@L%d bridges@L%d → %d triads",
                    outer_pass, child_level - 2, current_thr, child_level, bridge_level, n,
                )
                level_total += n
                current_thr = round(current_thr - 0.01, 2)

            pass_total += level_total
            if level_total == 0:
                break  # no triads at this child_level even at the lowest threshold
            child_level -= 1

        total += pass_total
        log.info("pass %d complete: %d new MBs (cumulative=%d)", outer_pass, pass_total, total)

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
