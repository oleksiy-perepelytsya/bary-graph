"""t09_extend — relaxed rescue sweep over any children t08 left unparented.

t08 only forms MetaBarys above ``meta_bary_cos_threshold``. This stage re-runs
the same formation at progressively lower thresholds (base − level×0.01,
mirroring s09's per-level floor) to absorb the remaining unparented BEs/MBs into
higher-order structure. Each new MetaBary is still classified by tev1 (Step IV).

Idempotent and safe to re-run: only ``parent_edge_id: None`` BEs are ever
consumed, so already-built triads are never touched.
"""

from __future__ import annotations

from collections.abc import Sequence

from tensegrity_pipeline import doc_bridge
from tensegrity_pipeline._base import bootstrap, finish
from tensegrity_pipeline.db import get_collection
from tensegrity_pipeline.pairing import text_resolver
from tensegrity_pipeline.stages.t08_metabary import _form_level
from tensegrity_pipeline.tev1 import get_decision_model

STAGE = "09_extend"
_STEP = 0.02
_MAX_ROUNDS_PER_THRESHOLD = 10


def _min_threshold(base: float, level: int) -> float:
    """Per-level floor: base − level × 0.01 (s09's rule)."""
    return round(base - level * 0.01, 2)


def run(argv: Sequence[str] | None = None) -> None:
    settings, args, log, cp = bootstrap(STAGE, argv)
    coll = get_collection(settings)
    base = settings.meta_bary_cos_threshold
    alpha = settings.level_factor_alpha
    log.info("start dry_run=%s base_threshold=%.2f max_level=%d",
             args.dry_run, base, args.max_level)

    if args.dry_run:
        cp.processed = cp.total = 0
        return

    model = get_decision_model(settings)
    resolve = text_resolver(coll)
    bridge_coll = doc_bridge.get_bridge_collection(settings)

    total = 0
    child_level = 15
    while child_level - 2 >= max(1, args.max_level):
        bridge_level = child_level - 1
        floor = _min_threshold(base, child_level)
        level_total = 0
        thr = base
        while thr >= floor - 1e-9:
            rounds = 0
            while rounds < _MAX_ROUNDS_PER_THRESHOLD:
                n = _form_level(
                    coll, child_level, bridge_level, thr, alpha,
                    model=model, resolve=resolve, bridge_coll=bridge_coll,
                    dry_run=False, log=log,
                )
                level_total += n
                rounds += 1
                if n == 0:
                    break
            log.info("L%d rescue sweep thr=%.2f done", child_level - 2, thr)
            thr = round(thr - _STEP, 2)
        log.info("L%d extend: created %d MetaBarys (children@L%d)",
                 child_level - 2, level_total, child_level)
        total += level_total
        child_level -= 1

    cp.processed = total
    cp.total = total
    finish(cp, settings, log)


if __name__ == "__main__":
    run()
