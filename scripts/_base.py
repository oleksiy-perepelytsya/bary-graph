from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Sequence

from lib import checkpoint as cp_mod
from lib.checkpoint import Checkpoint
from lib.config import Settings
from lib.log import get_logger, setup_logging

# stage name → module path (run() entrypoint). Dict order == execution order.
STAGES: dict[str, str] = {
    "01_parse": "scripts.s01_parse",
    "02_embed": "scripts.s02_embed",
    "03_insert_nodes": "scripts.s03_insert_nodes",
    "04_l15_edges": "scripts.s04_l15_edges",
    "05_word_vectors": "scripts.s05_word_vectors",
    "06_l14_edges": "scripts.s06_l14_edges",
    "07_orphan_reentry": "scripts.s07_orphan_reentry",
    "08_metabary": "scripts.s08_metabary",
    "09_extend": "scripts.s09_extend",
    "10_index": "scripts.s10_index",
}

STAGE_ORDER: list[str] = list(STAGES)


class StageOrderError(RuntimeError):
    """Raised when a stage is invoked before its prerequisite has completed."""


def _prev_stage(stage: str) -> str | None:
    idx = STAGE_ORDER.index(stage)
    return STAGE_ORDER[idx - 1] if idx > 0 else None


def make_parser(stage: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"bary {stage}", description=f"BaryGraph pipeline stage {stage}"
    )
    p.add_argument("--reset", action="store_true", help="discard checkpoint and start over")
    p.add_argument("--limit", type=int, default=None, help="process at most N items (dev)")
    p.add_argument("--dry-run", action="store_true", help="do not write to MongoDB / disk")
    p.add_argument("--batch-size", type=int, default=None, help="override BATCH_SIZE")
    p.add_argument("--kaikki-path", type=str, default=None, help="override KAIKKI_PATH")
    p.add_argument(
        "--force",
        action="store_true",
        help="bypass stage-order / already-done safeguards (use with care)",
    )
    return p


def _enforce_order(stage: str, settings: Settings, force: bool, log: logging.Logger) -> None:
    """Safeguard: refuse to run if the previous stage hasn't completed.

    Each stage produces data the next consumes (parsed → embedded → inserted
    → L15 BEs → word vectors → …). Running out of order silently corrupts
    downstream state, so we fail loudly instead. ``--force`` overrides for
    dev experimentation.
    """
    prev = _prev_stage(stage)
    if prev is None:
        return
    if cp_mod.is_done(prev, settings):
        return
    msg = (
        f"stage '{stage}' requires '{prev}' to be complete "
        f"(pipeline_state/{prev}.json missing or done=false)"
    )
    if force:
        log.warning("STAGE-ORDER BYPASSED (--force): %s", msg)
        return
    raise StageOrderError(msg)


def bootstrap(
    stage: str, argv: Sequence[str] | None = None
) -> tuple[Settings, argparse.Namespace, logging.Logger, Checkpoint]:
    """Common entrypoint plumbing for every pipeline stage.

    Enforces stage ordering, loads/creates the checkpoint, and warns if the
    stage has already completed (re-running a completed stage requires
    ``--reset`` or ``--force`` to avoid accidental double-ingest).
    """
    settings = Settings.load()
    setup_logging(settings.log_level)
    log = get_logger(stage)

    args = make_parser(stage).parse_args(argv)

    _enforce_order(stage, settings, args.force, log)

    if args.reset:
        cp_mod.reset(stage, settings)
        invalidate_downstream(stage, settings, log)
        log.info("checkpoint reset")

    cp = cp_mod.load(stage, settings) or Checkpoint(stage=stage)

    if cp.done and not args.force and not args.reset:
        raise StageOrderError(
            f"stage '{stage}' already completed at {cp.completed_at}; "
            f"re-run with --reset (start over) or --force (ignore)"
        )

    cp_mod.save(cp, settings)  # write initial / touched checkpoint

    log.info(
        "settings: db=%s coll=%s kaikki=%s fake_embed=%s",
        settings.mongo_db,
        settings.mongo_collection,
        args.kaikki_path or settings.kaikki_path,
        settings.fake_embed,
    )
    return settings, args, log, cp


def finish(cp: Checkpoint, settings: Settings, log: logging.Logger) -> None:
    """Mark a stage complete and persist the checkpoint."""
    cp.mark_done()
    cp_mod.save(cp, settings)
    log.info("stage complete: processed=%d total=%d", cp.processed, cp.total)


def invalidate_downstream(stage: str, settings: Settings, log: logging.Logger) -> None:
    """When a stage is re-run with ``--reset``, downstream checkpoints become
    stale. Clear them so the order guard forces a full re-run from here."""
    idx = STAGE_ORDER.index(stage)
    for s in STAGE_ORDER[idx + 1 :]:
        if cp_mod.load(s, settings) is not None:
            cp_mod.reset(s, settings)
            log.warning("invalidated downstream checkpoint: %s", s)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: bary <stage> [options]\n\nstages:")
        for name in STAGES:
            print(f"  {name}")
        return 0 if argv else 1
    stage, rest = argv[0], argv[1:]
    if stage not in STAGES:
        print(f"unknown stage: {stage}", file=sys.stderr)
        return 2
    mod = importlib.import_module(STAGES[stage])
    mod.run(rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
