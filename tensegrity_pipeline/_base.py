"""Common entrypoint plumbing for the tensegrity pipeline stages.

Mirrors ``scripts/_base.py`` (stage-order guard, checkpoint handling, shared
CLI flags) but targets :class:`TensegritySettings` and the tensegrity stage
names. The build profile defaults to ``tensegrity`` (``.env.tensegrity``) so the
pipeline runs without sourcing a file by hand; ``--env poc`` still works.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from collections.abc import Sequence

from lib import checkpoint as cp_mod
from lib.checkpoint import Checkpoint
from lib.log import get_logger, setup_logging
from tensegrity_pipeline.config import TensegritySettings

# stage name → module path (run() entrypoint). Dict order == execution order.
STAGES: dict[str, str] = {
    "01_parse": "tensegrity_pipeline.stages.t01_parse",
    "02_embed": "tensegrity_pipeline.stages.t02_embed",
    "03_insert_nodes": "tensegrity_pipeline.stages.t03_insert_nodes",
    "04_l15_edges": "tensegrity_pipeline.stages.t04_l15_edges",
    "05_title_vectors": "tensegrity_pipeline.stages.t05_title_vectors",
    "06_l14_edges": "tensegrity_pipeline.stages.t06_l14_edges",
    "07_orphan_reentry": "tensegrity_pipeline.stages.t07_orphan_reentry",
    "08_metabary": "tensegrity_pipeline.stages.t08_metabary",
    "09_extend": "tensegrity_pipeline.stages.t09_extend",
    "10_index": "tensegrity_pipeline.stages.t10_index",
}

STAGE_ORDER: list[str] = list(STAGES)

DEFAULT_ENV = "tensegrity"


class StageOrderError(RuntimeError):
    """Raised when a stage is invoked before its prerequisite has completed."""


def _prev_stage(stage: str) -> str | None:
    idx = STAGE_ORDER.index(stage)
    return STAGE_ORDER[idx - 1] if idx > 0 else None


def make_parser(stage: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"tensegrity {stage}",
        description=f"BaryGraph tensegrity pipeline stage {stage}",
    )
    p.add_argument("--reset", action="store_true", help="discard checkpoint and start over")
    p.add_argument("--limit", type=int, default=None, help="process at most N items (dev)")
    p.add_argument("--dry-run", action="store_true", help="do not write to MongoDB / disk")
    p.add_argument("--batch-size", type=int, default=None, help="override BATCH_SIZE")
    p.add_argument(
        "--env",
        type=str,
        default=None,
        metavar="NAME",
        help="load .env.NAME (default: tensegrity, i.e. .env.tensegrity)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="bypass stage-order / already-done safeguards (use with care)",
    )
    p.add_argument(
        "--doc",
        type=str,
        default=None,
        action="append",
        metavar="PATH",
        help="(01_parse only) input text/markdown file; repeatable for multiple documents",
    )
    p.add_argument(
        "--level",
        type=int,
        default=None,
        choices=(15, 14),
        help="(04/06/07 only) which node level to operate on",
    )
    p.add_argument(
        "--window",
        type=int,
        default=None,
        help="(07_orphan_reentry only) orphans per pairing round",
    )
    p.add_argument(
        "--phase",
        choices=("pair", "absorb", "both"),
        default="both",
        help="(07_orphan_reentry only) which phase(s) to run (default: both)",
    )
    p.add_argument(
        "--max-level",
        type=int,
        default=13,
        help="(08_metabary / 09_extend only) stop building MBs below this level",
    )
    return p


def _enforce_order(
    stage: str, settings: TensegritySettings, force: bool, log: logging.Logger
) -> None:
    prev = _prev_stage(stage)
    if prev is None:
        return
    if cp_mod.is_done(prev, settings):
        return
    msg = (
        f"stage '{stage}' requires '{prev}' to be complete "
        f"({settings.pipeline_state_dir}/{prev}.json missing or done=false)"
    )
    if force:
        log.warning("STAGE-ORDER BYPASSED (--force): %s", msg)
        return
    raise StageOrderError(msg)


def bootstrap(
    stage: str, argv: Sequence[str] | None = None
) -> tuple[TensegritySettings, argparse.Namespace, logging.Logger, Checkpoint]:
    args = make_parser(stage).parse_args(argv)

    # Default to the tensegrity profile unless the caller picked another one or
    # BARY_ENV is already set in the environment.
    env = args.env or os.environ.get("BARY_ENV") or DEFAULT_ENV
    os.environ["BARY_ENV"] = env

    settings = TensegritySettings.load()
    setup_logging(settings.log_level)
    log = get_logger(f"tensegrity.{stage}")

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

    cp_mod.save(cp, settings)

    log.info(
        "settings: profile=%s db=%s coll=%s parsed=%s fake_embed=%s fake_decision=%s",
        settings.profile,
        settings.mongo_db,
        settings.mongo_collection,
        settings.parsed_dir,
        settings.fake_embed,
        settings.fake_decision,
    )
    return settings, args, log, cp


def finish(cp: Checkpoint, settings: TensegritySettings, log: logging.Logger) -> None:
    cp.mark_done()
    cp_mod.save(cp, settings)
    log.info("stage complete: processed=%d total=%d", cp.processed, cp.total)


def invalidate_downstream(
    stage: str, settings: TensegritySettings, log: logging.Logger
) -> None:
    idx = STAGE_ORDER.index(stage)
    for s in STAGE_ORDER[idx + 1 :]:
        if cp_mod.load(s, settings) is not None:
            cp_mod.reset(s, settings)
            log.warning("invalidated downstream checkpoint: %s", s)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: tensegrity <stage> [options]\n\nstages:")
        for name in STAGES:
            print(f"  {name}")
        return 0 if argv else 1
    stage, rest = argv[0], argv[1:]
    if stage not in STAGES:
        print(f"unknown stage: {stage}", file=sys.stderr)
        return 2
    for i, a in enumerate(rest):
        if a == "--env" and i + 1 < len(rest):
            os.environ["BARY_ENV"] = rest[i + 1]
        elif a.startswith("--env="):
            os.environ["BARY_ENV"] = a.split("=", 1)[1]
    mod = importlib.import_module(STAGES[stage])
    mod.run(rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
