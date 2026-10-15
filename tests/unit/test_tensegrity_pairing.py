"""Pair-classification concurrency keeps serial semantics.

``classify_pairs`` fans decision calls out to threads when ``workers > 1``;
the Kaggle tunnel path relies on that to keep the GPU busy. The result order
must stay exactly the submission order, otherwise the written BaryEdges (and
the ``parent_edge_id`` stamps) would depend on completion timing.
"""

from __future__ import annotations

import threading
import time

from tensegrity_pipeline.pairing import classify_pairs
from tensegrity_pipeline.tev1 import FakeDecision
from tensegrity_pipeline.tg_vec import ROLES


def _pairs(n: int) -> list[tuple[int, int, float]]:
    return [(i, i + 1, 0.8) for i in range(0, 2 * n, 2)]


def test_classify_pairs_matches_serial_result_for_any_worker_count():
    texts = [f"sentence number {i} content" for i in range(24)]
    pairs = _pairs(12)
    serial = list(classify_pairs(FakeDecision(), texts, pairs, workers=1))
    for workers in (2, 4, 8):
        parallel = list(classify_pairs(FakeDecision(), texts, pairs, workers=workers))
        assert parallel == serial
        assert len(parallel) == len(pairs)
        assert all(c["role"] in ROLES for c in parallel)


def test_classify_pairs_uses_threads_when_workers_gt_1():
    texts = [f"t{i}" for i in range(16)]
    pairs = _pairs(8)
    seen: set[int] = set()

    class Recording(FakeDecision):
        def decide(self, state, questions):
            seen.add(threading.get_ident())
            time.sleep(0.01)  # hold the worker so overlap is observable
            return super().decide(state, questions)

    out = list(classify_pairs(Recording(), texts, pairs, workers=4))
    assert len(out) == len(pairs)
    assert len(seen) > 1  # more than one thread actually did work
