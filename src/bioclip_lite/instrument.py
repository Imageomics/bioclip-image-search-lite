"""Lightweight phase timing + capture for latency instrumentation.

Two uses from one API:

1. **Always-on logging.** Wrap any block in ``with phase("name"): ...`` and
   it logs ``⏱ name   12.3 ms`` at INFO. Running the Gradio app shows a clean
   per-phase breakdown of every search with no extra setup.

2. **Opt-in capture.** A benchmark wraps a request in ``with collect() as recs:``
   and every ``phase()`` that runs *on the same thread* appends a structured
   record ``{"phase", "ms", **fields}`` to ``recs``. No capture context → just
   logging, zero overhead beyond a thread-local lookup.

Capture is thread-local: orchestration phases (embed, search, fetch, render)
run on the calling thread and are captured. Work inside a ThreadPoolExecutor
(per-image fetches) is *not* auto-captured — time those at the pool boundary,
or have the worker return its own timing.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger("bioclip_lite.timing")

_local = threading.local()


def _sink() -> Optional[List[Dict]]:
    return getattr(_local, "sink", None)


@contextmanager
def collect() -> Iterator[List[Dict]]:
    """Activate timing capture for this thread; yields the record list.

    Nestable: restores the previous sink on exit, so an inner ``collect()``
    doesn't leak records into an outer one.
    """
    prev = getattr(_local, "sink", None)
    recs: List[Dict] = []
    _local.sink = recs
    try:
        yield recs
    finally:
        _local.sink = prev


@contextmanager
def phase(name: str, log: bool = True, **fields) -> Iterator[Dict]:
    """Time a block. Logs at INFO and, if a ``collect()`` is active, records it.

    Extra kwargs are attached to the record and appended to the log line
    (e.g. ``phase("image_fetch", k=20, variant="medium")``). The yielded dict
    can be mutated inside the block to add fields known only after the work
    runs (e.g. bytes fetched, n_ok).
    """
    rec: Dict = {"phase": name, **fields}
    t0 = time.perf_counter()
    err: Optional[str] = None
    try:
        yield rec
    except Exception as e:  # noqa: BLE001 - we re-raise; just annotating
        err = repr(e)
        raise
    finally:
        rec["ms"] = (time.perf_counter() - t0) * 1000.0
        if err is not None:
            rec["error"] = err
        if log:
            extra = "".join(f" {k}={v}" for k, v in rec.items()
                            if k not in ("phase", "ms"))
            logger.info("⏱ %-18s %8.1f ms%s", name, rec["ms"], extra)
        sink = _sink()
        if sink is not None:
            sink.append(rec)
