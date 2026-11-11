"""Per-stage tracing: latency into a dict (QueryResult.latency_ms) and one JSON line per stage in the log.

    timings = {}
    with trace("retrieve", timings):
        ...
    timings["retrieve"]  # milliseconds, recorded even if the stage raises
"""
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("finrag.trace")


def setup_trace_logging(log_file: str) -> None:
    """Append trace lines (bare JSON, one per line) to `log_file`; idempotent per file."""
    path = str(Path(log_file).resolve())
    if not any(getattr(h, "_finrag_trace_console", False) for h in log.handlers):
        # container hosts only show stdout/stderr, so the same JSON lines go to the console too
        console = logging.StreamHandler()
        console._finrag_trace_console = True
        console.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(console)
        log.setLevel(logging.INFO)
    if any(getattr(h, "_finrag_trace_file", None) == path for h in log.handlers):
        return
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler._finrag_trace_file = path
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def log_event(event: str, **fields) -> None:
    """One JSON line. Callers must redact PII from `fields` first."""
    log.info(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str))


@contextmanager
def trace(stage: str, timings: dict | None = None):
    start = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - start) * 1000
        if timings is not None:
            timings[stage] = timings.get(stage, 0.0) + ms
        log.info(json.dumps({"event": "stage", "stage": stage, "ms": round(ms, 3)}))
