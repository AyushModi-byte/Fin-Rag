"""CLI: python -m src.jobs.index_corpus"""
import time

from src.config import load_config
from src.ingestion.loader import setup_logging
from src.pipelines.index_pipeline import build_index


def _keep_awake() -> None:
    """Windows: stop the machine sleeping while the build runs (process-scoped, auto-reverts)."""
    try:
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS|SYSTEM
    except Exception:
        pass


def main() -> dict:
    _keep_awake()
    cfg = load_config()
    setup_logging(cfg.data.log_file)
    start = time.time()
    summary = build_index(cfg)
    summary["seconds"] = round(time.time() - start, 1)
    print(summary)
    return summary


if __name__ == "__main__":
    main()
