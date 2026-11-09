"""CLI: python -m src.jobs.index_corpus"""
import time

from src.config import load_config
from src.ingestion.loader import setup_logging
from src.pipelines.index_pipeline import build_index


def main() -> dict:
    cfg = load_config()
    setup_logging(cfg.data.log_file)
    start = time.time()
    summary = build_index(cfg)
    summary["seconds"] = round(time.time() - start, 1)
    print(summary)
    return summary


if __name__ == "__main__":
    main()
