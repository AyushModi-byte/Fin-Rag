import json
import logging
from pathlib import Path

from src.ingestion.parsers import parse_pdf
from src.schemas.models import DocMeta, Page


def setup_logging(log_file: str) -> None:
    """Send finrag.* logs to a file (and console) without duplicating handlers."""
    logger = logging.getLogger("finrag")
    if any(getattr(h, "_finrag_file", False) for h in logger.handlers):
        return
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler._finrag_file = True
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_corpus(
    raw_dir: Path, docs_json: Path, parsed_dir: Path | None = None
) -> list[tuple[DocMeta, list[Page]]]:
    """Parse every PDF listed in docs.json; write per-doc JSONL to parsed_dir if given."""
    raw_dir = Path(raw_dir)
    docs = json.loads(Path(docs_json).read_text(encoding="utf-8"))
    if parsed_dir is not None:
        Path(parsed_dir).mkdir(parents=True, exist_ok=True)
    corpus = []
    for doc_id, meta in docs.items():
        pages = parse_pdf(raw_dir / f"{doc_id}.pdf", doc_id=doc_id)
        if parsed_dir is not None:
            out = Path(parsed_dir) / f"{doc_id}.jsonl"
            out.write_text(
                "\n".join(p.model_dump_json() for p in pages) + "\n", encoding="utf-8"
            )
        corpus.append((DocMeta(**meta), pages))
    return corpus
