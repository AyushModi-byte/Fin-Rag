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
    # trace lines go to the log as bare JSON through their own handler (observability/tracing.py)
    handler.addFilter(lambda record: not record.name.startswith("finrag.trace"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_corpus(
    raw_dir: Path, docs_json: Path, parsed_dir: Path | None = None, reuse_parsed: bool = False
) -> list[tuple[DocMeta, list[Page]]]:
    """Parse every PDF listed in docs.json; write per-doc JSONL to parsed_dir if given.

    With reuse_parsed, an existing parsed JSONL is read instead of re-parsing the PDF (parsing the
    full corpus takes ~15 minutes).
    """
    raw_dir = Path(raw_dir)
    docs = json.loads(Path(docs_json).read_text(encoding="utf-8"))
    if parsed_dir is not None:
        Path(parsed_dir).mkdir(parents=True, exist_ok=True)
    corpus = []
    for doc_id, meta in docs.items():
        cached = Path(parsed_dir) / f"{doc_id}.jsonl" if parsed_dir is not None else None
        if reuse_parsed and cached is not None and cached.exists():
            lines = cached.read_text(encoding="utf-8").splitlines()
            corpus.append((DocMeta(**meta), [Page(**json.loads(l)) for l in lines if l.strip()]))
            continue
        pages = parse_pdf(raw_dir / f"{doc_id}.pdf", doc_id=doc_id)
        if parsed_dir is not None:
            out = Path(parsed_dir) / f"{doc_id}.jsonl"
            out.write_text(
                "\n".join(p.model_dump_json() for p in pages) + "\n", encoding="utf-8"
            )
        corpus.append((DocMeta(**meta), pages))
    return corpus
