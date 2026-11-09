import logging
from pathlib import Path

import pymupdf4llm

from src.schemas.models import Page

logger = logging.getLogger("finrag.ingestion")


def parse_pdf(path: Path, doc_id: str | None = None) -> list[Page]:
    """Per-page markdown (tables kept as markdown). Empty pages are skipped and logged.

    Page numbers are 1-based and match the PDF page order. No OCR: image-only pages are empty.
    """
    path = Path(path)
    doc_id = doc_id or path.stem
    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True, use_ocr=False)
    pages = []
    for number, chunk in enumerate(chunks, start=1):
        text = chunk["text"].strip()
        if not text:
            logger.warning("skipping empty page %s p.%d", doc_id, number)
            continue
        pages.append(Page(doc_id=doc_id, page=number, markdown=text))
    return pages
