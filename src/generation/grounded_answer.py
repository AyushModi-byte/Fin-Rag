import re
import time

from src.prompts.prompt_templates import IDK, SYSTEM_GROUNDED, build_grounded_prompt
from src.schemas.models import Citation, Hit, QueryResult

# [doc_id p.60]  or  [doc_id p.61, p.62]
_CITATION = re.compile(r"\[([^\[\]\s]+)\s+((?:p\.?\s*\d+\s*[,;]?\s*)+)\]")
_PAGE = re.compile(r"p\.?\s*(\d+)")


def _is_idk(answer: str) -> bool:
    normalized = answer.strip().rstrip(".").strip().lower().replace("’", "'")
    return normalized == IDK.lower()


def parse_citations(answer: str, hits: list[Hit]) -> list[Citation]:
    """Citations in the answer, deduplicated, keeping only pages that were in the context."""
    allowed = {(h.chunk.doc_id, h.chunk.page) for h in hits}
    seen: set[tuple[str, int]] = set()
    citations = []
    for m in _CITATION.finditer(answer):
        for page in _PAGE.findall(m.group(2)):
            ref = (m.group(1), int(page))
            if ref in allowed and ref not in seen:
                seen.add(ref)
                citations.append(Citation(doc_id=ref[0], page=ref[1]))
    return citations


def generate(q: str, hits: list[Hit], llm) -> QueryResult:
    """Grounded answer with [doc_id p.N] citations, or exactly "I don't know".

    With no retrieved context no LLM call is made.
    """
    if not hits:
        return QueryResult(answer=IDK, citations=[], contexts=[], latency_ms={"llm": 0.0})
    start = time.perf_counter()
    raw = llm.complete(build_grounded_prompt(q, hits), system=SYSTEM_GROUNDED)
    elapsed_ms = (time.perf_counter() - start) * 1000
    contexts = [h.chunk.text for h in hits]
    if _is_idk(raw):
        return QueryResult(answer=IDK, citations=[], contexts=contexts, latency_ms={"llm": elapsed_ms})
    answer = raw.strip()
    return QueryResult(
        answer=answer,
        citations=parse_citations(answer, hits),
        contexts=contexts,
        latency_ms={"llm": elapsed_ms},
    )
