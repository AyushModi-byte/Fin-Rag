import re
import time

from src.prompts.prompt_templates import IDK, SYSTEM_GROUNDED, build_grounded_prompt
from src.schemas.models import Citation, Hit, QueryResult

# The prompt asks for [doc_id p.60], but the model also writes U+3010/3011 corner brackets around it (the most
# common variant in the saved eval answers), U+FF3B/FF3D square brackets and (...), with stray spaces (including the narrow
# no-break space U+202F), page lists "p.61, p.62" / "p.61; p.62", ranges "p.60-62" (hyphen, U+2011, en dash)
# and several documents in one bracket "[A p.1; B p.2]". Anything inside a bracket group that reads
# "<doc_id> p.N..." is a citation; the answer text itself is never rewritten.
_GROUP = re.compile(r"[\[\u3010\uff3b\u3014(]([^\[\]\u3010\u3011\uff3b\uff3d\u3014\u3015()]*)[\]\u3011\uff3d\u3015)]")
_DASH = "-\u2010\u2011\u2012\u2013\u2014"  # hyphen, hyphen, non-breaking hyphen, figure dash, en dash, em dash
_RANGE = rf"\d+(?:\s*[{_DASH}]\s*\d+)?"
_PAGES = rf"pp?\.?\s*{_RANGE}(?:\s*[,;&]\s*(?:pp?\.?\s*)?{_RANGE})*"
_SEGMENT = re.compile(rf"([^\s,;]+)\s+({_PAGES})")
_PAGE_ITEM = re.compile(rf"(\d+)(?:\s*[{_DASH}]\s*(\d+))?")


def _is_idk(answer: str) -> bool:
    normalized = answer.strip().rstrip(".").strip().lower().replace("’", "'")
    return normalized == IDK.lower()


def parse_citations(answer: str, hits: list[Hit]) -> list[Citation]:
    """Citations in the answer, deduplicated, in order of appearance, keeping only pages that were in the
    context. A page range is expanded to the retrieved pages inside it (never to pages that were not)."""
    allowed: dict[str, set[int]] = {}
    for h in hits:
        allowed.setdefault(h.chunk.doc_id, set()).add(h.chunk.page)
    seen: set[tuple[str, int]] = set()
    citations = []
    for group in _GROUP.finditer(answer):
        for seg in _SEGMENT.finditer(group.group(1)):
            doc_id = seg.group(1)
            pages = allowed.get(doc_id)
            if not pages:
                continue
            for item in _PAGE_ITEM.finditer(seg.group(2)):
                start = int(item.group(1))
                end = int(item.group(2)) if item.group(2) else start
                for page in sorted(p for p in pages if start <= p <= max(start, end)):
                    if (doc_id, page) not in seen:
                        seen.add((doc_id, page))
                        citations.append(Citation(doc_id=doc_id, page=page))
    return citations


def generate(q: str, hits: list[Hit], llm, system: str = SYSTEM_GROUNDED) -> QueryResult:
    """Grounded answer with [doc_id p.N] citations, or exactly "I don't know".

    With no retrieved context no LLM call is made.
    """
    if not hits:
        return QueryResult(answer=IDK, citations=[], contexts=[], latency_ms={"llm": 0.0})
    start = time.perf_counter()
    raw = llm.complete(build_grounded_prompt(q, hits), system=system)
    elapsed_ms = (time.perf_counter() - start) * 1000
    contexts = [h.chunk.text for h in hits]
    sources = [Citation(doc_id=h.chunk.doc_id, page=h.chunk.page) for h in hits]
    if _is_idk(raw):
        return QueryResult(
            answer=IDK, citations=[], contexts=contexts, sources=sources, latency_ms={"llm": elapsed_ms}
        )
    answer = raw.strip()
    return QueryResult(
        answer=answer,
        citations=parse_citations(answer, hits),
        contexts=contexts,
        sources=sources,
        latency_ms={"llm": elapsed_ms},
    )
