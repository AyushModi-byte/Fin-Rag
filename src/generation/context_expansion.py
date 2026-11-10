"""Parent expansion for generation (experiment 5c).

Retrieval returns table ROW GROUPS (fragments of a big statement). The LLM answers better from the
whole table, so each retrieved row-group chunk is replaced by its full table parent:

* one table appears once (the first retrieved row group of a table decides its position),
* the context never grows past `max_tokens` (regex token count, same measure as elsewhere): a table
  that would not fit is left as the retrieved fragment,
* chunks that are not part of a split table (prose, whole tables) are untouched.

Retrieval ranking and metrics are not affected; only the text given to the LLM changes.
"""
from src.chunking.chunker import count_tokens
from src.schemas.models import Hit, Parent


def expand_context(hits: list[Hit], parents: dict[str, Parent], max_tokens: int) -> list[Hit]:
    if not parents:
        return hits
    total = sum(count_tokens(h.chunk.text) for h in hits)
    expanded: set[str] = set()
    out: list[Hit] = []
    for h in hits:
        tid = h.chunk.table_id
        parent = parents.get(tid) if tid else None
        if parent is None:
            out.append(h)
            continue
        if tid in expanded:
            total -= count_tokens(h.chunk.text)  # its content is already inside the full table
            continue
        delta = count_tokens(parent.text) - count_tokens(h.chunk.text)
        if total + delta > max_tokens:
            out.append(h)  # does not fit: keep the fragment
            continue
        total += delta
        expanded.add(tid)
        out.append(Hit(chunk_id=h.chunk_id, score=h.score, chunk=h.chunk.model_copy(update={"text": parent.text})))
    return out
