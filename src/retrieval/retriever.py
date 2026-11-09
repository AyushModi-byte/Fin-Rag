from src.schemas.models import Hit


def rrf(result_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal rank fusion. score(d) = sum over lists of 1 / (k + rank), rank is 1-based.

    Returns the deduplicated union sorted by fused score (the Hit.score becomes the fused score).
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda cid: -scores[cid])
    return [
        Hit(chunk_id=cid, score=scores[cid], chunk=first_seen[cid].chunk) for cid in ordered
    ]


class HybridRetriever:
    """Dense + BM25 retrieval, same filters on both, fused with RRF."""

    def __init__(self, store, bm25, embedder, cfg):
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.rrf_k: int = cfg.retrieval.rrf_k

    def retrieve(self, q: str, k: int, filters: dict | None = None) -> list[Hit]:
        vec = self.embedder.embed([q])[0]
        dense = self.store.search(vec, k, filters)
        lexical = self.bm25.search(q, k, filters)
        return rrf([dense, lexical], k=self.rrf_k)[:k]
