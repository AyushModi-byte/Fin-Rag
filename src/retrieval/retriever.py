from src.chunking.chunker import count_tokens
from src.schemas.models import Hit


def rrf(result_lists: list[list[Hit]], k: int = 60, weights: list[float] | None = None) -> list[Hit]:
    """Reciprocal rank fusion. score(d) = sum over lists of weight / (k + rank), rank is 1-based
    (weights default to 1 for every list).

    Returns the deduplicated union sorted by fused score (the Hit.score becomes the fused score).
    """
    weights = weights if weights is not None else [1.0] * len(result_lists)
    scores: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for hits, w in zip(result_lists, weights, strict=True):
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + w / (k + rank)
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
        self.weights = [cfg.retrieval.dense_weight, cfg.retrieval.bm25_weight]
        self.min_chunk_tokens: int = cfg.retrieval.min_chunk_tokens  # 0 = keep every chunk

    def retrieve(self, q: str, k: int, filters: dict | None = None) -> list[Hit]:
        vec = self.embedder.embed_query(q)  # applies embeddings.query_prefix
        dense = self._drop_tiny(self.store.search(vec, k, filters))
        lexical = self._drop_tiny(self.bm25.search(q, k, filters))
        return rrf([dense, lexical], k=self.rrf_k, weights=self.weights)[:k]

    def _drop_tiny(self, hits: list[Hit]) -> list[Hit]:
        """Page-number and header fragments match any company/year query through the context
        prefix; drop chunks with fewer than min_chunk_tokens tokens before fusing."""
        if self.min_chunk_tokens <= 0:
            return hits
        return [h for h in hits if count_tokens(h.chunk.text) >= self.min_chunk_tokens]
