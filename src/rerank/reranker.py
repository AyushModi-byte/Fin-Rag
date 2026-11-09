from fastembed.rerank.cross_encoder import TextCrossEncoder

from src.chunking.chunker import embedding_text
from src.schemas.models import Hit


class Reranker:
    """fastembed cross-encoder (CPU/ONNX). Scores (query, chunk) pairs jointly."""

    def __init__(self, cfg):
        self.model_id: str = cfg.rerank.model
        self._model = TextCrossEncoder(model_name=self.model_id)

    def rerank(self, q: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        scores = list(self._model.rerank(q, [embedding_text(h.chunk) for h in hits]))
        rescored = [
            Hit(chunk_id=h.chunk_id, score=float(s), chunk=h.chunk)
            for h, s in zip(hits, scores, strict=True)
        ]
        rescored.sort(key=lambda h: -h.score)
        return rescored[:top_n]
