import numpy as np
from fastembed import TextEmbedding


class Embedder:
    """fastembed wrapper (CPU/ONNX). Records the model id so indexes can refuse a mismatch.

    Speed settings come from config (Phase 0 benchmark on this CPU: batch 1 + length sorting was
    3.7x faster than batch 32, because padding to the longest text in a batch wastes work).
    """

    def __init__(self, cfg):
        e = cfg.embeddings
        self.model_id: str = e.model
        self.batch_size: int = e.batch_size
        self.sort_by_length: bool = getattr(e, "sort_by_length", False)
        self.dim: int = TextEmbedding.get_embedding_size(self.model_id)
        self._model = TextEmbedding(model_name=self.model_id, threads=getattr(e, "threads", None))

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        order = (
            sorted(range(len(texts)), key=lambda i: len(texts[i]))
            if self.sort_by_length
            else list(range(len(texts)))
        )
        vecs = self._model.embed([texts[i] for i in order], batch_size=self.batch_size)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for pos, v in zip(order, vecs, strict=True):
            out[pos] = v
        return out
