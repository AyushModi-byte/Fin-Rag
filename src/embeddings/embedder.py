import numpy as np
from fastembed import TextEmbedding


class Embedder:
    """fastembed wrapper (CPU/ONNX). Records the model id so indexes can refuse a mismatch."""

    def __init__(self, cfg):
        self.model_id: str = cfg.embeddings.model
        self.batch_size: int = cfg.embeddings.batch_size
        self.dim: int = TextEmbedding.get_embedding_size(self.model_id)
        self._model = TextEmbedding(model_name=self.model_id)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.embed(texts, batch_size=self.batch_size)
        return np.asarray(list(vecs), dtype=np.float32)
