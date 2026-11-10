import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from src.schemas.models import Chunk, Hit


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _qdrant_filter(filters: dict | None) -> models.Filter | None:
    if not filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in filters.items()
        ]
    )


class VectorStore:
    """Embedded (local-mode) Qdrant collection with cosine distance."""

    def __init__(self, path: Path, collection: str, dim: int):
        self.collection = collection
        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(path))
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    @contextmanager
    def bulk_load(self):
        """Embedded Qdrant commits (and fsyncs) once per point, which made a 16k-point build take
        hours on Windows (measured 1.3 points/s vs 2,300 points/s without). For a bulk load, relax
        sqlite durability; it is restored afterwards. Only for stores that are rebuilt from the
        embedding shards anyway: a crash mid-load means rebuild, never a half-trusted index.
        """
        conn = self._client._client.collections[self.collection].storage.storage
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA journal_mode=MEMORY")
        try:
            yield
        finally:
            conn.commit()
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=FULL")

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        points = [
            models.PointStruct(id=_point_id(c.id), vector=v.tolist(), payload=c.model_dump())
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(self.collection, points=points)

    def search(self, vec: np.ndarray, k: int, filters: dict | None = None) -> list[Hit]:
        res = self._client.query_points(
            self.collection,
            query=np.asarray(vec).tolist(),
            limit=k,
            query_filter=_qdrant_filter(filters),
            with_payload=True,
        )
        hits = []
        for p in res.points:
            chunk = Chunk(**p.payload)
            hits.append(Hit(chunk_id=chunk.id, score=float(p.score), chunk=chunk))
        return hits

    def count(self) -> int:
        return self._client.count(self.collection, exact=True).count

    def close(self) -> None:
        self._client.close()
