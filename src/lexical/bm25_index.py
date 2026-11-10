import json
from pathlib import Path

import bm25s
import numpy as np

from src.chunking.chunker import embedding_text
from src.lexical.tokenize import tokenize
from src.schemas.models import Chunk, Hit


class BM25Index:
    """bm25s index over `context + text` (so company/year tokens are searchable), with the same
    equality filters as the vector store."""

    def __init__(self, stopwords: str = "en", tokenizer: str = "v2", stem: bool = True):
        self.stopwords = stopwords
        self.tokenizer = tokenizer  # "v1" = bm25s default; "v2" = stemmed, numbers kept
        self.stem = stem
        self._bm25: bm25s.BM25 | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        texts = [embedding_text(c) for c in chunks]
        if self.tokenizer == "v2":
            tokens = [self._tokens(t) for t in texts]
        else:
            tokens = bm25s.tokenize(texts, stopwords=self.stopwords, show_progress=False)
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokens, show_progress=False)

    def _tokens(self, text: str) -> list[str]:
        return tokenize(text, self.stopwords, self.stem)

    def _mask(self, filters: dict | None) -> np.ndarray | None:
        if not filters:
            return None
        return np.array(
            [
                1.0 if all(getattr(c, k) == v for k, v in filters.items()) else 0.0
                for c in self._chunks
            ],
            dtype=np.float32,
        )

    def search(self, q: str, k: int, filters: dict | None = None) -> list[Hit]:
        if self._bm25 is None or not self._chunks:
            return []
        if self.tokenizer == "v2":
            query = [self._tokens(q)]
        else:
            query = bm25s.tokenize(
                q, stopwords=self.stopwords, return_ids=False, show_progress=False
            )
        vocab = self._bm25.vocab_dict
        query = [[t for t in query[0] if t in vocab]]
        if not query[0]:
            return []
        mask = self._mask(filters)
        docs, scores = self._bm25.retrieve(
            query, k=min(k, len(self._chunks)), weight_mask=mask, show_progress=False
        )
        hits = []
        for idx, score in zip(docs[0], scores[0], strict=True):
            if score <= 0:
                continue
            chunk = self._chunks[int(idx)]
            hits.append(Hit(chunk_id=chunk.id, score=float(score), chunk=chunk))
        return hits

    def count(self) -> int:
        return len(self._chunks)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(directory), show_progress=False)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(c.model_dump_json() + "\n")

    @classmethod
    def load(cls, directory: Path, stopwords: str = "en", tokenizer: str = "v1",
             stem: bool = True) -> "BM25Index":
        directory = Path(directory)
        idx = cls(stopwords, tokenizer, stem)
        idx._bm25 = bm25s.BM25.load(str(directory), load_corpus=False, show_progress=False)
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            idx._chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        return idx
