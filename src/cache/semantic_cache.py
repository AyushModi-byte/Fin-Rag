"""Answer cache: exact (normalised) match first, then embedding similarity >= cache.threshold.

In memory plus a JSON file (`cache.path`). A similarity hit also needs the same `scope` (company and
fiscal year found by the router): "FY2018 revenue" and "FY2019 revenue" embed almost identically but
have different answers. The file is dropped when the index version, prompt version or LLM model
changes (the `namespace`, see namespace()), so a stale answer is never served after a change.
"""
import hashlib
import json
import logging
import threading
from pathlib import Path

import numpy as np

from src.schemas.models import QueryResult

log = logging.getLogger("finrag.cache")


def _plain(obj):
    """SimpleNamespace config -> plain dicts, for hashing."""
    return {k: _plain(v) for k, v in vars(obj).items()} if hasattr(obj, "__dict__") else obj


def namespace(cfg) -> str:
    """Fingerprint of everything that shapes an answer. A cached result is only valid for the settings it was
    made with: index, retrieval, rerank (incl. top_n = how many passages the model sees), generation, model
    and `cache.version` (bump it when the result format or the citation parser changes)."""
    parts = {
        "index": cfg.index.version, "retrieval": _plain(cfg.retrieval), "rerank": _plain(cfg.rerank),
        "generation": _plain(cfg.generation), "model": cfg.llm.primary.model, "temperature": cfg.llm.temperature,
        "version": cfg.cache.version,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


def _scope_key(scope: dict | None) -> str:
    return json.dumps(scope or {}, sort_keys=True)


class SemanticCache:
    def __init__(self, cfg, embedder):
        self.threshold: float = cfg.cache.threshold
        self.path = Path(cfg.cache.path)
        self.embedder = embedder
        self.namespace = namespace(cfg)
        self._lock = threading.Lock()
        self._entries: list[dict] = []  # {q, scope, vec|None, result}
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("namespace") != self.namespace:
                return
            self._entries = [
                {"q": e["q"], "scope": e["scope"], "vec": e["vec"], "result": e["result"]}
                for e in data["entries"]
            ]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            self._entries = []  # missing or corrupt file: start empty

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"namespace": self.namespace, "entries": self._entries}), encoding="utf-8")
        tmp.replace(self.path)

    # ---- api ----
    def _embed(self, q: str) -> np.ndarray:
        v = np.asarray(self.embedder.embed_query(q), dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def get(self, q: str, scope: dict | None = None) -> QueryResult | None:
        key, sk = _norm(q), _scope_key(scope)
        with self._lock:
            entries = list(self._entries)
        for e in entries:
            if e["q"] == key and _scope_key(e["scope"]) == sk:
                return QueryResult.model_validate(e["result"])
        candidates = [e for e in entries if e["vec"] is not None and _scope_key(e["scope"]) == sk]
        if not candidates:
            return None
        qv = self._embed(q)
        sims = np.array([e["vec"] for e in candidates], dtype=np.float32) @ qv
        best = int(np.argmax(sims))
        if float(sims[best]) >= self.threshold:
            log.info("semantic cache hit (similarity %.3f)", float(sims[best]))
            return QueryResult.model_validate(candidates[best]["result"])
        return None

    def put(self, q: str, result: QueryResult, scope: dict | None = None) -> None:
        key, sk = _norm(q), _scope_key(scope)
        entry = {"q": key, "scope": scope or {}, "vec": self._embed(q).tolist(), "result": result.model_dump()}
        with self._lock:
            self._entries = [e for e in self._entries if not (e["q"] == key and _scope_key(e["scope"]) == sk)]
            self._entries.append(entry)
            self._save()
