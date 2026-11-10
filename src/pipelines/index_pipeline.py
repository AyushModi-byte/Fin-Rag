"""Offline pipeline: parsed pages -> chunks -> dense (Qdrant) + BM25 indexes under
`<cfg.index.dir>/<cfg.index.version>/`. Orchestration only.

The build is RESUMABLE. Embeddings are written to `shards/shard_NNNNN.npz` (about
`index.shard_size` chunks each, atomically). A restart skips every shard whose signature (embedder,
prefixes, chunk ids and texts) still matches. Qdrant and BM25 are then built from the shards, so
re-running on a finished index costs seconds, not an embedding pass. `manifest.json` is written
last: no manifest means an incomplete build and `load_index` refuses it.

`index.protected_versions` (e.g. v1, the rollback) can never be built into.
"""
import hashlib
import json
import logging
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.chunking.chunker import chunk_pages, embedding_text
from src.config import load_config
from src.embeddings.embedder import Embedder
from src.embeddings.tokenizer import EmbedTokenizer
from src.ingestion.loader import load_corpus
from src.lexical.bm25_index import BM25Index
from src.schemas.models import Chunk, Parent
from src.vector_db.vector_store import VectorStore

log = logging.getLogger("finrag.index")


class IndexMismatchError(RuntimeError):
    """The on-disk index was built with a different embedding model, version or chunking."""


class ReindexTooSlow(RuntimeError):
    """Projected embedding time exceeds `reindex.max_hours`."""


def index_path(cfg) -> Path:
    return Path(cfg.index.dir) / cfg.index.version


def _sha1(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _shard_signature(embedder, shard: list[Chunk]) -> str:
    body = [f"{c.id}:{_sha1(embedding_text(c))}" for c in shard]
    return _sha1(embedder.model_id, embedder.doc_prefix, str(embedder.dim), *body)


def project_hours(chunks: list[Chunk], embedder, sample: int, seed: int) -> float:
    """Time the embedder on a random sample and scale by text length to the whole corpus."""
    rng = random.Random(seed)
    pick = rng.sample(chunks, min(sample, len(chunks)))
    texts = [embedding_text(c) for c in pick]
    t0 = time.perf_counter()
    embedder.embed_documents(texts)
    secs = time.perf_counter() - t0
    scale = sum(len(embedding_text(c)) for c in chunks) / max(sum(len(t) for t in texts), 1)
    return secs * scale / 3600


def _write_shard(path: Path, sig: str, ids: list[str], vecs: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, sig=sig, ids=np.array(ids), v=vecs)
    tmp.replace(path)  # atomic: a half-written shard never has the final name


def _load_shard(path: Path, sig: str) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        return z["v"] if str(z["sig"]) == sig else None
    except Exception:  # corrupt or foreign file: recompute
        return None


def embed_in_shards(chunks: list[Chunk], embedder, shard_dir: Path, shard_size: int) -> np.ndarray:
    shard_dir.mkdir(parents=True, exist_ok=True)
    n = len(chunks)
    parts: list[np.ndarray] = []
    done = embedded_now = 0
    t0 = time.perf_counter()
    for i, lo in enumerate(range(0, n, shard_size)):
        shard = chunks[lo : lo + shard_size]
        sig = _shard_signature(embedder, shard)
        path = shard_dir / f"shard_{i:05d}.npz"
        vecs = _load_shard(path, sig)
        if vecs is None:
            vecs = embedder.embed_documents([embedding_text(c) for c in shard])
            _write_shard(path, sig, [c.id for c in shard], vecs)
            embedded_now += len(shard)
            status = "embedded"
        else:
            status = "reused"
        parts.append(vecs)
        done += len(shard)
        elapsed = time.perf_counter() - t0
        rate = embedded_now / elapsed if embedded_now and elapsed > 0 else 0.0
        eta = (n - done) / rate if rate else 0.0
        log.info("shard %d %s | %d/%d chunks | %.1f chunks/s | ETA %.0f s", i, status, done, n, rate, eta)
        print(f"[index] shard {i} {status}: {done}/{n} chunks, {rate:.1f} chunks/s, ETA {eta/60:.1f} min",
              flush=True)
    return np.concatenate(parts) if parts else np.zeros((0, embedder.dim), dtype=np.float32)


def _chunking_params(cfg) -> dict:
    return dict(vars(cfg.chunking))


def index_chunks(chunks: list[Chunk], cfg, embedder: Embedder | None = None,
                 parents: list[Parent] | None = None, tokenizer_info: dict | None = None) -> dict:
    """Build (or resume) the index for the configured version and write the manifest."""
    version = cfg.index.version
    if version in getattr(cfg.index, "protected_versions", []):
        raise IndexMismatchError(f"index version {version!r} is protected (rollback); refusing to build")
    embedder = embedder or Embedder(cfg)
    root = index_path(cfg)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").unlink(missing_ok=True)  # incomplete until the very end
    started = time.perf_counter()

    shard_size = getattr(cfg.index, "shard_size", 500)
    max_hours = getattr(getattr(cfg, "reindex", None), "max_hours", None)
    shard_dir = root / "shards"
    pending = any(
        _load_shard(shard_dir / f"shard_{i:05d}.npz", _shard_signature(embedder, chunks[lo : lo + shard_size]))
        is None
        for i, lo in enumerate(range(0, len(chunks), shard_size))
    )
    if pending and max_hours:
        projected = project_hours(chunks, embedder, cfg.reindex.probe_chunks, cfg.eval.seed)
        log.info("projected embedding time %.2f h (gate %.2f h)", projected, max_hours)
        print(f"[index] projected full embedding time: {projected:.2f} h (gate {max_hours} h)", flush=True)
        if projected > max_hours:
            raise ReindexTooSlow(f"projected {projected:.2f} h exceeds reindex.max_hours={max_hours}")

    vectors = embed_in_shards(chunks, embedder, shard_dir, shard_size)

    qdir = root / "qdrant"
    if qdir.exists():
        shutil.rmtree(qdir)  # rebuilt from the shards; point ids are deterministic
    store = VectorStore(qdir, cfg.index.collection, embedder.dim)
    try:
        batch = 1000
        for lo in range(0, len(chunks), batch):
            store.upsert(chunks[lo : lo + batch], vectors[lo : lo + batch])
        dense_count = store.count()
    finally:
        store.close()

    bm25 = BM25Index(cfg.lexical.stopwords, cfg.lexical.tokenizer, cfg.lexical.stem)
    bm25.build(chunks)
    bm25_dir = root / "bm25"
    if bm25_dir.exists():
        shutil.rmtree(bm25_dir)
    bm25.save(bm25_dir)

    if parents is not None:
        with (root / "parents.jsonl").open("w", encoding="utf-8") as f:
            for p in parents:
                f.write(p.model_dump_json() + "\n")

    manifest = {
        "index_version": version,
        "embedding_model": embedder.model_id,
        "embedding_dim": embedder.dim,
        "doc_prefix": embedder.doc_prefix,
        "query_prefix": embedder.query_prefix,
        "tokenizer": tokenizer_info or {},
        "chunking": _chunking_params(cfg),
        "collection": cfg.index.collection,
        "lexical_tokenizer": cfg.lexical.tokenizer,
        "lexical_stem": cfg.lexical.stem,
        "chunks": len(chunks),
        "dense_count": dense_count,
        "parents": len(parents) if parents is not None else None,
        "build_seconds": round(time.perf_counter() - started, 1),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _manifest(cfg) -> dict:
    path = index_path(cfg) / "manifest.json"
    if not path.exists():
        raise IndexMismatchError(f"no manifest at {path}: index missing or its build is incomplete")
    return json.loads(path.read_text(encoding="utf-8"))


def load_index(cfg) -> tuple[VectorStore, BM25Index]:
    """Open the index, refusing if it was built with a different model, version or chunking."""
    root = index_path(cfg)
    manifest = _manifest(cfg)
    if manifest["embedding_model"] != cfg.embeddings.model:
        raise IndexMismatchError(
            f"index built with embedding model {manifest['embedding_model']!r}, "
            f"config has {cfg.embeddings.model!r}; rebuild the index"
        )
    if manifest["index_version"] != cfg.index.version:
        raise IndexMismatchError(
            f"index version {manifest['index_version']!r} != config {cfg.index.version!r}; "
            "rebuild the index"
        )
    # v1 manifests carry none of the keys below, so v1 stays loadable as the rollback
    want_prefix = getattr(cfg.embeddings, "doc_prefix", "") or ""
    if "doc_prefix" in manifest and manifest["doc_prefix"] != want_prefix:
        raise IndexMismatchError(
            f"index doc_prefix {manifest['doc_prefix']!r} != config {want_prefix!r}; rebuild the index"
        )
    if "chunking" in manifest:
        want = _chunking_params(cfg)
        diff = {k: (manifest["chunking"].get(k), v) for k, v in want.items() if manifest["chunking"].get(k) != v}
        if diff:
            raise IndexMismatchError(f"index chunking params differ from config (index, config): {diff}")
    if manifest.get("lexical_tokenizer", "v1") != cfg.lexical.tokenizer:
        raise IndexMismatchError(
            f"index BM25 tokenizer {manifest.get('lexical_tokenizer', 'v1')!r} != config "
            f"{cfg.lexical.tokenizer!r}; rebuild the index"
        )
    store = VectorStore(root / "qdrant", manifest["collection"], manifest["embedding_dim"])
    bm25 = BM25Index.load(
        root / "bm25", cfg.lexical.stopwords, manifest.get("lexical_tokenizer", "v1"),
        manifest.get("lexical_stem", False),
    )
    return store, bm25


def load_parents(cfg) -> dict[str, Parent]:
    """Parent pages and full tables by id (v2 indexes only)."""
    path = index_path(cfg) / "parents.jsonl"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {p.id: p for p in (Parent(**json.loads(line)) for line in f if line.strip())}


def build_index(cfg=None) -> dict:
    """Chunk the parsed corpus (re-using data/parsed) and build (or resume) both indexes."""
    cfg = cfg or load_config()
    corpus = load_corpus(
        Path(cfg.data.raw_dir),
        Path(cfg.data.eval_dir) / "docs.json",
        Path(cfg.data.parsed_dir),
        reuse_parsed=True,
    )
    tok = EmbedTokenizer(cfg.embeddings.model)
    chunks: list[Chunk] = []
    parents: list[Parent] = []
    for meta, pages in corpus:
        doc_parents, doc_chunks = chunk_pages(pages, meta, cfg, tok=tok)
        parents.extend(doc_parents)
        chunks.extend(doc_chunks)
    manifest = index_chunks(chunks, cfg, parents=parents, tokenizer_info=tok.info())
    return {"docs": len(corpus), **manifest}
