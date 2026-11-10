"""Offline pipeline: parsed pages -> chunks -> dense (Qdrant) + BM25 indexes under
`<cfg.index.dir>/<cfg.index.version>/`. Orchestration only."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.chunking.chunker import chunk_pages, embedding_text
from src.config import load_config
from src.embeddings.embedder import Embedder
from src.embeddings.tokenizer import EmbedTokenizer
from src.ingestion.loader import load_corpus
from src.lexical.bm25_index import BM25Index
from src.schemas.models import Chunk
from src.vector_db.vector_store import VectorStore


class IndexMismatchError(RuntimeError):
    """The on-disk index was built with a different embedding model or index version."""


def index_path(cfg) -> Path:
    return Path(cfg.index.dir) / cfg.index.version


def index_chunks(chunks: list[Chunk], cfg, embedder: Embedder | None = None) -> dict:
    """Rebuild the index from scratch (so rebuilding is idempotent) and write the manifest."""
    embedder = embedder or Embedder(cfg)
    root = index_path(cfg)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    vectors = embedder.embed([embedding_text(c) for c in chunks])
    store = VectorStore(root / "qdrant", cfg.index.collection, embedder.dim)
    try:
        store.upsert(chunks, vectors)
        dense_count = store.count()
    finally:
        store.close()

    bm25 = BM25Index(cfg.lexical.stopwords, cfg.lexical.tokenizer, cfg.lexical.stem)
    bm25.build(chunks)
    bm25.save(root / "bm25")

    manifest = {
        "index_version": cfg.index.version,
        "embedding_model": embedder.model_id,
        "embedding_dim": embedder.dim,
        "collection": cfg.index.collection,
        "lexical_tokenizer": cfg.lexical.tokenizer,
        "lexical_stem": cfg.lexical.stem,
        "chunks": len(chunks),
        "dense_count": dense_count,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_index(cfg) -> tuple[VectorStore, BM25Index]:
    """Open the index, refusing if it was built with a different model or version."""
    root = index_path(cfg)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
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
    store = VectorStore(root / "qdrant", manifest["collection"], manifest["embedding_dim"])
    # an index without these keys is v1: bm25s default tokenizer, no stemming
    bm25 = BM25Index.load(
        root / "bm25", cfg.lexical.stopwords, manifest.get("lexical_tokenizer", "v1"),
        manifest.get("lexical_stem", False),
    )
    return store, bm25


def build_index(cfg=None) -> dict:
    """Chunk the parsed corpus (re-using data/parsed) and build both indexes."""
    cfg = cfg or load_config()
    corpus = load_corpus(
        Path(cfg.data.raw_dir),
        Path(cfg.data.eval_dir) / "docs.json",
        Path(cfg.data.parsed_dir),
        reuse_parsed=True,
    )
    tok = EmbedTokenizer(cfg.embeddings.model)
    chunks: list[Chunk] = []
    for meta, pages in corpus:
        _, doc_chunks = chunk_pages(pages, meta, cfg, tok=tok)
        chunks.extend(doc_chunks)
    manifest = index_chunks(chunks, cfg)
    return {"docs": len(corpus), **manifest}
