import json
from pathlib import Path

import pytest

from src.chunking.chunker import embedding_text
from src.config import load_config
from src.embeddings.embedder import Embedder
from src.lexical.bm25_index import BM25Index
from src.pipelines.index_pipeline import IndexMismatchError, index_chunks, load_index
from src.schemas.models import Chunk
from src.vector_db.vector_store import VectorStore

TEXTS = [
    ("3M", 2018, "Purchases of property, plant and equipment were 1,577 million in the cash flow statement."),
    ("3M", 2018, "The company manufactures adhesives, abrasives and Post-it notes for consumers."),
    ("Adobe", 2017, "Creative Cloud subscriptions drove digital media segment revenue growth."),
    ("Boeing", 2018, "Commercial airplanes deliveries of the 737 and 787 programs increased."),
    ("Verizon", 2021, "Wireless retail postpaid connections and fiber broadband expansion continued."),
]


def make_chunks() -> list[Chunk]:
    chunks = []
    for i, (company, year, text) in enumerate(TEXTS):
        chunks.append(
            Chunk(
                id=f"chunk{i}", doc_id=f"{company}_{year}_10K", parent_id=f"p{i}", text=text,
                context=f"{company} {year} 10-K, page {i + 1}", page=i + 1,
                company=company, fiscal_year=year,
            )
        )
    return chunks


@pytest.fixture(scope="module")
def embedder():
    return Embedder(load_config())


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.index.dir = str(tmp_path / "index")
    c.index.version = "vtest"  # v1 is protected (rollback)
    return c


@pytest.fixture
def built(cfg, embedder):
    chunks = make_chunks()
    index_chunks(chunks, cfg, embedder=embedder)
    store, bm25 = load_index(cfg)
    yield store, bm25, chunks
    store.close()


def test_embedder_records_model_id_and_shape(embedder):
    vecs = embedder.embed(["hello world", "another sentence"])
    assert embedder.model_id == "BAAI/bge-small-en-v1.5"
    assert vecs.shape == (2, embedder.dim)


def test_dense_search_finds_exact_chunk(built, embedder):
    store, _, chunks = built
    target = chunks[3]
    vec = embedder.embed([embedding_text(target)])[0]
    hits = store.search(vec, k=3, filters=None)
    assert hits[0].chunk_id == target.id
    assert hits[0].chunk == target


def test_bm25_finds_keyword_chunk(built):
    _, bm25, chunks = built
    hits = bm25.search("Creative Cloud subscriptions", k=3, filters=None)
    assert hits[0].chunk_id == chunks[2].id


def test_company_filter_excludes_other_companies(built, embedder):
    store, bm25, _ = built
    vec = embedder.embed(["company revenue and operations"])[0]
    dense = store.search(vec, k=5, filters={"company": "3M"})
    lexical = bm25.search("revenue operations company equipment", k=5, filters={"company": "3M"})
    assert dense and lexical
    assert {h.chunk.company for h in dense} == {"3M"}
    assert {h.chunk.company for h in lexical} == {"3M"}
    both = store.search(vec, k=5, filters={"company": "3M", "fiscal_year": 2017})
    assert both == []


def test_index_rebuild_is_idempotent(cfg, embedder):
    chunks = make_chunks()
    first = index_chunks(chunks, cfg, embedder=embedder)
    second = index_chunks(chunks, cfg, embedder=embedder)
    assert first["chunks"] == second["chunks"] == 5
    store, bm25 = load_index(cfg)
    try:
        assert store.count() == 5
        assert bm25.count() == 5
    finally:
        store.close()


def _manifest_path(cfg) -> Path:
    return Path(cfg.index.dir) / cfg.index.version / "manifest.json"


def test_load_refuses_embedding_model_mismatch(cfg, embedder):
    index_chunks(make_chunks(), cfg, embedder=embedder)
    manifest = json.loads(_manifest_path(cfg).read_text())
    manifest["embedding_model"] = "some/other-model"
    _manifest_path(cfg).write_text(json.dumps(manifest))
    with pytest.raises(IndexMismatchError, match="embedding"):
        load_index(cfg)


def test_load_refuses_index_version_mismatch(cfg, embedder):
    index_chunks(make_chunks(), cfg, embedder=embedder)
    manifest = json.loads(_manifest_path(cfg).read_text())
    manifest["index_version"] = "v0"
    _manifest_path(cfg).write_text(json.dumps(manifest))
    with pytest.raises(IndexMismatchError, match="version"):
        load_index(cfg)


def test_embedder_sorting_preserves_input_order(embedder):
    texts = ["a much longer sentence about revenue and margins " * 3, "short", "medium length text"]
    embedder.sort_by_length = False
    plain = embedder.embed(texts)
    embedder.sort_by_length = True
    sorted_ = embedder.embed(texts)
    assert abs(plain - sorted_).max() < 1e-4


def test_query_and_document_prefixes_are_applied(embedder):
    seen = []
    real = embedder._model.embed
    embedder._model.embed = lambda texts, **kw: (seen.extend(texts), real(texts, **kw))[1]
    try:
        embedder.query_prefix, embedder.doc_prefix = "Q: ", "D: "
        embedder.embed_query("revenue")
        embedder.embed_documents(["table"])
    finally:
        embedder._model.embed = real
        embedder.query_prefix = embedder.doc_prefix = ""
    assert seen == ["Q: revenue", "D: table"]
