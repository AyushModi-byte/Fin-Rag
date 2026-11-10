import hashlib

import numpy as np
import pytest

from src.config import load_config
from src.pipelines.index_pipeline import (
    IndexMismatchError, ReindexTooSlow, index_chunks, load_index, load_parents,
)
from src.schemas.models import Chunk, Parent


class FakeEmbedder:
    """Deterministic, instant. Optionally raises on the Nth embed_documents call (an interrupt)."""

    model_id = "fake/model"
    dim = 8
    doc_prefix = ""
    query_prefix = ""

    def __init__(self, fail_on_call: int | None = None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def embed_documents(self, texts):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise KeyboardInterrupt("simulated interrupt")
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha1(t.encode()).digest()
            out[i] = np.frombuffer(h[: self.dim], dtype=np.uint8) / 255.0 + 0.01
        return out


def chunks(n=11):
    return [
        Chunk(id=f"c{i:03d}", doc_id="D_2022_10K", parent_id="p", text=f"revenue line {i} amount {i * 7}",
              context="Acme FY2022 10-K | page 1", page=1 + i % 3, company="Acme", fiscal_year=2022)
        for i in range(n)
    ]


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.index.dir = str(tmp_path / "index")
    c.index.version = "vtest"
    c.index.shard_size = 3
    c.reindex.max_hours = None
    c.embeddings.model = FakeEmbedder.model_id
    return c


def stored_ids(cfg):
    store, _ = load_index(cfg)
    try:
        hits = store.search(np.ones(FakeEmbedder.dim, dtype=np.float32), 1000)
        return sorted(h.chunk_id for h in hits), store.count()
    finally:
        store.close()


def test_interrupt_then_resume_gives_same_ids_and_only_embeds_the_rest(cfg, tmp_path):
    cs = chunks(11)  # 4 shards of 3,3,3,2
    with pytest.raises(KeyboardInterrupt):
        index_chunks(cs, cfg, embedder=FakeEmbedder(fail_on_call=3))  # shards 0 and 1 done
    with pytest.raises(IndexMismatchError, match="incomplete"):
        load_index(cfg)  # no manifest yet: refuses
    resumed = FakeEmbedder()
    manifest = index_chunks(cs, cfg, embedder=resumed)
    assert resumed.calls == 2  # only shards 2 and 3
    assert manifest["chunks"] == manifest["dense_count"] == 11
    ids, count = stored_ids(cfg)

    cfg2 = load_config()
    cfg2.index.dir, cfg2.index.version, cfg2.index.shard_size = str(tmp_path / "other"), "vtest", 3
    cfg2.reindex.max_hours = None
    cfg2.embeddings.model = FakeEmbedder.model_id
    index_chunks(cs, cfg2, embedder=FakeEmbedder())
    ids2, count2 = stored_ids(cfg2)
    assert (ids, count) == (ids2, count2) == (sorted(c.id for c in cs), 11)


def test_rerun_on_a_finished_index_embeds_nothing(cfg):
    cs = chunks(7)
    index_chunks(cs, cfg, embedder=FakeEmbedder())
    again = FakeEmbedder()
    index_chunks(cs, cfg, embedder=again)
    assert again.calls == 0
    assert stored_ids(cfg)[1] == 7


def test_changed_chunk_text_invalidates_only_its_shard(cfg):
    cs = chunks(9)
    index_chunks(cs, cfg, embedder=FakeEmbedder())
    cs[4] = cs[4].model_copy(update={"text": "different text entirely"})  # shard 1
    e = FakeEmbedder()
    index_chunks(cs, cfg, embedder=e)
    assert e.calls == 1


def test_protected_version_is_never_built(cfg):
    cfg.index.version = "v1"
    with pytest.raises(IndexMismatchError, match="protected"):
        index_chunks(chunks(3), cfg, embedder=FakeEmbedder())


def test_time_gate_stops_before_embedding_everything(cfg):
    cfg.reindex.max_hours = 1e-12
    cfg.reindex.probe_chunks = 3
    e = FakeEmbedder()
    with pytest.raises(ReindexTooSlow):
        index_chunks(chunks(11), cfg, embedder=e)
    assert e.calls == 1  # only the probe


def test_load_refuses_changed_chunking_params(cfg):
    index_chunks(chunks(4), cfg, embedder=FakeEmbedder())
    cfg.chunking.child_tokens += 1
    with pytest.raises(IndexMismatchError, match="chunking"):
        load_index(cfg)


def test_load_refuses_changed_bm25_tokenizer(cfg):
    index_chunks(chunks(4), cfg, embedder=FakeEmbedder())
    cfg.lexical.tokenizer = "v1"
    with pytest.raises(IndexMismatchError, match="tokenizer"):
        load_index(cfg)


def test_parents_are_saved_and_loadable(cfg):
    parents = [Parent(id="t1", doc_id="D", page_start=2, page_end=2, text="|a|b|")]
    m = index_chunks(chunks(4), cfg, embedder=FakeEmbedder(), parents=parents)
    assert m["parents"] == 1
    assert load_parents(cfg)["t1"].text == "|a|b|"


def test_bulk_load_is_correct_and_searchable(tmp_path):
    from src.vector_db.vector_store import VectorStore

    cs = chunks(50)
    vecs = FakeEmbedder().embed_documents([c.text for c in cs])
    store = VectorStore(tmp_path / "q", "t", FakeEmbedder.dim)
    try:
        with store.bulk_load():
            store.upsert(cs[:25], vecs[:25])
            store.upsert(cs[25:], vecs[25:])
        assert store.count() == 50
        top = store.search(vecs[7], 1)[0]
        assert top.chunk_id == cs[7].id
    finally:
        store.close()
    again = VectorStore(tmp_path / "q", "t", FakeEmbedder.dim)  # reopened from disk
    try:
        assert again.count() == 50
    finally:
        again.close()
