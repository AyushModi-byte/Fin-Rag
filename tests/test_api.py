import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.routes import create_app
from src.cache.semantic_cache import SemanticCache
from src.config import load_config
from src.observability.tracing import trace
from src.pipelines.query_pipeline import QueryPipeline
from src.schemas.models import Chunk, Citation, Hit, QueryResult
from src.security.acl import allowed_companies, redact_pii

KNOWN = ["3M", "Adobe"]


def make_hit(cid, company="3M", page=1):
    chunk = Chunk(
        id=cid, doc_id=f"{company}_2018_10K", parent_id="p", text=f"text {cid}", context="ctx",
        page=page, company=company, fiscal_year=2018,
    )
    return Hit(chunk_id=cid, score=1.0, chunk=chunk)


class FakeRetriever:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def retrieve(self, q, k, filters=None):
        self.calls.append({"q": q, "k": k, "filters": filters})
        return list(self.hits)


class FakeLLM:
    def __init__(self, reply="The answer [3M_2018_10K p.1]."):
        self.reply, self.calls = reply, []

    def complete(self, prompt, system=None, **kwargs):
        self.calls.append(prompt)
        return self.reply


class FakeEmbedder:
    """Maps a question to a fixed vector; unknown questions get an orthogonal one."""

    def __init__(self, vectors):
        self.vectors, self.calls = vectors, 0

    def embed_query(self, text):
        self.calls += 1
        return np.array(self.vectors.get(text, [0.0, 0.0, 1.0]), dtype=np.float32)


def cfg_for(tmp_path, acl_users=None):
    cfg = load_config()
    cfg.retrieval.use_router_filter = True
    cfg.rerank.enabled = False
    cfg.cache.enabled = True
    cfg.cache.path = str(tmp_path / "semantic_cache.json")
    cfg.data.log_file = str(tmp_path / "app.log")
    if acl_users is not None:
        cfg.acl.users = type(cfg.acl.users)(**acl_users)
    return cfg


def make_pipeline(cfg, cache=None, hits=None):
    retriever = FakeRetriever(hits if hits is not None else [make_hit("h0")])
    llm = FakeLLM()
    return QueryPipeline(cfg, retriever, None, llm, KNOWN, cache=cache), retriever, llm


def sample_result(answer="A"):
    return QueryResult(answer=answer, citations=[Citation(doc_id="d", page=1)], contexts=["c"],
                       latency_ms={"llm": 5.0, "total": 9.0})


# ---------- API ----------


def test_empty_question_returns_422(tmp_path):
    pipe, _, _ = make_pipeline(cfg_for(tmp_path))
    client = TestClient(create_app(pipeline=pipe, cfg=cfg_for(tmp_path)))
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={"question": "   "}).status_code == 422
    assert client.post("/query", json={}).status_code == 422


def test_health_ok(tmp_path):
    client = TestClient(create_app(pipeline=make_pipeline(cfg_for(tmp_path))[0], cfg=cfg_for(tmp_path)))
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_health_does_not_build_the_pipeline(tmp_path):
    # /health must be instant and must not load models; the pipeline is built on the first query.
    built = []
    app = create_app(cfg=cfg_for(tmp_path), pipeline_factory=lambda cfg: built.append(1))
    assert TestClient(app).get("/health").status_code == 200
    assert built == []


def test_query_returns_query_result(tmp_path):
    cfg = cfg_for(tmp_path)
    pipe, _, _ = make_pipeline(cfg)
    r = TestClient(create_app(pipeline=pipe, cfg=cfg)).post("/query", json={"question": "3M FY2018 capex?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("The answer")
    assert body["citations"] == [{"doc_id": "3M_2018_10K", "page": 1}]
    assert {"retrieve", "llm", "total"} <= set(body["latency_ms"])


def test_overlong_question_returns_422(tmp_path):
    cfg = cfg_for(tmp_path)
    pipe, _, _ = make_pipeline(cfg)
    client = TestClient(create_app(pipeline=pipe, cfg=cfg))
    r = client.post("/query", json={"question": "x" * (cfg.api.max_question_chars + 1)})
    assert r.status_code == 422


def test_budget_exhausted_returns_503_not_500(tmp_path):
    from src.llm.llm_client import BudgetNearlyExhausted

    class Boom:
        def run(self, q, user=None):
            raise BudgetNearlyExhausted("groq 160/200")

    r = TestClient(create_app(pipeline=Boom(), cfg=cfg_for(tmp_path))).post("/query", json={"question": "q"})
    assert r.status_code == 503


# ---------- ACL ----------


def test_acl_default_allows_everything(tmp_path):
    assert allowed_companies("anyone", cfg_for(tmp_path)) == {"*"}
    assert allowed_companies(None, cfg_for(tmp_path)) == {"*"}


def test_acl_user_gets_only_configured_companies(tmp_path):
    cfg = cfg_for(tmp_path, {"alice": ["Adobe"]})
    assert allowed_companies("alice", cfg) == {"Adobe"}
    assert allowed_companies("bob", cfg) == {"*"}  # unknown user falls back to acl.default


def test_acl_blocks_disallowed_company(tmp_path):
    cfg = cfg_for(tmp_path, {"alice": ["Adobe"]})
    pipe, retriever, llm = make_pipeline(cfg)
    out = pipe.run("What was 3M's FY2018 capex?", user="alice")
    assert out.result.answer == "I don't know"
    assert out.result.contexts == [] and out.result.citations == []
    assert retriever.calls == [] and llm.calls == []


def test_acl_blocks_when_any_named_company_is_disallowed(tmp_path):
    cfg = cfg_for(tmp_path, {"alice": ["Adobe"]})
    pipe, retriever, llm = make_pipeline(cfg)
    out = pipe.run("Compare 3M and Adobe FY2018 revenue", user="alice")
    assert out.result.answer == "I don't know" and retriever.calls == [] and llm.calls == []


def test_acl_allows_permitted_company_and_filters_to_it(tmp_path):
    cfg = cfg_for(tmp_path, {"alice": ["3M"]})
    pipe, retriever, llm = make_pipeline(cfg)
    out = pipe.run("3M FY2018 capex", user="alice")
    assert retriever.calls[0]["filters"] == {"company": "3M", "fiscal_year": 2018}
    assert len(llm.calls) == 1 and out.result.contexts


def test_acl_applies_even_when_router_filter_is_off(tmp_path):
    cfg = cfg_for(tmp_path, {"alice": ["Adobe"]})
    cfg.retrieval.use_router_filter = False
    pipe, retriever, llm = make_pipeline(cfg)
    out = pipe.run("What was 3M's FY2018 capex?", user="alice")
    assert out.result.answer == "I don't know" and retriever.calls == []


# ---------- PII ----------


def test_redact_pii_emails_and_phones():
    text = "Mail jane.doe@example.com or call +1 (415) 555-0123 / 415-555-0199 / 415.555.0100."
    out = redact_pii(text)
    assert "@" not in out and "555" not in out
    assert "[EMAIL]" in out and out.count("[PHONE]") == 3


def test_redact_pii_leaves_financial_numbers_alone():
    text = "3M FY2018 revenue was $32,765 million, up 2.3% from 1,234,567.89; page 60-61; 10-K."
    assert redact_pii(text) == text


def test_pii_redacted_in_logs(tmp_path, caplog):
    cfg = cfg_for(tmp_path)
    pipe, _, _ = make_pipeline(cfg)
    with caplog.at_level(logging.INFO, logger="finrag"):
        pipe.run("3M capex? contact me at jane.doe@example.com or 415-555-0123")
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "jane.doe@example.com" not in logged and "415-555-0123" not in logged
    assert "[EMAIL]" in logged and "[PHONE]" in logged


def test_pii_never_reaches_the_llm(tmp_path):
    cfg = cfg_for(tmp_path)
    pipe, retriever, llm = make_pipeline(cfg)
    pipe.run("3M capex? mail jane.doe@example.com")
    assert "jane.doe@example.com" not in llm.calls[0]
    assert "jane.doe@example.com" not in retriever.calls[0]["q"]


# ---------- tracing ----------


def test_trace_records_stage_latency_and_logs_json_line(caplog):
    timings: dict = {}
    with caplog.at_level(logging.INFO, logger="finrag"):
        with trace("retrieve", timings):
            pass
    assert timings["retrieve"] >= 0
    rec = json.loads([r.getMessage() for r in caplog.records if "retrieve" in r.getMessage()][0])
    assert rec["stage"] == "retrieve" and rec["ms"] >= 0


def test_trace_records_even_when_the_stage_raises():
    timings: dict = {}
    with pytest.raises(ValueError):
        with trace("llm", timings):
            raise ValueError("boom")
    assert "llm" in timings


def test_trace_json_lines_reach_the_log_file(tmp_path):
    from src.observability.tracing import setup_trace_logging

    log = tmp_path / "app.log"
    setup_trace_logging(str(log))
    with trace("retrieve", {}):
        pass
    for h in logging.getLogger("finrag.trace").handlers:
        h.flush()
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines and lines[-1]["stage"] == "retrieve"


# ---------- semantic cache ----------


def test_cache_exact_hit_needs_no_embedding(tmp_path):
    emb = FakeEmbedder({})
    cache = SemanticCache(cfg_for(tmp_path), emb)
    cache.put("3M capex?", sample_result())
    emb.calls = 0
    hit = cache.get("  3M CAPEX?  ")  # normalised exact match
    assert hit is not None and hit.answer == "A" and emb.calls == 0


def test_cache_similar_question_hits_above_threshold(tmp_path):
    cfg = cfg_for(tmp_path)
    emb = FakeEmbedder({"q one": [1.0, 0.0, 0.0], "q one reworded": [0.99, 0.05, 0.0], "q other": [0.0, 1.0, 0.0]})
    cache = SemanticCache(cfg, emb)
    cache.put("q one", sample_result())
    assert cache.get("q one reworded") is not None
    assert cache.get("q other") is None


def test_cache_threshold_comes_from_config(tmp_path):
    cfg = cfg_for(tmp_path)
    cfg.cache.threshold = 0.9999
    emb = FakeEmbedder({"a": [1.0, 0.0, 0.0], "b": [0.99, 0.05, 0.0]})
    cache = SemanticCache(cfg, emb)
    cache.put("a", sample_result())
    assert cache.get("b") is None


def test_cache_similarity_hit_requires_same_scope(tmp_path):
    # "FY2018 revenue" and "FY2019 revenue" embed almost identically; they must not share an answer.
    cfg = cfg_for(tmp_path)
    emb = FakeEmbedder({"rev 2018": [1.0, 0.0, 0.0], "rev 2019": [0.999, 0.01, 0.0]})
    cache = SemanticCache(cfg, emb)
    cache.put("rev 2018", sample_result(), scope={"company": "3M", "fiscal_year": 2018})
    assert cache.get("rev 2019", scope={"company": "3M", "fiscal_year": 2019}) is None
    assert cache.get("rev 2019", scope={"company": "3M", "fiscal_year": 2018}) is not None


def test_cache_persists_to_json_file(tmp_path):
    cfg = cfg_for(tmp_path)
    SemanticCache(cfg, FakeEmbedder({})).put("persist me", sample_result("kept"))
    reloaded = SemanticCache(cfg, FakeEmbedder({}))
    assert reloaded.get("persist me").answer == "kept"
    json.loads((tmp_path / "semantic_cache.json").read_text(encoding="utf-8"))  # valid JSON


def test_cache_is_dropped_when_index_or_prompt_changes(tmp_path):
    cfg = cfg_for(tmp_path)
    SemanticCache(cfg, FakeEmbedder({})).put("q", sample_result())
    cfg.generation.prompt_version = "some_other_version"
    assert SemanticCache(cfg, FakeEmbedder({})).get("q") is None


def test_cache_survives_a_corrupt_file(tmp_path):
    cfg = cfg_for(tmp_path)
    (tmp_path / "semantic_cache.json").write_text("{not json", encoding="utf-8")
    cache = SemanticCache(cfg, FakeEmbedder({}))
    assert cache.get("q") is None
    cache.put("q", sample_result())
    assert cache.get("q") is not None


def test_semantic_cache_hit_skips_llm(tmp_path):
    cfg = cfg_for(tmp_path)
    emb = FakeEmbedder({"3M FY2018 capex?": [1.0, 0.0, 0.0], "3M FY2018 capex ?": [0.99, 0.05, 0.0]})
    pipe, retriever, llm = make_pipeline(cfg, cache=SemanticCache(cfg, emb))
    first = pipe.run("3M FY2018 capex?")
    assert len(llm.calls) == 1 and len(retriever.calls) == 1
    again = pipe.run("3M FY2018 capex?")  # exact
    similar = pipe.run("3M FY2018 capex ?")  # similarity
    assert len(llm.calls) == 1 and len(retriever.calls) == 1
    assert again.result.answer == first.result.answer == similar.result.answer
    assert "cache" in again.result.latency_ms and again.result.latency_ms["total"] < 100


def test_cache_disabled_in_config_means_no_cache_use(tmp_path):
    cfg = cfg_for(tmp_path)
    cfg.cache.enabled = False
    pipe, _, llm = make_pipeline(cfg, cache=SemanticCache(cfg, FakeEmbedder({})))
    pipe.run("3M FY2018 capex?")
    pipe.run("3M FY2018 capex?")
    assert len(llm.calls) == 2


def test_acl_denial_is_never_served_from_or_stored_in_cache(tmp_path):
    cfg = cfg_for(tmp_path, {"alice": ["Adobe"]})
    cache = SemanticCache(cfg, FakeEmbedder({}))
    pipe, _, llm = make_pipeline(cfg, cache=cache)
    pipe.run("3M FY2018 capex?")  # default user: allowed, cached
    out = pipe.run("3M FY2018 capex?", user="alice")  # alice must still be denied
    assert out.result.answer == "I don't know" and len(llm.calls) == 1


def test_out_of_scope_idk_is_not_cached(tmp_path):
    cfg = cfg_for(tmp_path)
    cache = SemanticCache(cfg, FakeEmbedder({}))
    pipe, _, _ = make_pipeline(cfg, cache=cache)
    pipe.run("What was Tesla's FY2020 revenue?")
    assert cache.get("What was Tesla's FY2020 revenue?") is None


def test_trace_lines_are_not_duplicated_when_both_log_setups_share_one_file(tmp_path, monkeypatch):
    from src.ingestion.loader import setup_logging
    from src.observability.tracing import setup_trace_logging

    monkeypatch.setattr(logging.getLogger("finrag"), "handlers", [])
    monkeypatch.setattr(logging.getLogger("finrag.trace"), "handlers", [])
    log = tmp_path / "app.log"
    setup_logging(str(log))
    setup_trace_logging(str(log))
    with trace("retrieve", {}):
        pass
    for lg in ("finrag", "finrag.trace"):
        for h in logging.getLogger(lg).handlers:
            h.flush()
    lines = [x for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == 1 and json.loads(lines[0])["stage"] == "retrieve"
