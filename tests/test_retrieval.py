from types import SimpleNamespace

import numpy as np
import pytest

from src.config import load_config
from src.query.query_router import route
from src.rerank.reranker import Reranker
from src.retrieval.retriever import HybridRetriever, rrf
from src.schemas.models import Chunk, Hit

KNOWN = ["3M", "Adobe", "AMD", "American Express", "Best Buy", "CVS Health", "PepsiCo"]


def make_hit(cid: str, score: float = 1.0, text: str = "text", company: str = "3M") -> Hit:
    chunk = Chunk(
        id=cid, doc_id="d", parent_id="p", text=text, context=f"{company} 2018 10-K, page 1",
        page=1, company=company, fiscal_year=2018,
    )
    return Hit(chunk_id=cid, score=score, chunk=chunk)


# ---- rrf ----

def test_rrf_ranks_doc_in_both_lists_first():
    dense = [make_hit("a"), make_hit("b"), make_hit("c")]
    lexical = [make_hit("d"), make_hit("a"), make_hit("e")]
    fused = rrf([dense, lexical], k=60)
    assert fused[0].chunk_id == "a"
    assert {h.chunk_id for h in fused} == {"a", "b", "c", "d", "e"}  # deduplicated union
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_scores_sorted_descending():
    fused = rrf([[make_hit("a"), make_hit("b")], [make_hit("c")]], k=60)
    scores = [h.score for h in fused]
    assert scores == sorted(scores, reverse=True)


# ---- HybridRetriever (fakes for the two stores; fusion logic is real) ----

class FakeStore:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, vec, k, filters=None):
        self.calls.append((k, filters))
        return self.hits


class FakeBM25(FakeStore):
    def search(self, q, k, filters=None):
        self.calls.append((q, k, filters))
        return self.hits


class FakeEmbedder:
    def embed(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


def test_hybrid_retriever_fuses_and_passes_filters_to_both():
    cfg = load_config()
    cfg.retrieval.min_chunk_tokens = 0  # the fake hits are 1-token texts; this test is about fusion
    store = FakeStore([make_hit("a"), make_hit("b")])
    bm25 = FakeBM25([make_hit("c"), make_hit("a")])
    retriever = HybridRetriever(store, bm25, FakeEmbedder(), cfg)
    filters = {"company": "3M"}
    hits = retriever.retrieve("capex", k=3, filters=filters)
    assert hits[0].chunk_id == "a"
    assert len(hits) == 3
    assert store.calls == [(3, filters)]
    assert bm25.calls == [("capex", 3, filters)]


# ---- Reranker (real cross-encoder) ----

@pytest.fixture(scope="module")
def reranker():
    return Reranker(load_config())


def test_rerank_returns_top_n_sorted(reranker):
    hits = [
        make_hit("x1", text="The cafeteria serves lunch from noon to two."),
        make_hit("rel", text="Purchases of property, plant and equipment (capital expenditures) were $1,577 million."),
        make_hit("x2", text="The board of directors held six meetings."),
        make_hit("x3", text="Employees receive annual safety training."),
    ]
    out = reranker.rerank("What were the capital expenditures?", hits, top_n=2)
    assert len(out) == 2
    assert out[0].chunk_id == "rel"
    assert out[0].score >= out[1].score


def test_rerank_handles_empty_and_fewer_than_top_n(reranker):
    assert reranker.rerank("q", [], top_n=5) == []
    out = reranker.rerank("capital expenditures", [make_hit("only")], top_n=5)
    assert [h.chunk_id for h in out] == ["only"]


# ---- query router ----

def test_route_extracts_company_and_year():
    r = route("3M FY2018 capex", KNOWN)
    assert r.filters == {"company": "3M", "fiscal_year": 2018}
    assert r.in_scope is True


@pytest.mark.parametrize(
    "question, year",
    [
        ("What was Adobe's revenue in fiscal year 2017?", 2017),
        ("Adobe FY 2016 operating margin", 2016),
        ("Adobe fiscal 2022 net income", 2022),
        ("What is Adobe's gross margin for 2015?", 2015),
    ],
)
def test_route_year_formats(question, year):
    assert route(question, KNOWN).filters == {"company": "Adobe", "fiscal_year": year}


def test_route_multiword_and_case_insensitive_company():
    assert route("what is american express net income fy2022", KNOWN).filters["company"] == "American Express"
    assert route("Best Buy FY2023 inventory", KNOWN).filters["company"] == "Best Buy"


def test_route_unknown_company_in_scope_false():
    r = route("What was Tesla's FY2020 revenue?", KNOWN)
    assert r.in_scope is False
    assert r.filters == {}


def test_route_does_not_match_inside_other_words():
    assert route("What does AMDOCS report for FY2019?", KNOWN).in_scope is False


def test_route_no_year_filters_company_only():
    assert route("Which segments does 3M report?", KNOWN).filters == {"company": "3M"}


def test_route_multiple_years_uses_the_latest():
    # the latest 10-K contains the prior-year comparatives
    assert route("How did 3M capex change from FY2017 to FY2018?", KNOWN).filters == {
        "company": "3M", "fiscal_year": 2018,
    }
    assert route("Has Adobe improved between 2022 and the 2021 fiscal period?", KNOWN).filters == {
        "company": "Adobe", "fiscal_year": 2022,
    }


@pytest.mark.parametrize(
    "question",
    ["Did AMD report customer concentration in FY22?", "AMD FY 22 revenue", "AMD fy'22 revenue", "AMD FY22."],
)
def test_route_two_digit_fiscal_year(question):
    assert route(question, KNOWN).filters == {"company": "AMD", "fiscal_year": 2022}


def test_route_four_digit_fy_is_not_read_as_two_digit():
    assert route("3M FY2018 capex", KNOWN).filters["fiscal_year"] == 2018
    assert route("3M FY2018 compared with FY2017", KNOWN).filters["fiscal_year"] == 2018


def test_route_fiscal_markers_win_over_bare_years():
    r = route("Adobe FY2016 operating income versus 2019 guidance", KNOWN)
    assert r.filters["fiscal_year"] == 2016


def test_route_two_digit_year_needs_the_fy_marker():
    # a bare two-digit number is not a year
    assert route("What were Adobe's 22 largest customers?", KNOWN).filters == {"company": "Adobe"}


def test_route_multiple_companies_in_scope_without_company_filter():
    r = route("Compare Adobe and AMD revenue in FY2022", KNOWN)
    assert r.in_scope is True
    assert r.filters == {"fiscal_year": 2022}


def test_route_keeps_query_text():
    q = "3M FY2018 capex"
    assert route(q, KNOWN).query == q


# ---- tiny-chunk filter (retrieval.min_chunk_tokens) ----

def _retriever_with(min_tokens, dense, lexical):
    cfg = load_config()
    cfg.retrieval.min_chunk_tokens = min_tokens
    return HybridRetriever(FakeStore(dense), FakeBM25(lexical), FakeEmbedder(), cfg)


def test_tiny_chunks_are_dropped_when_min_chunk_tokens_is_set():
    tiny = make_hit("tiny", text="21")
    real1 = make_hit("real1", text="Net cash provided by operating activities was $3.1 billion in 2022, driven by higher net income and favorable working capital changes.")
    real2 = make_hit("real2", text="Customer concentration: one customer accounted for 16% of consolidated net revenue in 2022, compared with 14% in the prior year.")
    r = _retriever_with(15, dense=[tiny, real1], lexical=[real2, tiny])
    ids = [h.chunk_id for h in r.retrieve("q", k=5)]
    assert "tiny" not in ids
    assert set(ids) == {"real1", "real2"}


def test_dropping_tiny_chunks_moves_real_chunks_up():
    tiny = make_hit("tiny", text="ii")
    real = make_hit("real", text="Operating income increased to $9.0 billion for the fiscal year, reflecting higher volumes, pricing actions and lower restructuring charges.")
    r = _retriever_with(15, dense=[tiny, real], lexical=[])
    assert [h.chunk_id for h in r.retrieve("q", k=5)] == ["real"]


def test_min_chunk_tokens_zero_keeps_everything():
    tiny = make_hit("tiny", text="21")
    real = make_hit("real", text="Operating income increased to $9.0 billion for the fiscal year, reflecting higher volumes, pricing actions and lower restructuring charges.")
    r = _retriever_with(0, dense=[tiny, real], lexical=[])
    assert {h.chunk_id for h in r.retrieve("q", k=5)} == {"tiny", "real"}


def test_tables_are_never_treated_as_tiny():
    table = make_hit("table", text="|Segment|2018|2017|\n|---|---|---|\n|Safety|3,000|2,900|\n|Health|2,500|2,400|")
    r = _retriever_with(15, dense=[table], lexical=[])
    assert [h.chunk_id for h in r.retrieve("q", k=5)] == ["table"]
