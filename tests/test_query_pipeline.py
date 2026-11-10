import json
from types import SimpleNamespace

import pytest

from src.config import load_config
from src.llm.llm_client import BudgetNearlyExhausted, LLMError
from src.pipelines.query_pipeline import QueryPipeline
from src.schemas.models import Chunk, Hit, QueryResult

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


class FakeReranker:
    """Reverses the order so tests can tell rerank ran."""

    def __init__(self):
        self.calls = []

    def rerank(self, q, hits, top_n):
        self.calls.append(top_n)
        return list(reversed(hits))[:top_n]


class FakeLLM:
    def __init__(self, reply="The answer [3M_2018_10K p.1]."):
        self.reply, self.calls = reply, []

    def complete(self, prompt, system=None, **kwargs):
        self.calls.append(prompt)
        return self.reply


def cfg_with(router: bool, rerank: bool):
    c = load_config()
    c.retrieval.use_router_filter = router
    c.rerank.enabled = rerank
    return c


def make_pipeline(cfg, hits, reranker=None, llm=None):
    retriever = FakeRetriever(hits)
    llm = llm or FakeLLM()
    return QueryPipeline(cfg, retriever, reranker, llm, KNOWN), retriever, llm


def test_router_and_rerank_switches_exist_in_config():
    # Values are tuned in Task 9; tests set them explicitly via cfg_with().
    cfg = load_config()
    assert isinstance(cfg.retrieval.use_router_filter, bool)
    assert isinstance(cfg.rerank.enabled, bool)


def test_router_off_sends_no_filters_and_no_scope_check():
    hits = [make_hit(f"h{i}") for i in range(8)]
    pipe, retriever, llm = make_pipeline(cfg_with(False, False), hits)
    out = pipe.run("Tell me about Tesla FY2020")  # unknown company, but router is off
    assert retriever.calls[0]["filters"] is None
    assert retriever.calls[0]["k"] == load_config().retrieval.top_k
    assert len(llm.calls) == 1
    assert isinstance(out.result, QueryResult)


def test_no_rerank_context_is_first_top_n_of_ranked_list():
    cfg = cfg_with(False, False)
    hits = [make_hit(f"h{i}") for i in range(8)]
    pipe, _, llm = make_pipeline(cfg, hits)
    out = pipe.run("3M capex")
    assert [h.chunk_id for h in out.ranked] == [h.chunk_id for h in hits]  # full list kept for metrics
    assert out.result.contexts == [f"text h{i}" for i in range(cfg.rerank.top_n)]


def test_rerank_on_reorders_full_list_and_context_uses_top_n():
    cfg = cfg_with(False, True)
    hits = [make_hit(f"h{i}") for i in range(8)]
    reranker = FakeReranker()
    pipe, _, _ = make_pipeline(cfg, hits, reranker=reranker)
    out = pipe.run("3M capex")
    assert reranker.calls == [len(hits)]  # whole candidate list reranked so MRR sees the new order
    assert [h.chunk_id for h in out.ranked] == [f"h{i}" for i in reversed(range(8))]
    assert out.result.contexts == [f"text h{i}" for i in reversed(range(8))][: cfg.rerank.top_n]


def test_router_on_passes_company_and_year_filters():
    pipe, retriever, _ = make_pipeline(cfg_with(True, False), [make_hit("h0")])
    pipe.run("3M FY2018 capex")
    assert retriever.calls[0]["filters"] == {"company": "3M", "fiscal_year": 2018}


def test_router_on_out_of_scope_says_idk_without_retrieval_or_llm():
    pipe, retriever, llm = make_pipeline(cfg_with(True, False), [make_hit("h0")])
    out = pipe.run("What was Tesla's FY2020 revenue?")
    assert out.result.answer == "I don't know"
    assert out.result.citations == []
    assert retriever.calls == [] and llm.calls == []
    assert out.ranked == []


def test_latency_has_stage_breakdown_and_total():
    pipe, _, _ = make_pipeline(cfg_with(False, True), [make_hit("h0")], reranker=FakeReranker())
    lat = pipe.run("3M capex").result.latency_ms
    assert {"retrieve", "rerank", "llm", "total"} <= set(lat)
    assert lat["total"] >= lat["llm"] >= 0


# ---------- run_eval ----------

from src.eval.rag_metrics import run_eval  # noqa: E402


class StubPipeline:
    """Returns canned results per question id; records calls."""

    def __init__(self, answers, ranked_by_q):
        self.answers, self.ranked_by_q, self.calls = answers, ranked_by_q, []

    def run(self, q, user=None):
        self.calls.append(q)
        if isinstance(self.answers.get(q), Exception):
            raise self.answers[q]
        answer = self.answers[q]
        result = QueryResult(
            answer=answer, citations=[], contexts=["ctx"],
            latency_ms={"retrieve": 1.0, "rerank": 0.0, "llm": 5.0, "total": 6.0},
        )
        return SimpleNamespace(result=result, ranked=self.ranked_by_q[q])


class JudgeLLM:
    """Verdicts keyed off the prediction text; tracks which provider 'served' last."""

    def __init__(self):
        self.calls = []
        self.last_call = SimpleNamespace(provider="gemini", model="m", cached=False)

    def complete(self, prompt, system=None, **kwargs):
        self.calls.append(prompt)
        if "claims" in (system or "").lower() or "claim" in prompt.lower() and "supported" in prompt.lower():
            return '{"claims": [{"claim": "x", "supported": true}, {"claim": "y", "supported": false}]}'
        return "CORRECT" if "RIGHT" in prompt else "INCORRECT"


def write_questions(tmp_path, rows):
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def eval_cfg(tmp_path, rows):
    cfg = load_config()
    cfg.data.eval_dir = str(tmp_path)
    cfg.eval.runs_dir = str(tmp_path / "runs")
    write_questions(tmp_path, rows)
    return cfg


def row(i, split="dev", pages=(1,)):
    return {"id": f"q{i}", "question": f"question {i}", "gold_answer": "gold", "doc_id": "3M_2018_10K",
            "gold_pages": list(pages), "split": split}


def test_run_eval_aggregates_and_writes_run_file(tmp_path):
    rows = [row(1), row(2), row(3, split="test")]
    cfg = eval_cfg(tmp_path, rows)
    ranked = {"question 1": [make_hit("a", page=1)], "question 2": [make_hit("b", page=2), make_hit("c", page=1)]}
    pipe = StubPipeline({"question 1": "RIGHT answer", "question 2": "I don't know"}, ranked)
    summary = run_eval("dev", cfg=cfg, pipeline=pipe, llm=JudgeLLM())
    assert pipe.calls == ["question 1", "question 2"]  # test split never touched
    assert summary["n"] == 2 and summary["errors"] == 0
    assert summary["recall_at_5"] == pytest.approx((1.0 + 1.0) / 2)
    assert summary["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    assert summary["accuracy"] == pytest.approx(0.5)  # "I don't know" is wrong
    assert summary["faithfulness"] == pytest.approx(0.5)  # only the answered question is judged
    assert summary["faithfulness_n"] == 1
    assert summary["idk"] == 1
    run_files = list((tmp_path / "runs").glob("*.json"))
    assert len(run_files) == 1
    saved = json.loads(run_files[0].read_text(encoding="utf-8"))
    assert saved["summary"]["n"] == 2 and len(saved["questions"]) == 2


def test_run_eval_limit_takes_first_n_by_id(tmp_path):
    rows = [row(3), row(1), row(2)]
    cfg = eval_cfg(tmp_path, rows)
    ranked = {f"question {i}": [make_hit("a")] for i in (1, 2, 3)}
    pipe = StubPipeline({f"question {i}": "RIGHT" for i in (1, 2, 3)}, ranked)
    summary = run_eval("dev", limit=2, cfg=cfg, pipeline=pipe, llm=JudgeLLM())
    assert pipe.calls == ["question 1", "question 2"]
    assert summary["n"] == 2 and summary["limit"] == 2


def test_run_eval_refuses_test_split_unless_final(tmp_path):
    cfg = eval_cfg(tmp_path, [row(1, split="test")])
    pipe = StubPipeline({}, {})
    with pytest.raises(PermissionError):
        run_eval("test", cfg=cfg, pipeline=pipe, llm=JudgeLLM())
    assert pipe.calls == []


def test_run_eval_records_question_errors_and_continues(tmp_path):
    cfg = eval_cfg(tmp_path, [row(1), row(2)])
    ranked = {"question 1": [make_hit("a")], "question 2": [make_hit("a")]}
    pipe = StubPipeline({"question 1": LLMError("all providers failed"), "question 2": "RIGHT"}, ranked)
    summary = run_eval("dev", cfg=cfg, pipeline=pipe, llm=JudgeLLM())
    assert summary["errors"] == 1 and summary["n"] == 2
    assert summary["accuracy"] == pytest.approx(0.5)  # errored question counts as not correct


def test_run_eval_budget_stop_saves_partial_and_reraises(tmp_path):
    cfg = eval_cfg(tmp_path, [row(1), row(2)])
    ranked = {"question 1": [make_hit("a")]}
    pipe = StubPipeline({"question 1": "RIGHT", "question 2": BudgetNearlyExhausted("gemini 160/200")}, ranked)
    with pytest.raises(BudgetNearlyExhausted):
        run_eval("dev", cfg=cfg, pipeline=pipe, llm=JudgeLLM())
    saved = [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "runs").glob("*.json")]
    assert len(saved) == 1 and saved[0]["summary"]["completed"] == 1 and saved[0]["summary"]["stopped_early"]
