"""Online pipeline: question -> (route) -> hybrid retrieve -> (rerank) -> grounded answer.

Orchestration only. Router and rerank are switched by config (`retrieval.use_router_filter`,
`rerank.enabled`). With the router OFF there is no scope check and no filter at all.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import load_config
from src.generation.context_expansion import expand_context
from src.generation.grounded_answer import generate
from src.prompts.prompt_templates import system_prompt
from src.query.query_router import route
from src.schemas.models import Hit, QueryResult


@dataclass
class PipelineOutput:
    result: QueryResult
    ranked: list[Hit]  # full ranked candidate list (used for retrieval metrics)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


class QueryPipeline:
    def __init__(self, cfg, retriever, reranker, llm, known_companies: list[str], closers=(), parents=None):
        self.cfg = cfg
        self.parents = parents or {}  # table/page parents by id, for generation.expand_tables
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.known_companies = known_companies
        self._closers = list(closers)

    def run(self, q: str, user: str | None = None) -> PipelineOutput:
        # `user` is accepted for the API; per-user ACL is added in Task 10.
        start = time.perf_counter()
        filters = None
        if self.cfg.retrieval.use_router_filter:
            routed = route(q, self.known_companies)
            if not routed.in_scope:
                result = generate(q, [], self.llm)
                result.latency_ms = {"retrieve": 0.0, "rerank": 0.0, "llm": 0.0, "total": _ms(start)}
                return PipelineOutput(result=result, ranked=[])
            filters = routed.filters or None

        t = time.perf_counter()
        ranked = self.retriever.retrieve(q, self.cfg.retrieval.top_k, filters)
        retrieve_ms = _ms(t)

        rerank_ms = 0.0
        if self.cfg.rerank.enabled and self.reranker is not None:
            t = time.perf_counter()
            # rerank the whole candidate list so MRR reflects the new order
            ranked = self.reranker.rerank(q, ranked, top_n=len(ranked))
            rerank_ms = _ms(t)

        context = ranked[: self.cfg.rerank.top_n]
        if self.cfg.generation.expand_tables:
            context = expand_context(context, self.parents, self.cfg.generation.max_context_tokens)
        result = generate(q, context, self.llm, system=system_prompt(self.cfg.generation.prompt_version))
        result.latency_ms = {
            "retrieve": retrieve_ms,
            "rerank": rerank_ms,
            "llm": result.latency_ms.get("llm", 0.0),
            "total": _ms(start),
        }
        return PipelineOutput(result=result, ranked=ranked)

    def close(self) -> None:
        for fn in self._closers:
            fn()


def build_pipeline(cfg=None) -> QueryPipeline:
    """Assemble the real pipeline from the on-disk index. Heavy: loads models."""
    from src.embeddings.embedder import Embedder
    from src.llm.llm_client import LLMClient
    from src.pipelines.index_pipeline import load_index, load_parents
    from src.rerank.reranker import Reranker
    from src.retrieval.retriever import HybridRetriever

    cfg = cfg or load_config()
    store, bm25 = load_index(cfg)
    retriever = HybridRetriever(store, bm25, Embedder(cfg), cfg)
    reranker = Reranker(cfg) if cfg.rerank.enabled else None
    docs = json.loads((Path(cfg.data.eval_dir) / "docs.json").read_text(encoding="utf-8"))
    known = sorted({d["company"] for d in docs.values()})
    parents = load_parents(cfg) if cfg.generation.expand_tables else None
    return QueryPipeline(cfg, retriever, reranker, LLMClient(cfg), known, closers=[store.close], parents=parents)


_default: QueryPipeline | None = None


def run_query(q: str, user: str | None = None) -> QueryResult:
    """Answer one question with the default pipeline (built lazily, once)."""
    global _default
    if _default is None:
        _default = build_pipeline()
    return _default.run(q, user).result
