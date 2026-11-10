"""Online pipeline: question -> redact -> ACL -> (cache) -> (route) -> hybrid retrieve -> (rerank) -> grounded answer.

Orchestration only. Router and rerank are switched by config (`retrieval.use_router_filter`,
`rerank.enabled`). With the router OFF there is no scope check and no filter at all (the router is
still used to enforce the ACL for a restricted user). The cache is passed in by the API; the eval
builds the pipeline without one so every question is really answered.
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import load_config
from src.generation.context_expansion import expand_context
from src.generation.grounded_answer import generate
from src.prompts.prompt_templates import system_prompt
from src.observability.tracing import log_event, trace
from src.query.query_router import route
from src.schemas.models import Hit, QueryResult
from src.security.acl import ALL, allowed_companies, is_allowed, redact_pii

log = logging.getLogger("finrag.pipeline")


@dataclass
class PipelineOutput:
    result: QueryResult
    ranked: list[Hit]  # full ranked candidate list (used for retrieval metrics)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


class QueryPipeline:
    def __init__(self, cfg, retriever, reranker, llm, known_companies: list[str], closers=(), parents=None, cache=None):
        self.cfg = cfg
        self.parents = parents or {}  # table/page parents by id, for generation.expand_tables
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.known_companies = known_companies
        self._closers = list(closers)
        self.cache = cache if (cache is not None and cfg.cache.enabled) else None

    def run(self, q: str, user: str | None = None) -> PipelineOutput:
        start = time.perf_counter()
        q = redact_pii(q)  # PII never reaches the logs, the cache, the retriever or the LLM
        log_event("query", user=user, question=q)

        allowed = allowed_companies(user, self.cfg)
        restricted = ALL not in allowed
        routed = route(q, self.known_companies) if (self.cfg.retrieval.use_router_filter or restricted) else None

        # a restricted user may only ask about companies that are named in the question AND in their ACL
        if restricted and (not routed.companies or not is_allowed(allowed, routed.companies)):
            log_event("acl_denied", user=user)
            return self._no_context(q, start)

        filters = None
        if self.cfg.retrieval.use_router_filter:
            if not routed.in_scope:
                return self._no_context(q, start)
            filters = routed.filters or None

        if self.cache is not None:
            cache_t: dict[str, float] = {}
            with trace("cache", cache_t):
                cached = self.cache.get(q, scope=filters)
            if cached is not None:
                cached.latency_ms = {"cache": cache_t["cache"], "total": _ms(start)}
                log_event("cache_hit", total_ms=round(cached.latency_ms["total"], 3))
                return PipelineOutput(result=cached, ranked=[])

        timings: dict[str, float] = {"rerank": 0.0}
        with trace("retrieve", timings):
            ranked = self.retriever.retrieve(q, self.cfg.retrieval.top_k, filters)

        if self.cfg.rerank.enabled and self.reranker is not None:
            with trace("rerank", timings):
                # rerank the whole candidate list so MRR reflects the new order
                ranked = self.reranker.rerank(q, ranked, top_n=len(ranked))

        context = ranked[: self.cfg.rerank.top_n]
        if self.cfg.generation.expand_tables:
            context = expand_context(context, self.parents, self.cfg.generation.max_context_tokens)
        with trace("generate", timings):
            result = generate(q, context, self.llm, system=system_prompt(self.cfg.generation.prompt_version))
        result.latency_ms = {
            "retrieve": timings["retrieve"],
            "rerank": timings["rerank"],
            "llm": result.latency_ms.get("llm", 0.0),
            "total": _ms(start),
        }
        if self.cache is not None:
            self.cache.put(q, result, scope=filters)
        log_event("answered", total_ms=round(result.latency_ms["total"], 3), citations=len(result.citations))
        return PipelineOutput(result=result, ranked=ranked)

    def _no_context(self, q: str, start: float) -> PipelineOutput:
        """"I don't know" without retrieval or an LLM call (out of scope or not allowed)."""
        result = generate(q, [], self.llm)
        result.latency_ms = {"retrieve": 0.0, "rerank": 0.0, "llm": 0.0, "total": _ms(start)}
        return PipelineOutput(result=result, ranked=[])

    def close(self) -> None:
        for fn in self._closers:
            fn()


def build_pipeline(cfg=None, with_cache: bool = False) -> QueryPipeline:
    """Assemble the real pipeline from the on-disk index. Heavy: loads models.

    `with_cache` is for the API only; the eval must never be served from the answer cache.
    """
    from src.cache.semantic_cache import SemanticCache
    from src.embeddings.embedder import Embedder
    from src.llm.llm_client import LLMClient
    from src.pipelines.index_pipeline import load_index, load_parents
    from src.rerank.reranker import Reranker
    from src.retrieval.retriever import HybridRetriever

    cfg = cfg or load_config()
    store, bm25 = load_index(cfg)
    embedder = Embedder(cfg)
    retriever = HybridRetriever(store, bm25, embedder, cfg)
    reranker = Reranker(cfg) if cfg.rerank.enabled else None
    docs = json.loads((Path(cfg.data.eval_dir) / "docs.json").read_text(encoding="utf-8"))
    known = sorted({d["company"] for d in docs.values()})
    parents = load_parents(cfg) if cfg.generation.expand_tables else None
    cache = SemanticCache(cfg, embedder) if with_cache else None
    return QueryPipeline(cfg, retriever, reranker, LLMClient(cfg), known, closers=[store.close], parents=parents, cache=cache)


_default: QueryPipeline | None = None


def run_query(q: str, user: str | None = None) -> QueryResult:
    """Answer one question with the default pipeline (built lazily, once, with the answer cache)."""
    global _default
    if _default is None:
        _default = build_pipeline(with_cache=True)
    return _default.run(q, user).result
