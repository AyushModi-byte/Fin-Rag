"""FastAPI app: POST /query, GET /health. Serves queries only; indexing stays in the offline job.

The pipeline (embedding model, index, LLM client) is built on the first query, not at import, so
/health answers instantly and tests can inject a fake pipeline.
"""
import logging
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.config import load_config
from src.llm.llm_client import BudgetNearlyExhausted, LLMError
from src.observability.tracing import setup_trace_logging
from src.schemas.models import QueryResult

log = logging.getLogger("finrag.api")


class QueryRequest(BaseModel):
    question: str
    user: str | None = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v


def _default_factory(cfg):
    from src.pipelines.query_pipeline import build_pipeline

    return build_pipeline(cfg, with_cache=True)


def create_app(pipeline=None, cfg=None, pipeline_factory=_default_factory) -> FastAPI:
    cfg = cfg or load_config()
    setup_trace_logging(cfg.data.log_file)
    max_chars = cfg.api.max_question_chars
    state = {"pipeline": pipeline}
    # The embedded Qdrant store and the BM25 index are not built for concurrent use: one query at a time.
    lock = threading.Lock()

    app = FastAPI(title="FinRag", description="RAG over SEC 10-K filings (FinanceBench subset)")

    def get_pipeline():
        if state["pipeline"] is None:
            state["pipeline"] = pipeline_factory(cfg)
        return state["pipeline"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResult)
    def query(req: QueryRequest) -> QueryResult:
        if len(req.question) > max_chars:
            raise HTTPException(status_code=422, detail=f"question longer than {max_chars} characters")
        try:
            with lock:
                return get_pipeline().run(req.question, req.user).result
        except BudgetNearlyExhausted as e:
            log.warning("LLM budget nearly exhausted: %s", e)
            raise HTTPException(status_code=503, detail="LLM daily budget nearly exhausted; try later") from e
        except LLMError as e:
            log.warning("LLM failure: %s", e)
            raise HTTPException(status_code=503, detail="LLM temporarily unavailable; try later") from e

    return app
