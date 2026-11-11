"""FastAPI app: POST /query, GET /health. Serves queries only; indexing stays in the offline job.

The pipeline (embedding model, index, LLM client) is built on the first query, not at import, so
/health answers instantly and tests can inject a fake pipeline.

Public-API guard on POST /query, in this order (a request stopped by a step never reaches the next):
  1. API key   - `api.require_api_key` closes the API to callers without X-API-Key (401).
  2. Per-IP    - sliding-window limit per client IP (429 + Retry-After); a valid owner key skips it.
  3. Daily cap - `api.daily_global_cap` accepted calls per UTC day over all clients (503 + Retry-After),
                 protecting the free-tier Groq token budget.
Rate-limited and unauthorised calls cost nothing. Every guarded call is logged with a hashed client label, never the
address or the question. CORS is allowed only for `api.cors_origins` (+ FINRAG_CORS_ORIGINS).

With `api.frontend_dir` set, the API also serves the web UI at / (same origin, so no CORS is needed), and every response
carries security headers and a strict Content-Security-Policy.
"""
import hmac
import logging
import os
import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from src.config import load_config
from src.llm.llm_client import BudgetNearlyExhausted, LLMError
from src.observability.tracing import log_event, setup_trace_logging
from src.schemas.models import QueryResult
from src.security.rate_limit import DailyCap, RateLimiter, client_ip, client_label

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


# No inline script or style, nothing from other origins: the UI is plain files and calls only its own API.
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}
CSP_EXEMPT = ("/docs", "/redoc", "/openapi.json")  # Swagger UI loads its script from a CDN


def _origins(cfg, env) -> list[str]:
    extra = [o.strip() for o in env.get(cfg.api.cors_origins_env, "").split(",") if o.strip()]
    return [*cfg.api.cors_origins, *extra]


def create_app(pipeline=None, cfg=None, pipeline_factory=_default_factory, clock=time.time, env=None) -> FastAPI:
    cfg = cfg or load_config()
    env = os.environ if env is None else env
    setup_trace_logging(cfg.data.log_file)
    max_chars = cfg.api.max_question_chars
    api = cfg.api
    owner_key = env.get(api.api_key_env, "")
    salt = env.get(api.log_salt_env) or secrets.token_hex(16)
    limiter = RateLimiter(api.rate_limit.per_minute, api.rate_limit.per_day, api.rate_limit.max_clients, clock)
    cap = DailyCap(api.daily_global_cap, api.usage_file, clock)
    if api.require_api_key and not owner_key:
        log.warning("require_api_key is on but %s is not set: every /query will be refused", api.api_key_env)
    state = {"pipeline": pipeline}
    # The embedded Qdrant store and the BM25 index are not built for concurrent use: one query at a time.
    lock = threading.Lock()

    app = FastAPI(title="FinRag", description="RAG over SEC 10-K filings (FinanceBench subset)")

    def refuse(status: int, detail: str, retry_after: int | None = None) -> JSONResponse:
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        return JSONResponse({"detail": detail}, status_code=status, headers=headers)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.method != "POST" or request.url.path != "/query":
            return await call_next(request)
        start = time.perf_counter()
        ip = client_ip(request.headers, request.client.host if request.client else None, api.trusted_proxy_hops)
        provided = request.headers.get("x-api-key", "")
        owner = bool(owner_key) and hmac.compare_digest(provided.encode(), owner_key.encode())
        limited = None
        if api.require_api_key and not owner:
            response = refuse(401, "A valid X-API-Key header is required")
        else:
            decision = None if owner else limiter.check(ip)
            if decision is not None and not decision.allowed:
                limited = decision.reason
                response = refuse(429, f"Rate limit reached ({decision.reason}); slow down", decision.retry_after)
            elif not cap.try_acquire():
                limited = "cap"
                response = refuse(503, "Daily capacity reached; try again tomorrow", cap.seconds_to_reset())
            else:
                response = await call_next(request)
        log_event(
            "request", client=client_label(ip, salt), method="POST", path="/query", status=response.status_code,
            ms=round((time.perf_counter() - start) * 1000, 1), auth=owner, limited=limited,
        )
        return response

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)  # added after `guard`, so it also covers its 401/429/503 answers
        for k, v in SECURITY_HEADERS.items():
            response.headers[k] = v
        if not request.url.path.startswith(CSP_EXEMPT):
            response.headers["Content-Security-Policy"] = CSP
        return response

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

    ui_dir = getattr(api, "frontend_dir", None)
    if ui_dir and Path(ui_dir).is_dir():  # mounted last: the API routes above always win
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

    origins = _origins(cfg, env)
    if origins:  # added last = outermost, so preflights, 401s, 429s and 503s all carry CORS headers
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-API-Key"], expose_headers=["Retry-After"], allow_credentials=False,
        )
    return app
