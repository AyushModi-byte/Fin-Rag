# FinRag: Production RAG over SEC 10-Ks (Small Subset) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A production-style RAG system (hybrid search, rerank, grounded citations, eval harness, API, Docker) over ~10 companies' 10-Ks from FinanceBench, finished fast, with honestly measured metrics for the resume.

**Architecture:** Two pipelines: offline index (PDF -> pages -> parent/child chunks with context prefix -> dense + BM25 indexes) and online query (route/filter -> hybrid retrieve + RRF -> cross-encoder rerank -> grounded answer with page citations). Frozen eval set with a dev/test split drives every tuning decision.

**Tech Stack:** Python 3.11, PyMuPDF/pymupdf4llm, fastembed (bge-small-en-v1.5, ONNX/CPU), Qdrant local mode, bm25s, fastembed cross-encoder reranker, Gemini Flash free tier primary with Groq fallback (model names in `config.yaml`), FastAPI, pytest, Docker Compose.

**Spec:** User's chat request + folder structure image ("RAG Project Structure - Scalable"). Folder layout below follows the image.

## Global Constraints

- Follow the image layout: `data/{raw,parsed,eval}`, `src/{schemas,ingestion,chunking,embeddings,vector_db,lexical,retrieval,rerank,query,cache,prompts,llm,generation,eval,observability,security,api,pipelines,jobs}`, `tests/`, `logs/`, `main.py`, `config.yaml`, `.env`, `.gitignore`, `Dockerfile`, `docker-compose.yaml`, `requirements.txt`, `README.md`.
- No GPU. CPU-only models, small. No Redis/Qdrant server needed (embedded Qdrant, file cache).
- Free tiers only: Gemini primary, Groq fallback (`GEMINI_API_KEY`, `GROQ_API_KEY` in `.env`). Every LLM call goes through `LLMClient` with a disk cache, so reruns cost nothing and are deterministic (temperature 0).
- Do not abuse the keys: no looped or redundant calls, smoke tests with `--limit 5`, full dev eval only after a real change. The user must never be blocked waiting on rate limits: throttle, cache, fall back, and do offline work if both providers are limited.
- All config (models, chunk size, k, weights, index version) lives in `config.yaml`. No magic numbers in code.
- Pin exact versions in `requirements.txt` (`pip freeze` after first working install).
- Python `-I`-safe paths: scripts take paths as arguments; downloaded PDFs only in `data/raw/`.

### Anti-failure rules (why the last attempt went wrong)

1. **One change at a time, measured.** Every tuning change is logged in `docs/experiments.md`: change, dev metrics before/after, keep or `git revert`. Worse metrics = revert, never "try again on top".
2. **Two-strike rule.** If a step fails twice with the same error, stop, write down the error and what was tried, and ask the user. No third blind retry.
3. **Commit after every green task.** Always have a known-good state to revert to.
4. **Build the naive end-to-end baseline first (Tasks 1-8), then improve.** No tuning before a baseline number exists.
5. **Test split is untouched until Task 11.** Tune only on dev; report final numbers on test.

## Review Focus

- Question names a company/year not in the corpus: answer "I don't know", no fabricated number.
- Empty or whitespace-only query: HTTP 422, not a crash.
- PDF page with a table and no text layer / parser returns empty page: skipped and logged, indexing continues.
- LLM rate-limit (429) or timeout: retry with backoff, then fallback model, then clean error; never hang.
- Question with no answer in retrieved context: answer says "I don't know", citations empty.

---

### Task 1: Scaffold, config, schemas

**Files:**
- Create: whole folder skeleton from the image (empty `__init__.py` per package), `config.yaml`, `.gitignore` (ignores `.env`, `data/raw`, `.cache/`, `logs/`), `requirements.txt`, `src/schemas/models.py`, `src/config.py`, `tests/test_config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces (`src/schemas/models.py`, pydantic): `Page(doc_id:str, page:int, markdown:str)`, `DocMeta(doc_id:str, company:str, fiscal_year:int, doc_type:str)`, `Parent(id:str, doc_id:str, page_start:int, page_end:int, text:str)`, `Chunk(id:str, doc_id:str, parent_id:str, text:str, context:str, page:int, company:str, fiscal_year:int)`, `Hit(chunk_id:str, score:float, chunk:Chunk)`, `Citation(doc_id:str, page:int)`, `QueryResult(answer:str, citations:list[Citation], contexts:list[str], latency_ms:dict[str,float])`
- Produces (`src/config.py`): `load_config(path: str = "config.yaml") -> Config` (nested attrs, e.g. `cfg.retrieval.top_k`)

- [ ] **Step 1: Write failing tests** `test_load_config_has_required_keys` (asserts `cfg.embeddings.model`, `cfg.retrieval.top_k == 20`, `cfg.rerank.top_n == 5`, `cfg.index.version`), `test_chunk_requires_page` (omitting `page` raises `ValidationError`).
- [ ] **Step 2: Implement** models and `load_config`; `.env` loaded with python-dotenv for `GEMINI_API_KEY`.
- [ ] **Step 3: Verify** `pytest tests/test_config.py -v` passes; `git init`, commit `chore: scaffold`.

### Task 2: Data subset and eval set

**Files:**
- Create: `src/jobs/prepare_data.py`, `data/eval/` outputs, `tests/test_prepare_data.py`
- Test: `tests/test_prepare_data.py`

**Interfaces:**
- Produces: `data/raw/*.pdf` (~12-15 filings, ~10 companies); `data/eval/questions.jsonl` rows `{id, question, gold_answer, doc_id, gold_pages:list[int], split:"dev"|"test"}`; `data/eval/docs.json` mapping `doc_id -> DocMeta`.
- Function: `select_docs(questions: list[dict], n_companies: int = 10) -> list[str]` picks companies with the most FinanceBench questions, 10-K only.

- [ ] **Step 1: Write failing tests**: `test_every_eval_doc_exists_on_disk`, `test_split_is_deterministic_and_disjoint` (fixed seed, ~50/50, no id in both), `test_min_questions` (>= 50 total).
- [ ] **Step 2: Implement**: download FinanceBench open-source set (Hugging Face `PatronusAI/financebench`) and its PDFs from the Patronus GitHub repo; keep only chosen docs; write eval JSONL with page-level gold evidence.
- [ ] **Step 3: Verify** tests pass; print and save question count per split. If total < 50, raise `n_companies` (don't proceed with fewer). Commit.

### Task 3: PDF ingestion

**Files:**
- Create: `src/ingestion/parsers.py`, `src/ingestion/loader.py`, `tests/test_parsers.py` (+ one tiny fixture PDF `tests/fixtures/sample.pdf`)
- Test: `tests/test_parsers.py`

**Interfaces:**
- Produces: `parse_pdf(path: Path) -> list[Page]` (pymupdf4llm per-page markdown, tables preserved as markdown); `load_corpus(raw_dir: Path, docs_json: Path) -> list[tuple[DocMeta, list[Page]]]`; writes parsed JSONL to `data/parsed/`.

- [ ] **Step 1: Write failing tests**: `test_parse_returns_one_page_per_pdf_page` (1-based page numbers), `test_empty_page_skipped_not_raised`.
- [ ] **Step 2: Implement**; log skipped pages to `logs/app.log`.
- [ ] **Step 3: Verify** tests; run on real PDFs; check one table page by eye contains markdown rows. Commit.

### Task 4: Chunking (parent-child + context prefix)

**Files:**
- Create: `src/chunking/chunker.py`, `tests/test_chunker.py`

**Interfaces:**
- Consumes: `Page`, `DocMeta`, `Parent`, `Chunk`.
- Produces: `chunk_pages(pages: list[Page], meta: DocMeta, cfg) -> tuple[list[Parent], list[Chunk]]`. Child chunk ~`cfg.chunking.child_tokens` (start 256), parent = page-level span. Markdown tables are never split. Each chunk's `context` = `"{company} {fiscal_year} 10-K, page {page}"`; the embedded text is `context + "\n" + text`.

- [ ] **Step 1: Write failing tests**: `test_table_not_split`, `test_every_chunk_has_parent_and_page`, `test_context_prefix_contains_company_and_year`, `test_chunk_ids_unique_and_stable` (same input twice gives same ids).
- [ ] **Step 2: Implement** with deterministic id = hash of `doc_id|page|offset`.
- [ ] **Step 3: Verify** and commit.

### Task 5: Embeddings, dense store, BM25, index pipeline

**Files:**
- Create: `src/embeddings/embedder.py`, `src/vector_db/vector_store.py`, `src/lexical/bm25_index.py`, `src/pipelines/index_pipeline.py`, `src/jobs/index_corpus.py`, `tests/test_index.py`

**Interfaces:**
- Produces: `Embedder.embed(texts: list[str]) -> np.ndarray` (stores `model_id`), `VectorStore.upsert(chunks, vectors)`, `VectorStore.search(vec, k:int, filters: dict|None) -> list[Hit]`, `BM25Index.build(chunks)`, `BM25Index.search(q:str, k:int, filters: dict|None) -> list[Hit]` (persists to `.cache/index/{cfg.index.version}/`), `build_index() -> dict` (counts), `python -m src.jobs.index_corpus`.

- [ ] **Step 1: Write failing tests** on a 5-chunk fixture: `test_dense_search_finds_exact_chunk`, `test_bm25_finds_keyword_chunk`, `test_company_filter_excludes_other_companies` (both stores), `test_index_rebuild_is_idempotent` (count unchanged).
- [ ] **Step 2: Implement**; embedding model id and index version written to a manifest, load refuses on mismatch.
- [ ] **Step 3: Verify** tests; run the job on the real corpus; note chunk count and time in `docs/experiments.md`. Commit.

### Task 6: Hybrid retrieval, rerank, query routing

**Files:**
- Create: `src/retrieval/retriever.py`, `src/rerank/reranker.py`, `src/query/query_router.py`, `tests/test_retrieval.py`

**Interfaces:**
- Produces: `rrf(result_lists: list[list[Hit]], k: int = 60) -> list[Hit]`; `HybridRetriever.retrieve(q:str, k:int, filters: dict|None) -> list[Hit]`; `Reranker.rerank(q:str, hits: list[Hit], top_n:int) -> list[Hit]`; `route(q:str, known_companies: list[str]) -> RoutedQuery(filters: dict, query: str, in_scope: bool)` (rule-based company + fiscal-year detection from the question text; LLM fallback only if rules find nothing).

- [ ] **Step 1: Write failing tests**: `test_rrf_ranks_doc_in_both_lists_first`, `test_route_extracts_company_and_year` ("3M FY2018 capex" gives `{company:"3M", fiscal_year:2018}`), `test_route_unknown_company_in_scope_false`, `test_rerank_returns_top_n_sorted`.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Verify** and commit.

### Task 7: LLM client, prompts, grounded generation

**Files:**
- Create: `src/llm/llm_client.py`, `src/prompts/prompt_templates.py`, `src/generation/grounded_answer.py`, `tests/test_llm_generation.py`

**Interfaces:**
- Produces: `LLMClient.complete(prompt:str, system:str|None=None) -> str` (timeout from config, `max_retries` with exponential backoff on 429/5xx, then automatic fallback Gemini -> Groq, on-disk cache keyed by hash of provider+model+system+prompt, per-provider request throttle under free-tier RPM, per-provider daily call counter that raises `BudgetNearlyExhausted` at `budget_warn_ratio` of `daily_budget`); `generate(q:str, hits: list[Hit], llm: LLMClient) -> QueryResult`. Prompt requires: use only the context, show the calculation for numeric questions, cite `[doc_id p.N]`, otherwise reply exactly `I don't know`.

- [ ] **Step 1: Write failing tests** with a fake HTTP layer: `test_retries_on_429_then_succeeds`, `test_falls_back_to_groq_after_gemini_retries_exhausted`, `test_budget_counter_blocks_at_80_percent`, `test_cache_hit_makes_no_second_call`, `test_generate_no_context_says_i_dont_know`, `test_citations_parsed_into_page_numbers`.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Verify**, then one real call as a smoke test. Commit.

### Task 8: Eval harness and naive baseline

**Files:**
- Create: `src/eval/rag_metrics.py`, `src/jobs/eval_nightly.py` (CLI: `python -m src.jobs.eval_nightly --split dev`), `src/pipelines/query_pipeline.py`, `tests/test_metrics.py`, `docs/experiments.md`

**Interfaces:**
- Produces: `recall_at_k(hits, gold_pages, k:int) -> float` (page-level), `mrr(hits, gold_pages) -> float`, `judge_correct(q, gold, pred, llm) -> bool`, `judge_faithful(answer, contexts, llm) -> float` (RAGAS-style claim check, own prompt, one call per question, avoids RAGAS dependency churn), `run_eval(split:str) -> dict` (writes `data/eval/runs/{timestamp}.json`), `run_query(q:str, user: str|None=None) -> QueryResult`.

- [ ] **Step 1: Write failing tests** with hand-made hits: `test_recall_at_k_hit_and_miss`, `test_mrr_rank_two_is_half`.
- [ ] **Step 2: Implement** harness; run baseline on dev (dense + BM25 + RRF, no rerank, no routing). Record recall@5, MRR, answer accuracy, faithfulness, p50 latency in `docs/experiments.md` as **Baseline**.
- [ ] **Step 3: Verify** the run completes with zero errors; commit. This is the number everything else must beat.

### Task 9: Measured tuning loop (dev split only)

**Files:**
- Modify: `config.yaml` values only (plus small code fixes found); log in `docs/experiments.md`

Run these in order, one at a time, each: change, `eval --split dev`, keep only if recall@5 or accuracy improves, else revert.

- [ ] Metadata filter via `route` on/off (expect the biggest recall gain).
- [ ] Rerank on/off, `top_k` 20/40 -> `top_n` 5/8.
- [ ] Child chunk size 256/384/512, context prefix on/off.
- [ ] Dense/BM25 weighting (RRF vs weighted) and parent-page expansion for generation.
- [ ] Prompt tweaks for numeric reasoning (show formula; units and fiscal-year check). Failure analysis: list the 10 worst dev questions, bucket the cause (retrieval miss, parse/table issue, reasoning error, judge error), fix the biggest bucket.
- [ ] Stop at diminishing returns (< 1 point gain for two consecutive changes). Commit the final config.

### Task 10: Cache, observability, security, API

**Files:**
- Create: `src/cache/semantic_cache.py`, `src/observability/tracing.py`, `src/security/acl.py`, `src/api/routes.py`, `main.py`, `tests/test_api.py`

**Interfaces:**
- Produces: `SemanticCache.get(q:str) -> QueryResult|None` / `.put(q, result)` (exact-match first, then embedding similarity >= `cfg.cache.threshold`, default 0.95, in-memory plus JSON file); `trace(stage:str)` context manager recording per-stage latency into `QueryResult.latency_ms` and `logs/app.log` (JSON lines); `redact_pii(text:str) -> str` (emails, phone numbers) and `allowed_companies(user:str) -> set[str]` (ACL from `config.yaml`; retrieval filter intersects with it); FastAPI `POST /query {question, user?} -> QueryResult`, `GET /health`.

- [ ] **Step 1: Write failing tests**: `test_empty_question_returns_422`, `test_health_ok`, `test_acl_blocks_disallowed_company` (returns "I don't know", no contexts), `test_semantic_cache_hit_skips_llm`, `test_pii_redacted_in_logs`.
- [ ] **Step 2: Implement**; API serves queries only, indexing stays in the offline job.
- [ ] **Step 3: Verify** with `pytest -q` (whole suite green) and `curl` one real question. Re-run dev eval: metrics must not drop. Commit.

### Task 11: Docker, final test-split numbers, README, deploy

**Files:**
- Create: `Dockerfile`, `docker-compose.yaml`, `README.md`, update `docs/experiments.md`

- [ ] **Step 1: Docker**: image builds, `docker compose up` serves `/health` and one `/query`; prebuilt index baked in or mounted.
- [ ] **Step 2: Final eval on the test split, once.** Report recall@5, MRR, accuracy, faithfulness, p50/p95 latency, and the ablation table (baseline vs each kept change). Do not tune after this.
- [ ] **Step 3: README**: architecture diagram, run instructions, results table with dev vs test, limitations (subset size, LLM-judge caveat). Deploy to Hugging Face Spaces (Docker) with `GEMINI_API_KEY` as a secret. Commit and tag `v1.0`.

## Target Metrics (goals, not promises; report what is measured)

- Page-level recall@5: >= 0.80 (with company/year filter + rerank)
- Answer accuracy (LLM-judged vs gold): 60-75% on test
- Faithfulness: >= 0.90
- p50 latency: < 4 s CPU + free API; cache hits < 100 ms

## Self-Review Notes

- Coverage: every folder in the image has an owning task (schemas T1, ingestion T3, chunking T4, embeddings/vector_db/lexical/pipelines/jobs T5, retrieval/rerank/query T6, llm/prompts/generation T7, eval T8, cache/observability/security/api T10, Docker/README T11).
- Signatures were checked across tasks (`Hit`, `Chunk`, `QueryResult`, `LLMClient.complete`).
