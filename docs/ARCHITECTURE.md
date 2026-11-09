# FinRag Architecture

## Two pipelines

```
OFFLINE (src/pipelines/index_pipeline.py, run via src/jobs/index_corpus.py)
data/raw/*.pdf -> ingestion/parsers -> data/parsed/*.jsonl
   -> chunking/chunker (parent page + child chunks + context prefix)
   -> embeddings/embedder -> vector_db/vector_store (Qdrant local)
   -> lexical/bm25_index (bm25s)            => .cache/index/<version>/

ONLINE (src/pipelines/query_pipeline.py, served by src/api/routes.py)
question -> security/acl + redact -> cache/semantic_cache (hit? return)
   -> query/query_router (company + fiscal year filter, in_scope?)
   -> retrieval/retriever (dense top_k + BM25 top_k -> RRF)
   -> rerank/reranker (top_n)
   -> generation/grounded_answer (llm/llm_client, prompts/prompt_templates)
   -> QueryResult (answer, citations, contexts, latency_ms) -> cache.put
   (observability/tracing wraps every stage)
```

## Module responsibilities

| Folder | File | Responsibility |
|---|---|---|
| schemas | models.py | Pydantic models shared by all modules |
| ingestion | parsers.py, loader.py | PDF -> per-page markdown (tables kept) |
| chunking | chunker.py | Parent/child chunks, never split a table, context prefix |
| embeddings | embedder.py | fastembed wrapper, records model id |
| vector_db | vector_store.py | Qdrant local: upsert, filtered search |
| lexical | bm25_index.py | BM25 build/search/persist, same filters |
| retrieval | retriever.py | Hybrid search + RRF fusion |
| rerank | reranker.py | Cross-encoder rerank of top candidates |
| query | query_router.py | Rule-based company/year detection, scope check |
| cache | semantic_cache.py | Exact then similarity cache |
| prompts | prompt_templates.py | Versioned prompt templates |
| llm | llm_client.py | Gemini client: retries, timeout, fallback, disk cache, throttle |
| generation | grounded_answer.py | Answer with `[doc_id p.N]` citations or `I don't know` |
| eval | rag_metrics.py | recall@k, MRR, LLM-judge accuracy, faithfulness |
| observability | tracing.py | Per-stage latency, JSON logs |
| security | acl.py | Per-user company ACL, PII redaction |
| api | routes.py | `POST /query`, `GET /health` |
| pipelines | index_pipeline.py, query_pipeline.py | Orchestration only, no logic |
| jobs | prepare_data.py, index_corpus.py, eval_nightly.py | CLI entry points |

## Schemas (src/schemas/models.py)

- `Page(doc_id, page, markdown)`
- `DocMeta(doc_id, company, fiscal_year, doc_type)`
- `Parent(id, doc_id, page_start, page_end, text)`
- `Chunk(id, doc_id, parent_id, text, context, page, company, fiscal_year)`
- `Hit(chunk_id, score, chunk)`
- `Citation(doc_id, page)`
- `QueryResult(answer, citations, contexts, latency_ms)`

## config.yaml defaults (starting values; tune in Task 9)

```yaml
index: {version: v1, dir: .cache/index}
embeddings: {model: BAAI/bge-small-en-v1.5, batch_size: 32}
chunking: {child_tokens: 256, overlap_tokens: 32, context_prefix: true}
retrieval: {top_k: 20, rrf_k: 60, use_router_filter: true}
rerank: {enabled: true, model: Xenova/ms-marco-MiniLM-L-6-v2, top_n: 5}
llm:
  primary: {provider: gemini, model: gemini-2.5-flash, rpm: 10, daily_budget: 200}
  fallback: {provider: groq, model: llama-3.3-70b-versatile, rpm: 25, daily_budget: 1000}
  temperature: 0
  timeout_s: 30
  max_retries: 2          # per provider, with exponential backoff, then switch provider
  cache_dir: .cache/llm
  budget_warn_ratio: 0.8  # stop and tell the user at 80% of a daily budget
cache: {enabled: true, threshold: 0.95}
acl: {default: "*", users: {}}
eval: {dev_ratio: 0.5, seed: 42}
```

Check the current Gemini and Groq free-tier model names and rate limits before Task 7 and update `llm.*` (models, rpm, daily_budget) to match. Budgets above are placeholders.

## LLM call policy

Order per request: disk cache -> primary (Gemini) -> on 429/5xx/timeout, backoff and retry up to `max_retries` -> fallback (Groq). Cache key includes provider and model, so metrics note which model answered. Record the provider used in `QueryResult.latency_ms` metadata/logs. For final reported eval numbers, note the share of answers served by each provider; if the fallback served more than ~10%, re-run those questions on the primary once the limit resets so the numbers describe one model.

## Eval definitions

- **Recall@k (page-level):** 1 if any of the top-k hits' pages is in gold pages for the right doc, averaged.
- **MRR:** reciprocal rank of the first hit on a gold page.
- **Accuracy:** LLM judge compares prediction to gold answer (numbers within rounding, same units).
- **Faithfulness:** LLM judge checks each claim in the answer is supported by the retrieved contexts, score = supported / total.
- **Latency:** p50/p95 per stage and end to end.

## Key design decisions

- Company/fiscal-year metadata filter: FinanceBench questions name them, so filtering removes most distractors.
- Context prefix on every chunk keeps year and company in the embedding.
- Tables stay whole: numeric questions depend on them.
- Own faithfulness judge instead of the RAGAS library: fewer dependencies, fewer rate-limit failures.
- Disk-cached LLM calls: deterministic reruns, free retries.
- Embedded Qdrant and file caches: zero infrastructure, simple Docker.

## Known limitations (for the README)

- Small subset (~10 companies); results do not generalize to all filings.
- LLM-as-judge is not human evaluation.
- Scanned or image-only pages are skipped (no OCR).
