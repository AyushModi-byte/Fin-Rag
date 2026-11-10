# FinRag Architecture

State: index **v2**, final dev config of experiment 12 (see `docs/experiments.md` for every measurement).
Rollback: `index.version: v1` via `config.v1.yaml` (`python -m src.jobs.<job> --config config.v1.yaml`).

## Two pipelines

```
OFFLINE (src/pipelines/index_pipeline.py, run via src/jobs/index_corpus.py)
data/raw/*.pdf -> ingestion/parsers -> data/parsed/*.jsonl
   -> chunking/chunker v2 (page parents, table parents, child chunks, context prefix;
                           sized with the EMBEDDER'S tokenizer)
   -> embeddings/embedder (mxbai-embed-large-v1) -> shards/*.npz (resumable)
   -> vector_db/vector_store (embedded Qdrant, bulk load) + lexical/bm25_index (bm25s, stemmed)
   -> parents.jsonl + manifest.json                     => .cache/index/<version>/

ONLINE (src/pipelines/query_pipeline.py, served by src/api/routes.py)
question -> security/acl + redact -> cache/semantic_cache (hit? return)
   -> query/query_router (company + fiscal year filter, in_scope?)
   -> retrieval/retriever (dense top_k + BM25 top_k -> weighted RRF 1.0 : 0.25)
   -> [rerank/reranker: OFF]  -> top_n chunks
   -> [generation/context_expansion: OFF]
   -> generation/grounded_answer (llm/llm_client, prompts/prompt_templates grounded_v1)
   -> QueryResult (answer, citations, contexts, latency_ms) -> cache.put
   (observability/tracing wraps every stage)
```

## Module responsibilities

| Folder | File | Responsibility |
|---|---|---|
| schemas | models.py | Pydantic models shared by all modules |
| ingestion | parsers.py, loader.py | PDF -> per-page markdown (tables kept) |
| chunking | chunker.py | v2 parent/child chunks: embedder-tokenizer sizing, table row groups with repeated headers, rich prefix, tiny-chunk drop |
| embeddings | embedder.py | fastembed wrapper: batch 1 + length sorting, query/doc prefixes from config, records model id |
| embeddings | tokenizer.py | The embedding model's own tokenizer (`tokenizer.json`, no ONNX session), used for chunk sizing |
| vector_db | vector_store.py | Qdrant local: upsert, filtered search, `bulk_load()` (see Index build) |
| lexical | bm25_index.py, tokenize.py | BM25 build/search/persist with the same filters; v2 tokenizer (stemming, numbers kept) |
| retrieval | retriever.py | Hybrid search, weighted RRF, tiny-chunk filter |
| rerank | reranker.py | Cross-encoder rerank (disabled, see Decisions) |
| query | query_router.py | Rule-based company/year detection, scope check |
| cache | semantic_cache.py | Exact then similarity cache |
| prompts | prompt_templates.py | Versioned prompts (`grounded_v1` active, `grounded_v2` tested and rejected) |
| llm | llm_client.py | Groq client: key rotation, retries, backoff with provider hints, disk cache, throttle, per-key daily request budget |
| generation | grounded_answer.py | Answer with `[doc_id p.N]` citations or `I don't know` |
| generation | context_expansion.py | Replace a retrieved table row group with its full table (flag `generation.expand_tables`, off) |
| eval | rag_metrics.py | recall@k, MRR, LLM-judge accuracy, faithfulness |
| observability | tracing.py | Per-stage latency, JSON logs |
| security | acl.py | Per-user company ACL, PII redaction |
| api | routes.py | `POST /query`, `GET /health` |
| pipelines | index_pipeline.py, query_pipeline.py | Orchestration only, no logic |
| jobs | prepare_data.py, index_corpus.py, eval_nightly.py | CLI entry points |
| jobs | bench_embed.py, pilot_bakeoff.py, compare_indexes.py, fusion_sweep.py, rerank_dev.py | Offline measurement tools (no LLM calls) behind the experiments log |

## Schemas (src/schemas/models.py)

- `Page(doc_id, page, markdown)`
- `DocMeta(doc_id, company, fiscal_year, doc_type)`
- `Parent(id, doc_id, page_start, page_end, text)`: one per page, plus one per table that had to be split (id = the `table_id` of its row groups)
- `Chunk(id, doc_id, parent_id, text, context, page, company, fiscal_year, table_id=None, row_group=None)`
- `Hit(chunk_id, score, chunk)`
- `Citation(doc_id, page)`
- `QueryResult(answer, citations, contexts, latency_ms)`

## Chunking v2 (src/chunking/chunker.py)

- **Sizing uses the embedder's own tokenizer.** v1 measured length with a regex, so 2.6% of chunks, and the gold page of 17 of 29 dev questions, held a table over the embedder's 512-token window: the bottom of the table was never embedded. v2 counts tokens with `tokenizer.json` of the embedding model, prefix included, and no chunk may exceed `chunking.max_embed_tokens` (480). The chunker takes the tokenizer as a parameter (`tok`); tests use a regex stand-in, the pipeline passes `EmbedTokenizer`. The regex count (`count_tokens`) is still used for two cheap things: the retrieval-time tiny-chunk filter and the generation context cap. It is not the measure that guarantees the embedder never truncates.
- **Prose:** paragraphs are packed up to `child_tokens` (300, minus the prefix); a paragraph over budget is cut into sliding token windows with `overlap_tokens` (40).
- **Tables:** a table that fits the cap stays whole (own chunk, `table_id` set). A larger table is split into **row groups**: a row is never split, the header rows (up to `max_header_rows`, including bold continuation rows) are repeated in every group, groups are balanced, and the whole table is stored as a `Parent` (`parent_id == table_id`) for later expansion. A header larger than half a group is not repeated; a single row wider than the cap is windowed. `chunking.table_group_tokens` (unset) can set a smaller target size.
- **Prefix on every chunk, deterministic:** `{company} FY{year} 10-K | page {n} | {nearest preceding heading} | {units}`. Heading: markdown headings, bold-only lines, ALL-CAPS lines, "Item N"/"Note N", carried over from the previous page; units from "in millions", "$ millions", "(Millions of Dollars)" and similar, no LLM.
- **Dropped at index time:** chunks under `chunking.min_tokens` (15): page numbers, "ii". The retrieval-time filter (`retrieval.min_chunk_tokens`) stays.
- Ids are a hash of (doc, page, start offset [, row group]), so rebuilds are stable.

## Embedder and lexical index

- Embedder: `mixedbread-ai/mxbai-embed-large-v1` (1024-d, CPU/ONNX via fastembed), **no query or doc prefix** (the instruction lowered MRR in the pilot and on all 29 dev questions). Speed on this CPU: batch size 1 with length sorting is 3.7x faster than batch 32; threads and fastembed `parallel` did not help.
- BM25 v2: bm25s over `context + text`, tokens lower-cased, stop-worded, Snowball-stemmed (`py_rust_stemmers`); `3m`, `10-k` (also `10`), `1,234.5` (also `1234.5`), `62%` (also `62`) stay searchable. v1 indexes load with the v1 tokenizer because the manifest records it.

## Index build (src/pipelines/index_pipeline.py)

- **Resumable:** embeddings are written to `shards/shard_NNNNN.npz` (500 chunks each, atomically). On restart a shard is reused only if its signature (model, prefixes, chunk ids and texts) still matches. Qdrant and BM25 are then rebuilt from the shards, so a finished index re-runs in seconds.
- **Time gate:** a probe embeds `reindex.probe_chunks` random chunks, projects the full time by text length, and stops with `ReindexTooSlow` above `reindex.max_hours`.
- **Manifest** (written last; a missing manifest means an incomplete build): model, dimension, doc/query prefix, tokenizer file hash, chunking parameters, BM25 tokenizer, counts, build time. `load_index` refuses a model, version, doc-prefix, chunking or BM25-tokenizer mismatch. v1 manifests lack the newer keys and still load.
- **Protected versions:** `index.protected_versions: [v1]` cannot be built into.
- **Embedded Qdrant is slow by default on this machine:** it commits and fsyncs once per point (1.3 points/s at 5,000 points). `VectorStore.bulk_load()` relaxes sqlite durability during the load (16,101 points in about 7 s) and restores it afterwards; safe because the store is rebuilt from the shards.
- v2 build: 16,101 chunks, 4,787 parents, 294 MB; 96 min of embedding on CPU (2.2 to 2.8 chunks/s).

## Retrieval and generation

- Router: company and fiscal-year metadata filter. Two-digit years ("FY22") are read; when several years are named the latest wins (the latest 10-K carries the prior-year comparatives).
- Fusion: weighted RRF, `score = sum(weight / (rrf_k + rank))`, dense 1.0 and BM25 0.25. With mxbai-large the dense ranking is stronger, and down-weighting BM25 improved every metric.
- Tiny chunks (under 15 regex tokens) are dropped from both lists before fusion.
- Generation: top_n = 5 chunks, prompt `grounded_v1`, `[doc_id p.N]` citations or exactly `I don't know`.

## config.yaml (final values; every setting lives in config.yaml, no magic numbers)

```yaml
index: {version: v2, dir: .cache/index, collection: finrag_chunks, shard_size: 500, upsert_batch: 1000, protected_versions: [v1]}
reindex: {max_hours: 2.5, probe_chunks: 60}
lexical: {stopwords: en, tokenizer: v2, stem: true}
embeddings: {model: mixedbread-ai/mxbai-embed-large-v1, batch_size: 1, sort_by_length: true, threads: null, query_prefix: "", doc_prefix: ""}
chunking: {child_tokens: 300, overlap_tokens: 40, max_embed_tokens: 480, min_tokens: 15, max_header_rows: 3, max_heading_chars: 80, context_prefix: true}
retrieval: {top_k: 20, rrf_k: 60, dense_weight: 1.0, bm25_weight: 0.25, use_router_filter: true, min_chunk_tokens: 15}
rerank: {enabled: false, model: Xenova/ms-marco-MiniLM-L-6-v2, top_n: 5}   # top_n is also the context size given to the LLM
generation: {expand_tables: false, max_context_tokens: 4000, prompt_version: grounded_v1}
llm:
  primary: {provider: groq, model: openai/gpt-oss-120b, rpm: 25, daily_budget: 1000, api_key_envs: [GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3]}
  fallback: null          # Gemini removed (persistent 503s)
  temperature: 0
  max_output_tokens: 2048
  key_cooldown_s: 60
  max_retry_wait_s: 65
  backoff_base_s: 2.0
  timeout_s: 30
  max_retries: 2
  cache_dir: .cache/llm
  budget_warn_ratio: 0.8
cache: {enabled: true, threshold: 0.95}
acl: {default: "*", users: {}}
eval: {dev_ratio: 0.5, seed: 42, n_companies: 10, min_questions: 50, recall_k: 5, runs_dir: data/eval/runs, judge_max_output_tokens: 4096}
```

## LLM call policy

Order per request: disk cache -> primary key (Groq) -> on 429/5xx/timeout, honor the provider's "try again in" hint, rotate to the next key (the three keys are separate Groq accounts, so quotas are independent), backoff and retry up to `max_retries`. The cache key includes provider and model. Groq `openai/gpt-oss-120b` free limits (checked 2026-11-10): 30 requests/min, 1,000 requests/day, 8,000 tokens/min, **200,000 tokens/day**, per organization. `LLMClient` counts requests per key per day and stops at `budget_warn_ratio` of `daily_budget`, but **does not count tokens**: on 2026-11-10 two accounts hit the 200,000 tokens/day limit (a 5d dev eval had 11 of 29 questions fail with 429) while only 719 of 3,000 requests were used. A token counter (Groq returns `usage` on every response) is the known missing guard.

## Eval definitions

- **Recall@k (page-level):** 1 if any of the top-k hits' pages is in gold pages for the right doc, averaged. It rewards any gold page, so a two-page question counts as a hit when only one page was retrieved.
- **MRR:** reciprocal rank of the first hit on a gold page.
- **Accuracy:** LLM judge compares prediction to gold answer (numbers within rounding, same units).
- **Faithfulness:** LLM judge checks each claim in the answer is supported by the retrieved contexts, score = supported / total, over answered questions only ("I don't know" is excluded, not scored 1.0).
- **Latency:** p50/p95 per stage and end to end, over uncached questions.
- **Offline retrieval tools (no LLM):** `compare_indexes` (recall, MRR, gold-page rank for any config), `fusion_sweep` (query prefix and fusion variants on cached candidate lists), `rerank_dev`, `pilot_bakeoff`.

## Key design decisions

- Company/fiscal-year metadata filter: FinanceBench questions name them, so filtering removes most distractors; retrieval only ever compares chunks inside one filing.
- Context prefix with heading and units keeps year, company, statement and scale in the embedding and in BM25.
- Tables: whole when they fit the embedder window, otherwise row groups with repeated headers (never cut inside a row). This replaced "tables stay whole", which silently truncated the bottom of every large statement.
- Chunk sizes are set by the embedder's tokenizer, not a regex.
- mxbai-large over bge-small/base, arctic-m, nomic, gte-base: best dense MRR in the offline pilot (0.408 vs 0.294 for bge-small on the same chunks) at about 1.5 h to index; the one-time cost was accepted.
- No reranker: on all 29 dev questions every candidate lowered MRR and pushed table chunks down (MiniLM-L12 in 12 of 12 table questions). The only helpful one in the pilot (bge-reranker-v2-m3-int8) takes about 12 s per question on CPU.
- Own faithfulness judge instead of the RAGAS library: fewer dependencies, fewer rate-limit failures.
- Disk-cached LLM calls: deterministic reruns, free retries.
- Embedded Qdrant and file caches: zero infrastructure, simple Docker.

## Known limitations (for the README)

- Small subset (~10 companies); dev n=29, so one question is 0.034 of any metric and results do not generalize to all filings. The BM25 weight was tuned on dev; the test split (Task 11) has not been run.
- LLM-as-judge is not human evaluation; the answerer and both judges are the same model.
- Multi-page questions (ratios needing two statements) fail when only one page is retrieved, and page-level recall@5 hides it.
- Scanned or image-only pages are skipped (no OCR).
- Groq free tier: 200,000 tokens/day per account, not tracked by `LLMClient`.
- Untested ideas: statement-type boost, ratio-term expansion, multi-page diversification of the context, sentence-aware windowing, top_n 8 (run invalid, see experiment 16).
