# FinRag

Retrieval-augmented question answering over SEC 10-K filings (a ~10-company subset of
[FinanceBench](https://github.com/patronus-ai/financebench)). CPU only, free tiers only: local embeddings
(`mxbai-embed-large-v1`), embedded Qdrant + BM25, and Groq `openai/gpt-oss-120b` for generation and judging.

Every answer carries `[doc_id p.N]` citations or exactly `I don't know`. The retrieval, generation and eval
code is tested (232 tests) and every tuning step is logged in [`docs/experiments.md`](docs/experiments.md).

## Results (numbers exactly as measured; n = 29 questions per split)

| Split | Recall@5 | MRR | Accuracy | Faithfulness | "I don't know" | p50 | p95 |
|---|---|---|---|---|---|---|---|
| dev (tuned on) | 0.586 (17/29) | 0.452 | 0.586 (17/29) | 0.930 (n=22) | 7 | 2344 ms | n/a |
| **test (run once, not tuned on)** | **0.552 (16/29)** | **0.324** | **0.414 (12/29)** | **0.842 (n=19)** | 10 | 2620 ms | 10081 ms |

The test row is the honest estimate. Every metric is lower than on dev, and **the plan's targets were missed**
(recall@5 >= 0.80, accuracy 60-75%, faithfulness >= 0.90); only p50 < 4 s was met. Cache hits take about 1.5 ms inside the
pipeline (0.2 s end to end through HTTP). Test run: `data/eval/runs/20261111T081204_test.json`, 65 Groq calls, no errors.

- Recall@5 is page-level (any gold page in the top 5 counts), MRR is the rank of the first gold page.
- Accuracy and faithfulness are LLM-judged (see Limitations). Faithfulness is scored over answered questions only.

### Ablation (dev split, one run each; changes after row 1 are stacked on the kept ones)

| # | Change | Recall@5 | MRR | Accuracy | Faithfulness | Decision |
|---|---|---|---|---|---|---|
| 0 | Baseline: dense + BM25 + RRF, no filter, no rerank | 0.172 | 0.108 | 0.207 | 0.947 (n=10) | baseline |
| 1 | + company / fiscal-year metadata filter (rule-based router) | 0.207 | 0.124 | 0.448 | 0.900 (n=17) | keep |
| 2 | + cross-encoder rerank | 0.172 | 0.099 | 0.310 | 0.905 (n=13) | revert |
| 8 | + router reads "FY22" and takes the latest year when several are named | 0.310 | 0.153 | 0.517 | 0.904 (n=19) | keep |
| 9 | + drop chunks under 15 tokens at retrieval | 0.345 | 0.196 | 0.517 | 0.909 (n=19) | keep |
| 10 | **Index v2**: embedder-tokenizer chunk sizing, table row groups with repeated headers, rich prefix, mxbai-large, stemmed BM25 | 0.517 | 0.411 | 0.552 | 0.908 (n=21) | keep |
| 11 | Query instruction prefix (offline only) | 0.517 | 0.379 | n/a | n/a | not adopted |
| 12 | Weighted fusion, BM25 weight 1.0 -> 0.25 | **0.586** | **0.452** | **0.586** | **0.930 (n=22)** | **keep (final)** |
| 13 | Parent expansion (whole table for generation) | 0.586 | 0.452 | 0.586 | 0.930 | not adopted (no change) |
| 14 | Numeric-reasoning prompt `grounded_v2` | 0.586 | 0.452 | 0.552 | 0.846 | revert |
| 15 | 4 cross-encoder rerankers, offline | best 0.621 | all < 0.452 | n/a | n/a | not adopted |
| 16 | `top_n` 5 -> 8 | n/a | n/a | n/a | n/a | not completed (Groq daily token limit; invalid run) |

Experiments 3 to 7 are pilots or duplicates of the rows above; the full log, with run files and notes, is in
`docs/experiments.md`. The BM25 weight (row 12) was chosen on dev from three values, so the dev numbers are slightly optimistic.

## Architecture

```
OFFLINE  (python -m src.jobs.index_corpus)
data/raw/*.pdf -> ingestion (PyMuPDF, tables kept) -> chunking v2 (page parents, table parents, child chunks,
                  context prefix; sized with the embedder's own tokenizer)
   -> embeddings (mxbai-embed-large-v1, CPU, resumable shards) -> embedded Qdrant + bm25s (stemmed)
   -> .cache/index/v2  (+ manifest: model, chunking and tokenizer params; a mismatch refuses to load)

ONLINE   (src/pipelines/query_pipeline.py, served by src/api/routes.py)
question -> redact PII -> ACL -> semantic cache (hit? return)
   -> router (company + fiscal year filter, in scope?)
   -> dense top 20 + BM25 top 20 -> weighted RRF (1.0 : 0.25) -> top 5 chunks      [cross-encoder rerank: built, OFF]
   -> grounded answer via LLMClient (disk cache, key rotation, retries, per-key request and token budgets)
   -> QueryResult(answer, citations, contexts, latency_ms)
```

Details, schemas and every config default: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). All settings live in `config.yaml`.

## Run it

Requirements: Python 3.13 (the tests ran on 3.13.3), ~1 GB for the embedding model, three Groq API keys
(free accounts) in `.env`:

```
GROQ_API_KEY_1=...
GROQ_API_KEY_2=...
GROQ_API_KEY_3=...
```

### Locally

```bash
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pytest                                                 # 232 tests, no network or API keys needed

python main.py                                         # API on http://127.0.0.1:8000 (api.host / api.port)
curl localhost:8000/health
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "What was Adobe total revenue in fiscal year 2022?"}'
```

The API needs the prebuilt index in `.cache/index/v2`. It is not in git. Building it is offline work
(`python -m src.jobs.prepare_data`, then `python -m src.jobs.index_corpus`; about 1.6 h of CPU embedding, resumable).
Index v1 is kept as the rollback (`config.v1.yaml`) and is protected from rebuilds.

### Docker

```bash
docker compose up --build            # serves http://localhost:8000, needs .env and .cache/index/v2 at build time
```

The image (python 3.13-slim, about 3.3 GB) bakes in the v2 index and the embedding model, so it starts offline;
the keys come from `.env` at run time and are never in the image. `./.cache/llm` is mounted so the container and the
host share one LLM response cache and one set of per-key daily budget counters. The semantic cache inside the
container is not persisted. Verified: `/health` reports healthy, and `/query` answered a cached question and a new one.

### API

| Route | Behaviour |
|---|---|
| `GET /health` | `{"status": "ok"}`, no models loaded |
| `POST /query` `{question, user?}` | `QueryResult`; 422 for a blank or too long question; 503 when the LLM budget is nearly used or the provider is down |

`acl.users` in `config.yaml` restricts a user to some companies (`{alice: [Adobe]}`); a restricted user who names no
permitted company gets `I don't know` with no retrieval and no LLM call. E-mails and phone numbers in questions are redacted before
they reach the logs, the cache, the retriever or the LLM. Every stage writes a JSON line with its latency to `logs/app.log`.

### Evaluate

```bash
python -m src.jobs.eval_nightly --split dev --limit 5     # smoke test
python -m src.jobs.eval_nightly --split dev               # full dev run (about 70-90 requests if nothing is cached)
```

The test split needs `--final` and was run once. Do not use it for tuning.

## Limitations

- **Small subset.** About 10 companies, 29 dev and 29 test questions, so one question is 0.034 of any metric and the
  results do not generalise to all filings. Dev to test dropped by 5 accuracy questions; with this n a difference of
  1 to 2 questions is noise, and part of the gap can come from choosing the BM25 weight on dev.
- **LLM-as-judge.** Accuracy and faithfulness are judged by the same model that answers (`gpt-oss-120b`), not by humans.
- **Multi-page questions.** Ratios that need two statements fail when only one page is retrieved, and page-level recall@5 hides it.
- **Citations.** The model does not always use the requested `[doc_id p.N]` form (about 40% of its citations used full-width `【…】` brackets). The parser now accepts those, other bracket and space variants, page ranges and several documents in one bracket, and keeps only pages that were retrieved; the old parser missed about half of the cited answers. The run file of the test split was made with the old parser, so its `citations` are incomplete (metrics are not affected).
- **No OCR.** Scanned or image-only pages are skipped.
- **Free tier limits.** Groq allows 200,000 tokens/day per account; `LLMClient` counts them and stops at 80%. One full
  uncached eval of 29 questions used about 104,000 tokens over the three keys.
- **No reranker.** It lowered MRR on every candidate tried (a rerank that helped took ~12 s per question on CPU).
- **Not deployed.** The Hugging Face Spaces deploy from the original plan was not done.
- Untested ideas: statement-type boost, ratio-term expansion, multi-page diversification of the context, `top_n` 8 (offline dev recall@8 is 0.759 vs 0.586 at 5, but the answer-level effect was never measured).

## Layout

`src/{ingestion,chunking,embeddings,vector_db,lexical,retrieval,rerank,query,cache,prompts,llm,generation,eval,observability,security,api,pipelines,jobs}`,
`tests/`, `config.yaml`, `main.py`, `Dockerfile`, `docker-compose.yaml`, `docs/` (architecture, experiments, plan).
