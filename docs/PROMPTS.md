# FinRag: Claude Code Prompts

Run one prompt per session step. Start a fresh `/clear` between tasks. After each task, check tests pass and `git log` shows a commit before moving on.

## Session start (paste every new session)

```
Read CLAUDE.md, docs/ARCHITECTURE.md, and docs/superpowers/plans/2026-11-09-production-rag-sec-filings.md.
Run `git log --oneline -10` and `pytest -q` to see current state.
Tell me in 3 lines: which task is next, and whether the repo is green. Do not write code yet.
```

## Task 1: Scaffold

```
Do Task 1 from the plan only. Create the folder skeleton exactly as in CLAUDE.md, config.yaml with the defaults in ARCHITECTURE.md, pydantic models, load_config, .gitignore, requirements.txt (pinned after install). Write the tests first. Run pytest, then git init and commit. Stop after Task 1 and report.
```

## Task 2: Data and eval set

```
Do Task 2 only. Download the FinanceBench open-source questions and the PDFs for the ~10 companies with the most 10-K questions (select_docs). Build data/eval/questions.jsonl with a deterministic dev/test split (seed in config) and data/eval/docs.json. Tests first. Report: number of docs, questions per split. If total questions < 50, tell me before continuing. If a download fails twice, stop and ask me.
```

## Task 3: Ingestion

```
Do Task 3 only. Implement parse_pdf and load_corpus with pymupdf4llm. Add a tiny fixture PDF for tests. Skip and log empty pages. Run it on the real PDFs and show me one table page's markdown so I can check the table survived.
```

## Task 4: Chunking

```
Do Task 4 only. Implement chunk_pages with parent page spans, child chunks of config size, never splitting a markdown table, context prefix, and deterministic ids. Tests first as listed in the plan. Report chunk count per document.
```

## Task 5: Index

```
Do Task 5 only. Implement Embedder, VectorStore (embedded Qdrant), BM25Index, build_index and the index_corpus job with an index manifest that refuses a model/version mismatch. Tests on a 5-chunk fixture first. Then index the real corpus and log chunk count and time in docs/experiments.md.
```

## Task 6: Retrieval, rerank, routing

```
Do Task 6 only. Implement rrf, HybridRetriever, Reranker (fastembed cross-encoder) and query_router (rule-based company/fiscal-year extraction, in_scope flag). Tests first. Do not change any config values yet.
```

## Task 7: LLM and generation

```
Do Task 7 only. Implement LLMClient with: disk cache, throttle to each provider's rpm, timeout, max_retries with backoff on 429/5xx, automatic fallback from Gemini to Groq, and a daily call counter per provider that stops and warns me at 80% of daily_budget. Add prompt templates and generate() with [doc_id p.N] citations and exact "I don't know" fallback. Use a fake HTTP layer in tests (including a test that a 429 on Gemini falls through to Groq, and a test that the budget counter blocks at 80%). Before coding, check the current Gemini and Groq free-tier model names and rate limits and tell me if config needs changing. Then do ONE real smoke call per provider and nothing more. Never loop API calls.
```

## Task 8: Eval and baseline

```
Do Task 8 only. Implement recall_at_k, mrr, judge_correct, judge_faithful, run_eval, run_query and the eval_nightly CLI with a --limit N flag. First run with --limit 5 as a smoke test and show me the output. Only then run the naive baseline (hybrid RRF, router filter OFF, rerank OFF) on the dev split and record it as row 0 in docs/experiments.md. Zero errors required. Do not tune anything.
```

## Task 9: Tuning (repeat per experiment)

```
Run experiment #N from docs/experiments.md: [describe the one change]. Change only that setting in config.yaml, run `python -m src.jobs.eval_nightly --split dev`, fill the row in the table, and keep the change only if recall@5 or accuracy improved. If worse, git revert it. Do not touch the test split. Stop after this one experiment.
```

Failure analysis prompt (after experiments 1-4):

```
List the 10 worst dev questions from the latest run. For each: question, gold pages, what we retrieved, and the cause (retrieval miss / table parse / reasoning / judge error). Fill the failure-analysis table in docs/experiments.md. Propose the single highest-impact fix. Do not implement it until I say so.
```

## Task 10: Cache, observability, security, API

```
Do Task 10 only. Implement SemanticCache, tracing, redact_pii, ACL, FastAPI routes and main.py. Tests first as listed in the plan. Then run the full pytest suite and re-run the dev eval: metrics must not drop versus the last kept result. Report both.
```

## Task 11: Docker, final numbers, README

```
Do Task 11 only. Write Dockerfile and docker-compose.yaml, verify `docker compose up` serves /health and one real /query. Then run the final eval on the TEST split exactly once and fill the Final table. Write README.md (architecture diagram, run steps, dev vs test results, ablation table from experiments.md, limitations from ARCHITECTURE.md). Do not tune after the test run.
```

## When stuck (two-strike rule triggered)

```
Stop. Do not retry. Write: (1) the exact error, (2) what you tried, (3) your top two hypotheses, (4) what you need from me. Then wait.
```

## When metrics got worse

```
Metrics dropped versus the last kept result. Run `git diff` for the last change, git revert it, re-run the dev eval to confirm we are back to the previous numbers, and log the failed experiment as "reverted" in docs/experiments.md. Then tell me what you think went wrong, without fixing it yet.
```
