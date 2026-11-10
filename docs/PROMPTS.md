# FinRag: Claude Code Prompts

## Where we are (updated 2026-11-10)

Tasks 1-9 are DONE (the Task 9 tuning loop is finished: experiments 0-16 in `docs/experiments.md`, index v2 built, token counting added to `LLMClient`). **Resume at Task 10.** Task 11 follows. Nothing for Tasks 10 and 11 exists yet (`src/cache`, `src/observability`, `src/security`, `src/api` hold only `__init__.py`; no `main.py`, `Dockerfile`, `docker-compose.yaml` or `README.md`).

Last kept dev result, which Task 10's "metrics must not drop" check compares against (experiment 12, run `20261110T140849_dev.json`): recall@5 0.586, MRR 0.452, accuracy 0.586, faithfulness 0.930. The test split has NOT been run; it is reserved for Task 11.

Things the Task 10 and 11 prompts below do not say, but you should know:
- Task 10 says `pytest -q`: check the REAL exit status (no piping), 201 tests pass at this point.
- The index is `.cache/index/v2` (294 MB, not committed, built with mxbai-large). For Task 11 the Docker image must mount it or rebuild it (about 1.6 h of CPU embedding), and the image needs the embedding model files (0.64 GB) and the Groq keys as `GROQ_API_KEY_1..3` (not `GEMINI_API_KEY`).
- Task 11's plan text still says "GEMINI_API_KEY as a secret" and "rerank" in the target metrics; both are out of date (Groq only; rerank is off).
- Before the Task 11 test run, check `LLMClient` token budgets: a full run regenerates about 29+ answers and judge calls on a split of unknown size.
- Pick the Docker base image to match the venv's Python (3.13.3), since CLAUDE.md's old 3.11 was never what the tests ran on.
- Open ideas that were not tried (see `docs/experiments.md` failure analysis): multi-page retrieval for two-page ratio questions, statement-type boost, ratio-term expansion, top_n 8 (run was invalid).

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
