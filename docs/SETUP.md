# FinRag Setup

## One-time

1. Python and Git installed (the project venv currently runs Python 3.13.3). Docker Desktop installed (needed for Task 11 only).
2. Gemini is no longer used (persistent 503s). It is not needed.
3. Create the project and put these files in place:
   ```
   FinRag/
     CLAUDE.md
     docs/ARCHITECTURE.md
     docs/PROMPTS.md
     docs/SETUP.md
     docs/experiments.md
     docs/superpowers/plans/2026-11-09-production-rag-sec-filings.md
   ```
4. Get free Groq API keys from the Groq console. The project uses three keys from three separate accounts (each account has its own daily quota: 1,000 requests and 200,000 tokens for `openai/gpt-oss-120b`).
   Create `.env` in the project root (never commit it):
   ```
   GROQ_API_KEY_1=your_first_key
   GROQ_API_KEY_2=your_second_key
   GROQ_API_KEY_3=your_third_key
   ```
5. Create a virtual environment: `python -m venv .venv`, activate it.
6. Open Claude Code in the project root and paste the "Session start" prompt from `docs/PROMPTS.md`.

## Daily workflow

1. `/clear`, then paste the Session start prompt.
2. Paste the next task prompt.
3. When it finishes: `pytest -q` and `git log --oneline -3` yourself. Do not trust "done" without green tests and a commit.
4. If Claude Code fails the same step twice, it should stop and report. If it keeps retrying, interrupt and paste the "When stuck" prompt.

## Free-tier cautions

- The Groq free tier has low limits, and tokens per day (200,000 per account) run out before requests per day. The LLM client throttles, caches, rotates between the three keys, counts requests and tokens per key per day, and stops at 80% of either budget (`BudgetNearlyExhausted`). Reruns of the same eval are free thanks to the cache.
- Claude Code must not burn the keys: smoke tests use `--limit 5`, the full dev eval runs only after a real change. The rules are in `CLAUDE.md`.
- If all keys are limited, Claude Code works on offline tasks (tests, chunking, indexing, docs) instead of waiting.
- Hugging Face Spaces: add `GROQ_API_KEY_1`, `GROQ_API_KEY_2` and `GROQ_API_KEY_3` as Space secrets, never in the repo.

## Time estimate

Roughly 2-3 focused days: Tasks 1-8 in day 1-2, tuning and Tasks 10-11 in day 2-3. Actual: Tasks 1-9 are done (the v2 re-index alone is about 1.6 h of CPU embedding).
