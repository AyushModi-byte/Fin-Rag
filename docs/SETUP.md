# FinRag Setup

## One-time

1. Python 3.11 and Git installed. Docker Desktop installed (needed for Task 11 only).
2. Get a free Gemini API key at Google AI Studio.
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
4. Get a free Groq API key from the Groq console (fallback provider).
   Create `.env` in the project root:
   ```
   GEMINI_API_KEY=your_gemini_key
   GROQ_API_KEY=your_groq_key
   ```
5. Create a virtual environment: `python -m venv .venv`, activate it.
6. Open Claude Code in the project root and paste the "Session start" prompt from `docs/PROMPTS.md`.

## Daily workflow

1. `/clear`, then paste the Session start prompt.
2. Paste the next task prompt.
3. When it finishes: `pytest -q` and `git log --oneline -3` yourself. Do not trust "done" without green tests and a commit.
4. If Claude Code fails the same step twice, it should stop and report. If it keeps retrying, interrupt and paste the "When stuck" prompt.

## Free-tier cautions

- Both free tiers have low limits. The LLM client throttles, caches, and switches Gemini -> Groq automatically, so you should never be stuck waiting. Reruns of the same eval are free thanks to the cache.
- Claude Code must not burn the keys: smoke tests use `--limit 5`, the full dev eval runs only after a real change. The rules are in `CLAUDE.md`.
- If both providers are limited, Claude Code works on offline tasks (tests, chunking, indexing, docs) instead of waiting.
- Hugging Face Spaces: add `GEMINI_API_KEY` and `GROQ_API_KEY` as Space secrets, never in the repo.

## Time estimate

Roughly 2-3 focused days: Tasks 1-8 in day 1-2, tuning and Tasks 10-11 in day 2-3.
