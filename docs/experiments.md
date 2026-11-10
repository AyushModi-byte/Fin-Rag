# FinRag Experiment Log

Rule: one change per row. Metrics are on the **dev** split. Keep only if it improves; otherwise revert.

| # | Date | Change | Recall@5 | MRR | Accuracy | Faithfulness | p50 latency | Decision |
|---|------|--------|----------|-----|----------|--------------|-------------|----------|
| 0 | 2026-11-10 | Baseline: dense + BM25 + RRF, no rerank, no routing (top_k 20 -> top 5 to the LLM; Groq gpt-oss-120b only) | 0.172 | 0.108 | 0.207 | 0.947 (n=10) | 2441 ms | baseline |
| 1 | | + company/year metadata filter | | | | | | |
| 2 | | + cross-encoder rerank (top_k 20 -> top_n 5) | | | | | | |
| 3 | | Child chunk size 256 / 384 / 512 | | | | | | |
| 4 | | Context prefix on/off | | | | | | |
| 5 | | RRF vs weighted fusion | | | | | | |
| 6 | | Parent-page expansion for generation | | | | | | |
| 7 | | Numeric-reasoning prompt | | | | | | |

### Baseline notes (row 0)

- Run: `data/eval/runs/20261110T061837_dev.json`; dev split, n=29, **0 errors**. Router OFF, rerank OFF.
- Counts behind the numbers: recall@5 = 5/29 questions with a gold page in the top 5; accuracy = 6/29 judged correct; 19/29 answers were "I don't know" (counted as incorrect).
- Faithfulness is averaged over the 10 answered questions only (n=10); "I don't know" answers make no claims and are excluded, not scored 1.0.
- Latency p50 (end to end, retrieval + generation, excluding the two judge calls) is over the 25 questions whose answer was not served from the disk cache; 4 came from the cache after the `--limit 5` smoke run.
- Model: every call was served by Groq `openai/gpt-oss-120b` (Gemini was removed after persistent 503s), so the numbers describe one model. Answer and both judges use the same model.
- During the run two keys hit 429 and the client rotated to the next key (GROQ_API_KEY_1 -> 2 -> 3); no question failed.
- Not tuned: first measurement of the naive pipeline.

## Final (test split, run once)

| Recall@5 | MRR | Accuracy | Faithfulness | p50 | p95 |
|----------|-----|----------|--------------|-----|-----|
| | | | | | |

## Failure analysis (10 worst dev questions)

| Question id | Cause (retrieval miss / table parse / reasoning / judge) | Fix |
|-------------|-----------------------------------------------------------|-----|

## Index builds

| Date | Index version | Embedding model | Docs | Chunks (dense / BM25) | Build time | Notes |
|------|---------------|-----------------|------|------------------------|------------|-------|
| 2026-11-09 | v1 | BAAI/bge-small-en-v1.5 (384-d) | 22 | 15925 / 15925 | 11558.3 s (as reported by the job, includes embedding on CPU; parsed pages reused) | child_tokens 256, overlap 32, context prefix on. Index size 104 MB. Time is far above the ~15 min parse; wall-clock may include machine sleep, not re-measured. |
