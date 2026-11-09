# FinRag Experiment Log

Rule: one change per row. Metrics are on the **dev** split. Keep only if it improves; otherwise revert.

| # | Date | Change | Recall@5 | MRR | Accuracy | Faithfulness | p50 latency | Decision |
|---|------|--------|----------|-----|----------|--------------|-------------|----------|
| 0 | | Baseline: dense + BM25 + RRF, no rerank, no routing | | | | | | baseline |
| 1 | | + company/year metadata filter | | | | | | |
| 2 | | + cross-encoder rerank (top_k 20 -> top_n 5) | | | | | | |
| 3 | | Child chunk size 256 / 384 / 512 | | | | | | |
| 4 | | Context prefix on/off | | | | | | |
| 5 | | RRF vs weighted fusion | | | | | | |
| 6 | | Parent-page expansion for generation | | | | | | |
| 7 | | Numeric-reasoning prompt | | | | | | |

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
