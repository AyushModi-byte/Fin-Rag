# FinRag Experiment Log

Rule: one change per row. Metrics are on the **dev** split. Keep only if it improves; otherwise revert.

| # | Date | Change | Recall@5 | MRR | Accuracy | Faithfulness | p50 latency | Decision |
|---|------|--------|----------|-----|----------|--------------|-------------|----------|
| 0 | 2026-11-10 | Baseline: dense + BM25 + RRF, no rerank, no routing (top_k 20 -> top 5 to the LLM; Groq gpt-oss-120b only) | 0.172 | 0.108 | 0.207 | 0.947 (n=10) | 2441 ms | baseline |
| 1 | 2026-11-10 | + company/year metadata filter (`retrieval.use_router_filter: true`) | 0.207 | 0.124 | 0.448 | 0.900 (n=17) | 3012 ms | keep |
| 2 | 2026-11-10 | + cross-encoder rerank (top_k 20 -> top_n 5; `rerank.enabled: true`, on top of exp 1) | 0.172 | 0.099 | 0.310 | 0.905 (n=13) | 2455 ms | revert |
| 3 | | Child chunk size 256 / 384 / 512 | | | | | | |
| 4 | | Context prefix on/off | | | | | | |
| 5 | | RRF vs weighted fusion | | | | | | |
| 6 | | Parent-page expansion for generation | | | | | | |
| 7 | | Numeric-reasoning prompt | | | | | | |
| 8 | 2026-11-10 | Router year parsing: FY22 two-digit years + latest year when several are named (from the failure analysis; on top of exp 1) | 0.310 | 0.153 | 0.517 | 0.904 (n=19) | 2002 ms | keep |

### Baseline notes (row 0)

- Run: `data/eval/runs/20261110T061837_dev.json`; dev split, n=29, **0 errors**. Router OFF, rerank OFF.
- Counts behind the numbers: recall@5 = 5/29 questions with a gold page in the top 5; accuracy = 6/29 judged correct; 19/29 answers were "I don't know" (counted as incorrect).
- Faithfulness is averaged over the 10 answered questions only (n=10); "I don't know" answers make no claims and are excluded, not scored 1.0.
- Latency p50 (end to end, retrieval + generation, excluding the two judge calls) is over the 25 questions whose answer was not served from the disk cache; 4 came from the cache after the `--limit 5` smoke run.
- Model: every call was served by Groq `openai/gpt-oss-120b` (Gemini was removed after persistent 503s), so the numbers describe one model. Answer and both judges use the same model.
- During the run two keys hit 429 and the client rotated to the next key (GROQ_API_KEY_1 -> 2 -> 3); no question failed.
- Not tuned: first measurement of the naive pipeline.

### Experiment 1 notes (router filter ON)

- Change: `retrieval.use_router_filter: false -> true`, nothing else (commit `59e7633`). Vs baseline: recall@5 0.172 -> 0.207 (+1 question: 5/29 -> 6/29), MRR 0.108 -> 0.124, accuracy 0.207 -> 0.448 (6/29 -> 13/29), "I don't know" answers 19 -> 12. Decision: keep (recall@5 and accuracy both improved).
- Counted run (zero errors): `data/eval/runs/20261110T062942_dev.json`, n=29. Its answers and judge replies were served from the disk cache, so it has no latency; the latency above (p50 3012 ms, p95 4707 ms, 26 uncached questions) is from the first run of this experiment, `20261110T062546_dev.json`.
- That first run had 1 error (accuracy 0.414, MRR 0.119) caused by infrastructure, not retrieval, so it is not the counted result: (1) a Groq tokens-per-minute 429 ("try again in 15.7s") outlasted the 2 s / 4 s backoff. The 429 limit is per Groq organization, so rotating the 3 keys cannot help; fixed by honoring the provider's wait hint (commit `d2091bd`). (2) The faithfulness judge's JSON was cut off at `max_output_tokens` (reasoning tokens share that budget) and the broken reply had been cached; fixed by treating truncated output as an error that is never cached, with a 4096-token budget for the faithfulness judge only (commit `5ebca16`, no extra LLM calls). A scan of the cache found no truncated answers in either run.
- Recall@5 improved only slightly while accuracy more than doubled: most correct answers now come from filtered results whose gold page is not in the top 5 (the filter removes distractor filings; the right page is often still ranked lower). Page-level recall understates answerability; see the failure analysis after experiments 1-4.
- The router found the year for only about half of the dev questions (15/29), so for the rest it filters by company only.

### Experiment 2 notes (rerank ON, on top of experiment 1)

- Change: `rerank.enabled: false -> true` (commit `a51c932`, reverted). Counted run, zero errors: `data/eval/runs/20261110T064102_dev.json`, n=29, 29 uncached questions.
- Vs experiment 1 (the config in force before this change): recall@5 0.207 -> 0.172 (6/29 -> 5/29), MRR 0.124 -> 0.099, accuracy 0.448 -> 0.310 (13/29 -> 9/29), "I don't know" answers 12 -> 16. Faithfulness 0.900 -> 0.905 is over different answered sets (n=17 vs n=13), so it is not comparable. Per question: accuracy gained 1 and lost 5; recall gained 1 and lost 2. Decision: revert (both recall@5 and accuracy got worse).
- Cost: rerank stage p50 about 1.2 s on CPU; end-to-end p50 2455 ms. The p95 of 15467 ms came from Groq rate-limit waits in the LLM stage on 2 questions, not from reranking.
- Hypothesis (not tested): the general-purpose passage cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`) scores table-heavy financial chunks poorly (it truncates long inputs and favors prose that mirrors the question's words), so it pushes numeric tables down. If this is revisited, try a different reranker model as a separate experiment; do not stack it on other changes.

### Experiment 8 notes (router year parsing)

- Change: `query_router._detect_year` now reads "FY22" as 2022 and, when several years are named, filters to the latest (commit `exp 8`; 138 tests pass). No config change, no re-index, no extra LLM calls. Counted run, zero errors: `data/eval/runs/20261110T070256_dev.json`, n=29.
- Vs experiment 1: recall@5 0.207 -> 0.310 (6/29 -> 9/29), MRR 0.124 -> 0.153, accuracy 0.448 -> 0.517 (13/29 -> 15/29), "I don't know" 12 -> 10. Per question: accuracy gained 2 and lost 0 (ids 00917, 01279, both AMD FY22), recall gained 3 and lost 0. Decision: keep.
- Latency 2002 ms p50 / 4555 ms p95 is over only the 10 questions whose answer was not already in the disk cache (the other 19 had unchanged context, so their answers were reused); treat as indicative.
- The offline prototype predicted the year filter would be right for 27 of 29 dev questions (was 15); the retrieval gain is smaller than that headline because many gold pages still rank below the top 5 once the right filing is selected.

## Final (test split, run once)

| Recall@5 | MRR | Accuracy | Faithfulness | p50 | p95 |
|----------|-----|----------|--------------|-----|-----|
| | | | | | |

## Failure analysis (10 worst dev questions)

| Question id | Cause (retrieval miss / table parse / reasoning / judge) | Fix |
|-------------|-----------------------------------------------------------|-----|
| financebench_id_00438 | Retrieval miss. Adobe FY2022 operating margin; gold p.54, our top 5 were p.38/88/89/64/45; gold ranks 26th in the fused top 20 (dense 15th, BM25 40th). The gold page is an income-statement table with a 666-token chunk (embedder truncates at 512), and the margin has to be computed. Answered "I don't know". | Split big tables into row groups with the header repeated (needs re-index). |
| financebench_id_00585 | Retrieval miss + table parse + router. Boeing ETR FY2022 vs FY2021; gold p.55 (ranks 22nd). Two years named, so no year filter and 2018-filing chunks (p.78, columns 2018/2017/2016) crowd the top 5. The p.77 table we used has garbled header cells (202, 2, 202, 1), and the answer gave (0.6)% and 14.7% (sign lost) vs gold 0.62% and -14.76%. | Router: several years named, filter to the latest. Later: repair split header cells. |
| financebench_id_00757 | Router + junk chunks. AMD customer concentration FY22; gold p.12 (not in fused top 20, dense 35th, BM25 48th). "FY22" is not parsed, so the filter is company-only and AMD_2015 chunks (p.49 about 2015/2014/2013) crowd the results; page-number chunks "21" and "10" are in the top 3. | Router: parse two-digit FY22. Drop tiny chunks (under 15 tokens). |
| financebench_id_00799 | Retrieval miss + reasoning. Amcor quick ratio FY2023 vs FY2022; gold p.52 balance sheet (812-token chunk) is not in the dense or BM25 top 200 even with the company filter. The ratio is never named on the page, it must be derived. Two years named, so no year filter. | Router latest-year rule; table row-group splitting. Query expansion is a later idea. |
| financebench_id_00995 | Router + junk chunks. AMD products FY22; gold p.4 ranks 26th (dense 30th, BM25 13th). "FY22" not parsed so filings are mixed; page-number chunk "22" is in the top 3. Our answer is detailed and plausible but judged incorrect against the gold list (possible judge strictness, not verified). | Router: parse two-digit FY22. Drop tiny chunks. |
| financebench_id_01009 | Retrieval miss (not a router case). PepsiCo geographies FY2022; gold p.4-5 ranks 32nd even with the correct company and year filter (dense 19th, BM25 16th). Financial-note pages outrank the business-overview pages; our answer cited 200 countries and R&D sites, not the geography list. | No router fix. Candidates: section-aware chunks or larger candidate pool; test separately. |
| financebench_id_01981 | Retrieval miss (not a router case). AmEx card member retention 2022; gold p.45 (answer "Yes") is not in the dense top 200 and ranks 42nd in BM25. Vague qualitative question; page-number chunks "1" and "ii" are in the top 5. | Drop tiny chunks. Later: query rewriting. |
| financebench_id_04672 | Retrieval miss / table size. 3M FY2018 net PP&E; gold p.58 balance sheet is one 868-token chunk, so only the first 512 tokens are embedded (dense 49th, BM25 123rd). Router was correct (3M, 2018). | Split big tables into row groups with the header repeated (needs re-index). |
| financebench_id_01279 | Router + big table. AMD cash flow by activity FY22; gold p.58 ranks 19th (807-token table chunk). "FY22" not parsed, so AMD_2015 chunks (p.48, text about 2014/2015) take the top 3 slots. | Router: parse two-digit FY22; table row-group splitting. |
| financebench_id_07507 | Router + retrieval miss + reasoning. Adobe operating income change FY2015 to FY2016; gold p.62 income statement (665-token chunk) ranks 15th (dense 8th, BM25 138th). Two years named, so filings are mixed; the 65.4% has to be computed. | Router latest-year rule. |

### Failure analysis summary (experiment 1 config, run `20261110T062942_dev.json`, offline, no LLM calls)

- 16 of 29 dev answers are incorrect; 12 of those are "I don't know" and 4 are wrong answers. The 10 rows above are the incorrect answers with the lowest MRR. No judge errors were found among them.
- **Router year gaps affect 6 of the 10 rows** (three "FY22" questions, three that name two years). Measured over all 29 dev questions against the gold filing's year: current router 15 correct and 14 no-year; adding two-digit years gives 19 correct; adding "several years named: use the latest" gives 27 correct, 2 no-year (the question names no year at all), 0 wrong.
- **Tiny chunks** (under 15 tokens, mostly page numbers): 595 of 15925 chunks (3.7%). They fill 12 of 145 top-5 slots (8.3%) across 7 of 29 dev questions, because the "company year page" prefix makes them match queries about that company and year.
- **Chunks over 512 tokens** (truncated by the embedder): 421 of 15925 (2.6%), but the gold page of 17 of 29 dev questions holds one. Those questions were answered correctly 6 of 17 times versus 7 of 12 for the rest. Confounded with question type (statement tables), so only suggestive.
- Not fixable by routing: 01009, 01981, 04672, 00438 (right company and year already, gold page still ranks 15th to 42nd).

**Proposed single highest-impact fix (not implemented):** router year extraction: (1) parse two-digit fiscal years such as "FY22"; (2) when several years are named, filter to the latest one (the latest 10-K contains the prior-year comparatives). It is a rule-only change: no re-index, no LLM calls, one dev eval, and it makes the year filter right for 27 of 29 dev questions instead of 15 (offline prototype, 0 wrong). It touches 12 of 29 dev questions, including 6 of the 10 worst. Caveat: this raises the filter's correctness; it does not guarantee the gold page then reaches the top 5, which only a dev eval can show. Runner-ups: drop tiny chunks at retrieval time (no re-index), then split big tables into row groups (re-index, about 3 h).


## Index builds

| Date | Index version | Embedding model | Docs | Chunks (dense / BM25) | Build time | Notes |
|------|---------------|-----------------|------|------------------------|------------|-------|
| 2026-11-09 | v1 | BAAI/bge-small-en-v1.5 (384-d) | 22 | 15925 / 15925 | 11558.3 s (as reported by the job, includes embedding on CPU; parsed pages reused) | child_tokens 256, overlap 32, context prefix on. Index size 104 MB. Time is far above the ~15 min parse; wall-clock may include machine sleep, not re-measured. |
