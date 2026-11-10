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
| 9 | 2026-11-10 | Drop chunks under 15 tokens at retrieval (`retrieval.min_chunk_tokens: 0 -> 15`; from the failure analysis; on top of exp 8) | 0.345 | 0.196 | 0.517 | 0.909 (n=19) | 1672 ms | keep |
| 10 | 2026-11-10 | **Index v2** (one re-index): chunker v2 (embedder-tokenizer sizing, table row groups, rich prefix), mxbai-embed-large-v1, BM25 v2 (stemming), `index.version: v2`; router + min_chunk_tokens 15, rerank off | 0.517 | 0.411 | 0.552 | 0.908 (n=21) | 2553 ms | keep |

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

### Experiment 9 notes (drop tiny chunks)

- Change: `retrieval.min_chunk_tokens: 0 -> 15` (the setting was added at 0 in its own no-behavior-change commit; the experiment is the one-value flip). The retriever drops dense and BM25 hits under 15 tokens (page numbers, headers) before fusing; tables are never that short. No re-index, no extra LLM calls. Counted run, zero errors: `data/eval/runs/20261110T070512_dev.json`, n=29.
- Vs experiment 8: recall@5 0.310 -> 0.345 (9/29 -> 10/29), MRR 0.153 -> 0.196, accuracy 0.517 -> 0.517 (15/29 both: gained 00757, lost 00790). Faithfulness 0.904 -> 0.909 (n=19 both). Decision: keep, because recall@5 improved; accuracy did not.
- Honest size of the effect: recall@5 moved by one question (00757, the AMD customer-concentration question that had page-number chunks in its top 3), so on n=29 this is a small gain; the MRR change (+0.043) is the stronger signal that real pages moved up. The accuracy swap (+1/-1) is within run-to-run noise of the answerer; 00790 had a different context set this time.
- Latency 1672 ms p50 / 3036 ms p95 is over only the 8 uncached questions (the rest reused cached answers), so it is indicative only.

### Experiment 10 notes (index v2)

- Change: everything index-time in one re-index (commits `36de644`, `6e9f4af`, `83f51b4`, `4a6e043`): chunker v2, `mixedbread-ai/mxbai-embed-large-v1` (1024-d, no query/doc prefix), child 300 / overlap 40 / cap 480, BM25 v2. Query-time settings unchanged from experiment 9 (router, min_chunk_tokens 15, rerank off, top_k 20, top_n 5). Counted run, zero errors: `data/eval/runs/20261110T133621_dev.json`, n=29. v1 is untouched (`index.version: v1` with `config.v1.yaml` rolls back).
- **Offline retrieval first** (`python -m src.jobs.compare_indexes`, no LLM calls, all 29 dev questions): recall@5 0.345 -> 0.517, MRR 0.196 -> 0.411, mean gold rank 18.8 -> 13.1, geo-mean 8.7 -> 4.5, median 7 -> 3, missing in top 100: 1 -> 0. Gold rank better on 19 questions, worse on 9, same on 1. Recall@5 gained on 8 questions (00464, 00917, 00995, 01107, 03882, 04735, 07507, 10285) and lost on 3 narrowly (00563 4 -> 7, 00566, 01279 5 -> 6). Largest regression: 00720 (18 -> 63, a miss in both; the "gross margin is not a useful metric" question).
- **Dev eval vs experiment 9 (v1):** recall@5 0.345 -> 0.517 (10/29 -> 15/29), MRR 0.196 -> 0.411, accuracy 0.517 -> 0.552 (15/29 -> 16/29: gained 00790, 00995, 03069, 07507; lost 00720, 01226, 01279), faithfulness 0.909 (n=19) -> 0.908 (n=21), "I don't know" 10 -> 8. Decision rule (keep if recall@5 or MRR improves and accuracy is not worse by more than one question): **keep**.
- Honest size of the effect: retrieval moved a lot (recall@5 +5 questions, MRR more than doubled) but accuracy only +1 question and +4/-3 swaps, within answerer noise. Retrieval is no longer the main limit on accuracy for the questions it now finds; the remaining misses are reasoning/answering and the lost-cause questions (00720, 01981).
- Latency p50 2553 ms / p95 11995 ms over 24 uncached questions (p95 from Groq rate-limit waits); retrieval stage p50 about 290 ms with mxbai-large (query embedding on CPU). All LLM calls served by Groq gpt-oss-120b (`groq: 55`, cache 16).
- Which component earned it is NOT isolated: embedder, chunking and BM25 changed together by design (one re-index). The pilot suggests most of the gain is the embedder (dense MRR 0.408 vs 0.294 for bge-small on the same v2 chunks).

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
| 2026-11-10 | v2 | mixedbread-ai/mxbai-embed-large-v1 (1024-d) | 22 | 16101 / 16101 (4787 parents) | embedding 96 min wall (16:56 -> 18:32, 2.2 to 2.8 chunks/s; probe had projected 2.06 h) + about 25 min lost in a slow Qdrant upsert, aborted, + 42 s to finish from the shards | child_tokens 300, overlap 40, cap 480, min_tokens 15, BM25 v2. Gate raised 2.0 -> 2.5 h (approved). Index size 294 MB. **Lesson:** embedded Qdrant commits and fsyncs per point (1.3 points/s at 5k points on this Windows machine); fixed with `VectorStore.bulk_load()` (16k points in 7 s). This probably also explains much of v1's 11,558 s, not only the embedding. |
| 2026-11-09 | v1 | BAAI/bge-small-en-v1.5 (384-d) | 22 | 15925 / 15925 | 11558.3 s (as reported by the job, includes embedding on CPU; parsed pages reused) | child_tokens 256, overlap 32, context prefix on. Index size 104 MB. Time is far above the ~15 min parse; wall-clock may include machine sleep, not re-measured. |

## Embedding speed (Phase 0, bge-small-en-v1.5, 300 random v1 chunks, Ryzen 7 4800H 8C/16T, no LLM calls)

| Variant | chunks/s |
|---|---|
| batch 32 (v1 setting) | 8.4 |
| batch 64 | 7.3 |
| batch 8 | 12.9 |
| batch 32, threads 4 / 8 / 16 | 7.6 / 8.4 / 7.9 |
| batch 32, fastembed parallel 2 / 4 (1 thread each) | 4.9 / 6.9 |
| batch 32, length-sorted | 10.7 |
| batch 32, max_length 256 (truncates; not usable for tables) | 18.0 |
| sorted, batch 16 / 8 / 4 | 12.6 / 15.4 / 19.2 |
| sorted, batch 2 | 28.7 |
| **sorted, batch 1 (applied)** | **30.7** |

- Padding to the longest text in a batch is the cost: smaller batches are faster, threads and data-parallel workers do not help. Applied: `embeddings.batch_size: 1`, `sort_by_length: true` (3.7x faster than v1's setting).
- At 0.12 s/chunk (batch 32) the v1 build should have taken about 32 min, not 11,558 s (0.73 s/chunk); machine sleep or heavy background load during that build is the likely cause (not re-measured). CPU load during the benchmark was ~23% from other processes (Chrome, Docker), balanced power plan, on battery 96%.

## Phase 2 pilot bake-off (offline, no LLM calls; 3 filings, 11 dev questions, 2,061 v2 chunks)

Script: `python -m src.jobs.pilot_bakeoff retrieval|rerank`; raw results `docs/pilot_*.json`. Filings: AMD 2022 (5 q), Boeing 2022 (3), AmEx 2022 (3). The filter is the gold doc (oracle router). Metrics are page level over the full filtered ranking (depth 100; "missing" = rank 101). **n = 11, so one question moves recall@5 by 0.09: treat differences below about 0.1 recall or 3 mean-rank points as noise.** Projected full time = pilot chunks/s scaled to 16,101 chunks.

| Variant (v2 chunks c300 unless noted) | mean rank | median | R@5 | R@20 | MRR | chunks/s | full build |
|---|---|---|---|---|---|---|---|
| **v1 control**: v1 chunks, bge-small, BM25 v1, hybrid | 13.6 | 5 | 0.545 | 0.818 | 0.246 | 30.5 | (v1) |
| v1 control, dense only | 17.3 | 7 | 0.364 | 0.818 | 0.305 | | |
| bge-small, dense (no prefix) | 19.1 | 7 | 0.455 | 0.727 | 0.294 | 28.8 | 0.16 h |
| bge-small, hybrid + BM25 v1 | 18.3 | 9 | 0.455 | 0.818 | 0.293 | | |
| bge-small, hybrid + BM25 v2 | 22.2 | 6 | 0.455 | 0.818 | 0.291 | | |
| bge-base, dense (no prefix) | 17.9 | 7 | 0.455 | 0.818 | 0.298 | 10.1 | 0.44 h |
| bge-base, hybrid + BM25 v2 | 21.2 | 5 | 0.545 | 0.727 | 0.315 | | |
| bge-base, dense, query instruction ON | 14.5 | 6 | 0.455 | 0.727 | 0.253 | | |
| arctic-m, dense, query instruction OFF | 25.8 | 17 | 0.091 | 0.636 | 0.080 | 10.7 | 0.42 h |
| arctic-m, dense, query instruction ON | 15.7 | 6 | 0.455 | 0.818 | 0.282 | | |
| arctic-m, hybrid + BM25 v2 (ON) | 23.3 | 8 | 0.455 | 0.727 | 0.299 | | |
| nomic-v1.5, dense (prefixes ON) | 22.5 | 15 | 0.273 | 0.727 | 0.152 | 7.4 | 0.60 h |
| gte-base, dense | 14.2 | 6 | 0.455 | 0.727 | 0.279 | 10.7 | 0.42 h |
| gte-base, hybrid + BM25 v2 | 20.0 | 7 | 0.455 | 0.818 | 0.303 | | |
| **mxbai-large, dense (no prefix)** | **12.3** | **5** | **0.636** | 0.727 | **0.408** | 3.1 | 1.46 h |
| mxbai-large, hybrid + BM25 v2 (no prefix) | 18.1 | 6 | 0.455 | 0.818 | 0.386 | | |
| mxbai-large, dense, query instruction ON | 16.3 | 5 | 0.636 | 0.727 | 0.345 | | |

- BM25 alone, same 11 questions: v1 tokenizer mean 32.6 / median 27 / R@5 0.182 / MRR 0.103; **v2 (stemmed, numbers kept) 34.5 / 11 / 0.364 / 0.182** (identical for every model's chunking up to tokenizer differences). v2 is clearly better at the median, but the hybrid result is dominated by the dense side.
- Query instruction: helps arctic-m a lot (required), hurts or is neutral for bge and mxbai (dense MRR 0.408 -> 0.345 for mxbai). Offline A/B only; to be re-checked in Phase 5a.
- Hybrid is not better than dense alone for mxbai-large (mean rank 18.1 vs 12.3): fusion adds BM25's weak ranks.

### v1 vs v2 chunking, same model (bge-small), same 11 questions
Per question, hybrid rank v1 -> v2 (BM25 v1 on both): 00464 13->17, 00563 4->10, 00585 28->18, **00720 19->77**, 00757 1->1, 00917 4->5, 00995 4->3, 01091 2->3, **01279 5->1**, 01351 13->9, 01981 57->57. Mean rank is dominated by 00720 (AmEx p.96: a whole 1,900-character income statement used to be one chunk, v2 splits it into three row groups and the dense match drops 4 -> 36) . MRR is better in v2 (0.293 vs 0.246), recall@5 one question worse (0.455 vs 0.545), mean rank worse (18.3 vs 13.6), median 9 vs 5. This is mixed, not a clear win.

### Reranker bake-off (top 20 -> reorder, candidates = mxbai-large hybrid + BM25 v2; gold page is in the top-20 pool for 9 of 11)
| Reranker | mean rank | median | R@5 | MRR | latency / 20 docs | improved / worsened |
|---|---|---|---|---|---|---|
| none | 18.1 | 6 | 0.455 | 0.386 | | |
| MiniLM-L6 (current, control) | 18.2 | 5 | 0.545 | 0.315 | 1.0 s | 3 / 3 |
| MiniLM-L12 | 18.0 | 4 | 0.636 | 0.331 | 1.9 s | 4 / 3 |
| bge-reranker-base | 17.9 | 6 | 0.364 | 0.427 | 4.8 s | 3 / 3 |
| bge-reranker-v2-m3-int8 | 16.6 | 3 | 0.727 | 0.385 | 10.4 s | 5 / 1 |
| jina-turbo | 18.8 | 6 | 0.455 | 0.225 | 1.2 s | 3 / 4 |
| jina-tiny | 18.7 | 6 | 0.364 | 0.316 | 0.7 s | 3 / 3 |

Only bge-reranker-v2-m3-int8 looks useful (5 up / 1 down, R@5 0.727) but at 10 s per 20 docs it is far over the 2.5 s CPU target. MiniLM-L12 is within budget (1.9 s) and also shows R@5 up, MRR down. All differences are within noise at n=11.

### Phase 2b: table-splitting fix, offline, bge-small (11 questions, no LLM calls)

Question 00720 (AmEx p.96) was the biggest v1 -> v2 loss, so I first checked its cause by swapping embeddings of only that page's chunks (cached vectors for the rest). Dense gold rank: v2 baseline 36; same row groups with the old short prefix 47; no prefix 101; prefix with company/year/page only 22; **the page as ONE whole table chunk (v1 style, truncated at 512) with the new prefix 23** (v1's own dense rank was 4). So neither the row-group split nor the longer prefix explains the drop: the v1 -> v2 difference comes from *other* chunks that now outrank this page. 00720 is a "gross margin is not a useful metric" question with no natural evidence chunk.

Then one change at a time on top of the v2 baseline (new knob `chunking.table_group_tokens`, target size of row groups for tables over the cap; default = cap = no behaviour change). Hybrid + BM25 v2, query instruction off:

| Variant | mean rank | geo-mean rank | median | R@5 | R@20 | MRR | dense-only median / R@5 / MRR |
|---|---|---|---|---|---|---|---|
| v1 control (v1 chunks, BM25 v1) | 13.6 | 7.3 | 5 | 0.545 | 0.818 | 0.246 | 7 / 0.364 / 0.305 |
| v2 baseline (cap 480, groups fill the cap) | 22.2 | 7.6 | 6 | 0.455 | 0.818 | 0.291 | 7 / 0.455 / 0.294 |
| row groups of about 250 tokens | 28.7 | 7.2 | 6 | 0.455 | 0.818 | 0.306 | 3 / 0.545 / 0.360 |
| row groups of about 350 tokens | 25.5 | 7.7 | 5 | 0.545 | 0.727 | 0.300 | 5 / 0.545 / 0.295 |
| cap 480 -> 512 | 22.2 | 7.6 | 6 | 0.455 | 0.818 | 0.292 | 6 / 0.455 / 0.284 |

- The arithmetic mean rank is dominated by two lost-cause questions (00720 and 01981, ranks 72 to 138 in every variant, 57 in v1). The outlier-robust **geometric-mean rank is 7.2 to 7.7 for every variant, v1 control 7.3**: v2 chunking is not worse than v1 on this pilot, and not better either.
- None of the three changes is a clear fix. Row groups of 250 help dense retrieval (median 7 -> 3, R@5 +1 question, MRR 0.294 -> 0.360) and rescue 00585 (18 -> 7), but push 00720 further down; hybrid R@5 does not move. Raising the cap to 512 changes almost nothing (tables are 640 to 870 tokens). All differences are one question.
- Decision: no table-splitting change adopted. Defaults stay (`table_group_tokens` unset). Raw results `docs/pilot_tablefix_*.json`.

### Phase 2c: child size with the winner (mxbai-large, no query prefix, same 11 questions)

| child_tokens | chunks (3 filings) | dense MRR / geo-mean rank / R@5 | hybrid+BM25 v2 geo-mean rank / R@5 | full-build time projected from the pilot |
|---|---|---|---|---|
| 256 | 2,419 | 0.473 / 4.1 / 0.545 | 5.1 / 0.636 | about 2.0 h |
| **300** | 2,061 | 0.408 / 5.0 / 0.636 | 5.9 / 0.455 | **about 1.4 h** |
| 384 | 1,658 | 0.420 / 4.7 / 0.727 | 4.9 / 0.545 | about 2.0 h |

Differences are within noise at n=11 (one question = 0.09 recall), so the rule "within noise: pick the faster" gives **child_tokens 300**. (The `projected_full_hours` column in `docs/pilot_*.json` assumes 16,101 chunks for every size; the times above are recomputed from each run's own chunk rate and the corpus ratio 7.8x.) Chosen v2 config: mxbai-large (1024-d), no query/doc prefix, child 300, overlap 40, cap 480, BM25 v2, rerank off. Rollback: `config.v1.yaml` (index v1, bge-small, BM25 v1); run any job with `--config config.v1.yaml`.
