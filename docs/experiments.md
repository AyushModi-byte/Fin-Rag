# FinRag Experiment Log

Rule: one change per row. Metrics are on the **dev** split. Keep only if it improves; otherwise revert.

| # | Date | Change | Recall@5 | MRR | Accuracy | Faithfulness | p50 latency | Decision |
|---|------|--------|----------|-----|----------|--------------|-------------|----------|
| 0 | 2026-11-10 | Baseline: dense + BM25 + RRF, no rerank, no routing (top_k 20 -> top 5 to the LLM; Groq gpt-oss-120b only) | 0.172 | 0.108 | 0.207 | 0.947 (n=10) | 2441 ms | baseline |
| 1 | 2026-11-10 | + company/year metadata filter (`retrieval.use_router_filter: true`) | 0.207 | 0.124 | 0.448 | 0.900 (n=17) | 3012 ms | keep |
| 2 | 2026-11-10 | + cross-encoder rerank (top_k 20 -> top_n 5; `rerank.enabled: true`, on top of exp 1) | 0.172 | 0.099 | 0.310 | 0.905 (n=13) | 2455 ms | revert |
| 3 | 2026-11-10 | Child chunk size 256 / 300 / 384 (offline pilot, mxbai-large, 11 questions; see Phase 2c) | n/a | n/a | n/a | n/a | n/a | 300 kept (all within noise; 256 and 384 project to about 2.0 h, the build gate) |
| 4 | | Context prefix on/off | | | | | | not tested (prefix kept on; v2 prefix with heading + units was only checked on one page in Phase 2b) |
| 5 | 2026-11-10 | RRF vs weighted fusion | 0.586 | 0.452 | 0.586 | 0.930 (n=22) | 2344 ms | done as experiment 12 (BM25 weight 0.25): keep |
| 6 | 2026-11-10 | Parent expansion for generation | 0.586 | 0.452 | 0.586 | 0.930 (n=22) | 3338 ms | done as experiment 13: not adopted |
| 7 | 2026-11-10 | Numeric-reasoning prompt | 0.586 | 0.452 | 0.552 | 0.846 (n=22) | 3669 ms | done as experiment 14: reverted |
| 8 | 2026-11-10 | Router year parsing: FY22 two-digit years + latest year when several are named (from the failure analysis; on top of exp 1) | 0.310 | 0.153 | 0.517 | 0.904 (n=19) | 2002 ms | keep |
| 9 | 2026-11-10 | Drop chunks under 15 tokens at retrieval (`retrieval.min_chunk_tokens: 0 -> 15`; from the failure analysis; on top of exp 8) | 0.345 | 0.196 | 0.517 | 0.909 (n=19) | 1672 ms | keep |
| 10 | 2026-11-10 | **Index v2** (one re-index): chunker v2 (embedder-tokenizer sizing, table row groups, rich prefix), mxbai-embed-large-v1, BM25 v2 (stemming), `index.version: v2`; router + min_chunk_tokens 15, rerank off | 0.517 | 0.411 | 0.552 | 0.908 (n=21) | 2553 ms | keep |
| 11 | 2026-11-10 | 5a: query instruction (`Represent this sentence for searching relevant passages: `) ON for mxbai-large (offline only, no eval spent; on top of exp 10) | 0.517 | 0.379 | n/a | n/a | n/a | not adopted (offline: MRR 0.411 -> 0.379, mean rank 13.1 -> 16.6) |
| 12 | 2026-11-10 | 5e: weighted fusion `retrieval.bm25_weight: 1.0 -> 0.25` (dense weight 1.0; on top of exp 10) | 0.586 | 0.452 | 0.586 | 0.930 (n=22) | 2344 ms | keep |
| 13 | 2026-11-10 | 5c: parent expansion for generation (`generation.expand_tables: true`, `max_context_tokens: 4000`; on top of exp 12) | 0.586 | 0.452 | 0.586 | 0.930 (n=22) | 3338 ms | not adopted (no change) |
| 14 | 2026-11-10 | 5f: numeric-reasoning prompt `grounded_v2` (formula, units, fiscal-year check, "I don't know" only if a number is absent; `generation.prompt_version`; on top of exp 12) | 0.586 | 0.452 | 0.552 | 0.846 (n=22) | 3669 ms | revert (accuracy and faithfulness worse) |
| 15 | 2026-11-10 | 5b: cross-encoder rerank on top of exp 12 (offline, all 29 dev questions: MiniLM-L12, jina-turbo, bge-reranker-base, bge-reranker-v2-m3-int8; pool top_k 20 and 30) | best 0.621 (jina-turbo, k20) vs 0.586 | all lower than 0.452 (best 0.387) | n/a | n/a | 1.3 to 11.9 s per question | not adopted (no eval spent) |
| 16 | 2026-11-10 | 5d: `rerank.top_n` 5 -> 8 (chunks given to the LLM; on top of exp 12) | n/a | n/a | n/a (invalid run: 11/29 errors) | n/a | n/a | **not completed** (Groq tokens-per-day limit); rerun when the quota frees |
| 17 | 2026-11-11 | Task 10 check (no tuning): cache, tracing, ACL, PII redaction, API added around the unchanged pipeline; eval builds the pipeline without the cache | 0.586 | 0.452 | 0.586 | 0.930 (n=22) | n/a (all 29 answers and 73 judge calls served from the LLM disk cache, 0 API calls) | no regression; run `20261110T204907_dev.json` |
| 18 | 2026-11-11 | 5d follow-up: offline recall at `top_n` 8 vs 5 on dev (production top-20 list sliced; no LLM, no eval spent) | recall@8 0.759 (22/29) vs recall@5 0.586 (17/29) | n/a | n/a | n/a | n/a | measured only; LLM run for accuracy not done (see notes) |
| 19 | 2026-11-11 | Citation parser fix (full-width brackets, spaces, ranges, several docs per bracket); not a tuning change, answers untouched | 0.586 | 0.452 | 0.586 | 0.930 (n=22) | n/a (all cached) | keep; run `20261111T090503_dev.json`, 0 API calls |

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

### Experiments 11 and 12 notes (query prefix, fusion weights; offline sweep `python -m src.jobs.fusion_sweep`, results `docs/fusion_sweep.json`)

The sweep retrieves the dense and BM25 candidate lists once per dev question (same router filter, same tiny-chunk filter, production depth 20 per list, plus depth 100 for ranks) and re-fuses them for every variant. Its baseline row reproduces the kept v2 numbers exactly (recall@5 0.517, MRR 0.411), so it matches the pipeline. No LLM calls.

| Variant (n=29) | recall@5 | MRR | mean rank | geo-mean | median |
|---|---|---|---|---|---|
| RRF 1:1, prefix off (exp 10) | 0.517 (15) | 0.411 | 13.1 | 4.5 | 3 |
| RRF 1:1, prefix ON | 0.517 (15) | 0.379 | 16.6 | 5.0 | 4 |
| dense only, prefix off / ON | 0.621 (18) / 0.621 (18) | 0.426 / 0.400 | 8.5 / 10.7 | 4.0 / 4.5 | 5 / 4 |
| dense 1 : BM25 0.5 | 0.552 (16) | 0.426 | 9.4 | 3.8 | 2 |
| **dense 1 : BM25 0.25 (adopted)** | 0.586 (17) | 0.452 | 8.6 | 3.6 | 2 |
| dense 1 : BM25 0.1 | 0.621 (18) | 0.480 | 8.5 | 3.8 | 3 |
| RRF 1:1, rrf_k 10 / 30 / 100 | 0.552 / 0.517 / 0.517 | 0.437 / 0.411 / 0.411 | 13.4 / 13.5 / 13.2 | 4.5 / 4.6 / 4.6 | 3 |
| BM25 only | 0.241 (7) | 0.159 | 32.8 | 15.6 | 16 |

- **5a (experiment 11):** the query instruction does not help mxbai-large (MRR down in both hybrid and dense-only, recall@5 unchanged). It stays off (`embeddings.query_prefix: ""`). No dev eval was spent because offline retrieval did not improve.
- **5e (experiment 12):** the trend is monotonic: the less weight BM25 gets, the better the ranking, matching the pilot where mxbai's dense ranking beat its hybrid. Adopted 0.25, not the best-scoring 0.1, deliberately: 0.1 is one question better (18 vs 17 at recall@5) but nearly dense-only, and with n=29 and three tried values I prefer to keep BM25 doing real work on exact numbers. `rrf_k` changes did nothing meaningful and were not adopted.
- **Dev eval for 12** (`data/eval/runs/20261110T140849_dev.json`, zero errors): vs exp 10, recall@5 0.517 -> 0.586 (15/29 -> 17/29; gained 00563, 00566; lost none), MRR 0.411 -> 0.452 (exactly the offline prediction), accuracy 0.552 -> 0.586 (16/29 -> 17/29; gained 04672; lost none), faithfulness 0.908 -> 0.930 (n=21 -> 22), "I don't know" 8 -> 7. Decision: keep. Honest size: +2 questions on recall@5 and +1 on accuracy, small on n=29, but no question got worse and the offline sweep predicted it. Note the weight was chosen on this same dev set (3 values tried), so the test split is the real check.
- Rollback: set `retrieval.bm25_weight: 1.0` (plain RRF).

### Experiment 13 notes (5c parent expansion)

- Change: each retrieved table row group is replaced by its full table parent in the LLM context (de-duplicated, capped at 4,000 regex tokens, falls back to the fragment when the table does not fit; `src/generation/context_expansion.py`). Retrieval ranking and metrics are untouched. Offline prefilter: the LLM context changes for 10 of 29 dev questions; mean context 1,128 -> 1,315 regex tokens, max 1,767 -> 3,358. Counted run, zero errors: `data/eval/runs/20261110T144247_dev.json`.
- Result vs exp 12: recall@5 0.586 and MRR 0.452 (unchanged by design), **accuracy 0.586 (17/29), faithfulness 0.930 (n=22), "I don't know" 7: all identical**. 6 answers changed wording (00585, 01226, 02608, 03069, 04672, 07507); none changed verdict; no question gained or lost. Decision: **not adopted** (no improvement, bigger prompts, p50 3338 ms over 10 uncached questions). Code and tests stay, flag default `false`.
- Why it probably does not help: of the 12 wrong dev answers, 8 are retrieval misses (recall 0) and expansion cannot fix those; the 3 "I don't know" answers with the gold page in the top 5 (03856, 03882, 04735) are not fixed by showing more of the same table, so the cause is reasoning/prompt, or the needed row is on a different page than the one retrieved.

### Experiment 14 notes (5f numeric-reasoning prompt)

- Change: `generation.prompt_version: grounded_v1 -> grounded_v2` (system prompt only; retrieval untouched; v1 kept as the default). Counted run, zero errors: `data/eval/runs/20261110T145336_dev.json`; all 29 answers regenerated (73 Groq calls); p95 74,980 ms comes from Groq rate-limit waits.
- Result vs exp 12: recall@5 0.586 / MRR 0.452 (unchanged by design), **accuracy 0.586 -> 0.552 (17/29 -> 16/29; gained 00720, lost 00790 and 02608), faithfulness 0.930 -> 0.846 (n=22 both)**, "I don't know" 7 -> 7. Decision: **revert** (both worse).
- It did not fix what it was written for: 03856, 03882 and 04735 (numeric questions with the gold page in the top 5) still answer "I don't know", so the numbers they need are not in the context (they are on another chunk or page); a prompt cannot supply them. 00566 (Verizon debt) is still wrong.
- Faithfulness fell on 6 questions (00464 1.0 -> 0.5, 01091 1.0 -> 0.5, 03069, 00757, 01351, 00917) because the step-by-step answers carry more claims for the judge to check; it rose on 01009 (0.5 -> 0.86). Rolled back: `prompt_version: grounded_v1`.
- Remaining dev failures (12 wrong at exp 12): 8 are retrieval misses (recall 0: 00438, 00585, 00720, 00799, 01009, 01226, 01279, 01981), 4 had the gold page in the top 5 (00566 wrong computation; 03856, 03882, 04735 "I don't know").

### Experiment 15 notes (5b rerank, offline on all 29 dev questions; `python -m src.jobs.rerank_dev`, raw lines in `docs/rerank_dev_partial.jsonl`)

Candidates are the current hybrid results (mxbai-large, BM25 weight 0.25, router filter). Reference without rerank: top_k 20 -> recall@5 0.586 (17/29), MRR 0.452, mean/geo/median gold rank 17.2 / 4.4 / 4; top_k 30 -> recall@5 0.586, MRR 0.477, 14.8 / 4.1 / 3.

| Reranker | top_k | recall@5 | MRR | geo rank | latency per question | questions better / worse | table-gold questions pushed down / up (of 12-13) |
|---|---|---|---|---|---|---|---|
| MiniLM-L12 | 20 | 0.483 | 0.285 | 6.7 | 2.3 s | 2 / 20 | **12 / 0** |
| jina-turbo | 20 | 0.621 | 0.387 | 5.2 | 1.3 s | 5 / 7 | 2 / 3 |
| bge-reranker-base | 20 | 0.552 | 0.374 | 5.4 | 5.4 s | 8 / 12 | 6 / 4 |
| bge-reranker-v2-m3-int8 | 20 | 0.552 | 0.331 | 5.6 | 11.9 s | 8 / 12 | 8 / 3 |
| MiniLM-L12 | 30 | 0.517 | 0.291 | 6.7 | 3.3 s | 4 / 18 | 11 / 1 |
| jina-turbo | 30 | 0.483 | 0.370 | 5.6 | 2.0 s | 5 / 11 | 4 / 4 |
| bge-reranker-base | 30 | 0.483 | 0.358 | 5.8 | 8.0 s | 7 / 13 | 7 / 3 |

- **No reranker beats the un-reranked ranking.** Every reranker lowers MRR (0.331 to 0.387 vs 0.452) and most lower recall@5; the single higher recall@5 (jina-turbo, k20, 0.621, +1 question) comes with a much worse MRR and 7 questions worse vs 5 better. On all 29 dev questions, the table concern is confirmed: MiniLM-L12 pushes the gold table chunk down in 12 of 12 table questions, and the bigger models do it in 6 to 8. Decision: **not adopted**; `rerank.enabled` stays false. No dev eval was spent.
- The last variant (bge-reranker-v2-m3-int8 at top_k 30) did not run: the job was stopped by the system because the machine was low on memory. It is not needed for the decision: the same model at top_k 20 averages 11.9 s per question (target about 2.5 s) and scores below no rerank.
- Side observation, not an experiment: widening the candidate pool without rerank (`retrieval.top_k` 20 -> 30) leaves recall@5 at 0.586 and lifts MRR 0.452 -> 0.477 and mean gold rank 17.2 -> 14.8. By the rule (keep only if recall@5 or accuracy improves) it is not worth a dev eval.

### Experiment 16 notes (5d top_n 8: NOT completed)

- Offline prefilter: with the current retrieval, gold page in the top 5 for 18 questions, at rank 6 for 3 (00585, 00790, 01279), ranks 10 to 14 for 4 more; so top_n 8 can only help up to 3 questions.
- The dev run (`data/eval/runs/20261110T153328_dev.json`) is **invalid: 11 of 29 questions errored with Groq HTTP 429 on tokens per day (TPD)**: two of the three accounts reported 199,087 and 198,001 of 200,000 tokens used. Its recall@5 0.379 and accuracy 0.345 are NOT results of top_n 8 and must not be quoted. 18 answers were served before the quota ran out and are in the LLM disk cache, so a rerun needs only about 11 new answers plus judges.
- Root cause on our side: `LLMClient` counts requests per key per day (719 of 3,000 today) but not tokens, so its 80% guard never fired. Config reverted to `top_n: 5`. **Fixed afterwards (commit `c1fe39f`):** `LLMClient` now counts tokens per key from each response's `usage` block, stops at 80% of `llm.primary.daily_token_budget` (200,000), and a 429 naming a per-day limit retires that key for the day instead of sleeping on the provider's hint. 5d itself is still untested (skipped by decision).

### Experiment 17 notes (Task 10 regression check, not a tuning change)

- Same four metrics as experiment 12, to three decimals, n=29, errors 0, 7 "I don't know". Everything came from the LLM disk cache (the prompts are byte-identical), so there is no latency figure and no tokens were spent. This shows the pipeline refactor (tracing, redaction, ACL hook) did not change what is retrieved or asked; it is not a fresh measurement of the LLM.
- PII redaction runs on the question inside the pipeline. Checked beforehand: none of the 150 FinanceBench questions matches the e-mail or phone patterns, so eval prompts are unchanged.
- Live check (`python main.py`, `curl`): `/health` ok, empty question 422, a Verizon dev question answered in 11.9 s the first time (includes loading the embedding model; the answer came from the LLM disk cache), then 0.2 s end to end from the semantic cache (1.5 ms inside the pipeline). A question containing an e-mail address reached the log redacted (0 occurrences of the address in `logs/app.log`).
- **Found, not fixed (out of Task 10 scope, would change eval inputs):** the model sometimes writes citations in full-width brackets, `【VERIZON_2022_10K p.23】`, which `parse_citations` does not match, so `citations` comes back empty although the answer cites a page. Accuracy and faithfulness do not use `citations`; recall/MRR use the retrieved list. Needs a decision before Task 11 reports citations.

### Experiment 18 notes (recall at top_n 8, offline, dev only)

- Production retrieval (router, top_k 20, weighted RRF), cut at k, 29 dev questions, numbers as measured: recall@3 14/29 (0.483), **@5 17/29 (0.586)**, @6 19/29 (0.655), **@8 22/29 (0.759)**, @10 25/29 (0.862), @20 25/29 (0.862). recall@5 reproduces the kept 0.586, so the method matches the eval.
- `top_n` 8 would put the gold page in the LLM context for 5 more questions (00438, 00585, 00790, 01279, 04672). Whether the LLM then answers them correctly, and what 3 more chunks do to faithfulness and tokens, is NOT measured: that needs a dev eval with every answer regenerated (about 100,000 tokens over the three keys, as the test run did) and has not been run. Experiment 16 therefore stays "not completed" for accuracy.
- Caveat for reading `compare_indexes` output: its `gold_rank` comes from a top-100 retrieval, whose fusion order can differ from the top-20 list the pipeline uses (question 01009 ranks 2 there but is a miss at 5 in production), so recall computed from `gold_rank` (18/29 at 5) is not the pipeline's recall (17/29). Use the production list, as above.
- The test split was not touched.

### Experiment 19 notes (citation parser)

- On the 153 distinct saved answers that name a document id, the old parser found a citation in 72. Formats seen in the data: `[DOC p.N]` (228), U+3010/3011 corner brackets `【DOC p.N】` (177), a trailing space inside the bracket (4), a page range with a non-breaking hyphen (3), parentheses (1), two documents in one bracket (1). The new parser (`src/generation/grounded_answer.py`) handles these and finds one in 151; the other 2 cite pages that were not in the retrieved context, which are dropped on purpose.
- Re-running the dev eval gave the same four metrics (accuracy and faithfulness judge the answer text, which is never rewritten; a test pins this), and all 22 answered dev questions now carry citations.
- The test-split result (experiment "Final") was produced with the old parser: its answers' text and metrics are unaffected, but `citations` in that run file under-report.

## Final (test split, run once)

Run `20261111T081204_test.json` (2026-11-11, `--split test --final`, config = experiment 12 unchanged, run once, n=29, errors 0). Numbers exactly as measured.

| Recall@5 | MRR | Accuracy | Faithfulness | p50 | p95 |
|----------|-----|----------|--------------|-----|-----|
| 0.552 (16/29) | 0.324 | 0.414 (12/29) | 0.842 (n=19) | 2620 ms | 10081 ms |

- "I don't know" on 10 of 29 questions (dev: 7); faithfulness is scored over the 19 answered ones only. Stage p50: retrieve 285 ms, llm 2329 ms. 65 Groq calls and about 104,000 tokens in total over the three keys (37,754 / 33,414 / 32,797), 0 errors, all calls uncached, so the latency covers all 29 questions.
- Dev to test: recall@5 0.586 -> 0.552, MRR 0.452 -> 0.324, accuracy 0.586 -> 0.414, faithfulness 0.930 -> 0.842. Every metric is lower on test. The BM25 weight was chosen on dev, and with n=29 one question is 0.034, so a drop of 5 questions in accuracy is larger than the usual +/-1 to 2 question run-to-run noise but the test split is also a different set of questions; no cause was tested, and nothing was tuned after this run.
- Against the plan's targets (recall@5 >= 0.80, accuracy 60-75%, faithfulness >= 0.90) the test numbers miss all three; p50 under 4 s is met.

## v1 best vs v2 baseline vs final (dev split, n=29, one run each, numbers exactly as measured)

| Config | recall@5 | MRR | accuracy | faithfulness | "I don't know" | p50 latency | run file |
|---|---|---|---|---|---|---|---|
| Naive baseline (exp 0: no router, no rerank) | 0.172 | 0.108 | 0.207 (6/29) | 0.947 (n=10) | 19 | 2441 ms | `20261110T061837_dev.json` |
| **v1 best** (exp 9: v1 index + router + year fix + min chunk 15) | 0.345 (10/29) | 0.196 | 0.517 (15/29) | 0.909 (n=19) | 10 | 1672 ms* | `20261110T070512_dev.json` |
| **v2 baseline** (exp 10: v2 index, same query-time settings) | 0.517 (15/29) | 0.411 | 0.552 (16/29) | 0.908 (n=21) | 8 | 2553 ms | `20261110T133621_dev.json` |
| **Final** (exp 12: v2 + BM25 weight 0.25) | **0.586 (17/29)** | **0.452** | **0.586 (17/29)** | **0.930 (n=22)** | **7** | 2344 ms | `20261110T140849_dev.json` |

\* latency is over only the questions whose answer was not already cached, so it is indicative; v2 retrieval costs about 290 ms per query (mxbai-large on CPU) of which most is query embedding.

Final config: v2 index (chunker v2: embedder-tokenizer sizing, table row groups, prefix `{company} FY{year} 10-K | page | heading | units`, child 300 / overlap 40 / cap 480; `mixedbread-ai/mxbai-embed-large-v1` 1024-d, no query prefix; BM25 v2 with stemming), router with year fix, `retrieval.min_chunk_tokens: 15`, weighted fusion 1.0 dense : 0.25 BM25 with `rrf_k: 60`, top_k 20, rerank off, top_n 5, prompt `grounded_v1`, no parent expansion. Answers, accuracy judge and faithfulness judge all by Groq `openai/gpt-oss-120b`.

Caveats: n=29, so one question is 0.034 of recall or accuracy; the BM25 weight was chosen on this same dev set (3 values tried); accuracy and faithfulness come from an LLM judge and an answerer with some run-to-run noise (an unchanged config swapped +1/-1 questions in exp 9). The test split was run once afterwards (see Final above): all four metrics were lower.

## Failure analysis, final config (12 wrong dev answers; run `20261110T140849_dev.json`, offline, no LLM calls)

12 of 29 answers are incorrect (v1: 14): 7 are "I don't know" (00720, 00799, 01279, 01981, 03856, 03882, 04735) and 5 are wrong answers (00438, 00566, 00585, 01009, 01226). Retrieval rank = first gold page in the production top 20 (blank = not in it); deep rank = same from a depth-100 list.

| Question id | Gold page(s): rank (top 20 / deep) | Cause | Fix candidates |
|---|---|---|---|
| 00720 | p.96: none / 33 | Retrieval miss + no natural evidence page. AmEx "what drove gross margin", gold answer is "not measured through gross margin" (a non-applicability question). Answered "I don't know". | Not fixable by retrieval; judge/gold style. Largest v1 -> v2 regression (18 -> 33 deep). |
| 00799 | p.52: none / 39 | Retrieval miss + derived metric. Amcor quick ratio is never named on the balance-sheet page. "I don't know". | Ratio-term expansion (rule based): "quick ratio" -> cash, short-term investments, receivables, current liabilities. Untested. |
| 01981 | p.45: none / 57 | Retrieval miss. Vague qualitative AmEx retention question (gold "Yes"). | Query rewriting; untested. |
| 01009 | p.4-5: 9 / 2 | Gold ranks 9th in the production list but 2nd in a deeper fusion: fusion depth matters. Our answer cited "200 countries" from p.32, not the geography list. | Wider candidate pool: top_k 30 lifted MRR in the rerank study (0.452 -> 0.477); untested end to end. |
| 01226 | p.27: 9 / 10 | Retrieval miss. 3M operating-margin drivers; p.30/31/20/21 (related MD&A pages) outrank p.27. Answer is a plausible table of drivers judged incorrect against the gold list. | Statement/section-aware boost; untested. |
| 00438 | p.54: 7 / 10 | Retrieval miss (was 26th in v1). Adobe operating margin FY22; gold is the income statement (p.54), our top 5 were MD&A pages. Margin has to be computed. | Statement-type boost; untested. |
| 00585 | p.55: 7 / 6 | Retrieval miss. Boeing effective tax rate; answered from p.24 with the sign lost (-0.6 / 14.8 vs gold 0.62 / -14.76). | top_n 8 would include it (gold at deep rank 6; experiment not completed, see exp 16). |
| 01279 | p.58: 6 / 6 | Retrieval miss by one position. AMD cash flow by activity; p.52 x3, p.59, p.51 in the top 5. Answered "I don't know". | top_n 8 (not completed). |
| 00566 | p.77: 4 / 1 | **Reasoning**: gold page retrieved. Verizon debt change: summed the wrong line items and concluded "Yes" (gold: debt fell by $229 million). | Reasoning prompt did not help (exp 14). |
| 03882 | p.50: 3 / 3 | Gold page retrieved at rank 3 but "I don't know" for Amcor net AR; the needed line item is probably in a different row group of the table (not verified). | Parent expansion did not change it (exp 13), so cause unclear; inspect the chunk. |
| 03856 | p.57, p.61: 1 / 1 | **Two pages needed.** Adobe operating cash flow ratio needs cash from operations (p.61) and current liabilities (p.57); only p.61 is in the top 5, so the ratio cannot be computed. Page-level recall@5 counts this as a hit. | Multi-page retrieval: diversify top_n across pages; untested. |
| 04735 | p.59, p.63: 1 / 1 | Same pattern: gold p.63 in the top 5, p.59 missing; "I don't know". | Same. |

What this says: 8 of the 12 are retrieval misses (gold page not in the top 5: 00720, 00799, 01981, 01009, 01226, 00438, 00585, 01279; of these 00720 is a non-applicability question and 01981 a vague qualitative one, and four more sit at rank 6 to 10, so they are near misses). The other 4 had a gold page in the top 5: 03856 and 04735 are two-page ratios where only one of the two needed pages was retrieved (page-level recall@5 counts them as hits, which hides this), 00566 is a wrong computation, and 03882 is unexplained (probably the needed row sits in another row group). No judge errors were found.

## Failure analysis, v1 config (historical, from experiment 1; superseded by the section above)

### v1 table (kept for the record)


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
