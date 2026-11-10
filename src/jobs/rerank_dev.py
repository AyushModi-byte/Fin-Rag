"""Phase 5b offline: rerankers on all 29 dev questions with the CURRENT retrieval config. No LLM calls.

    python -m src.jobs.rerank_dev --top-k 20 30 --out docs/rerank_dev.json

For each candidate pool size (top_k) the hybrid retriever produces the candidates once (same router
filter and settings as the pipeline); every reranker then reorders that pool. Reported per variant:
recall@5, MRR, gold-page rank (mean / geo-mean / median), latency per question, and the table check:
for questions whose best gold chunk is a table chunk, did the rerank move it down?
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from src.chunking.chunker import embedding_text
from src.config import load_config
from src.query.query_router import route

MODELS = [
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "jinaai/jina-reranker-v1-turbo-en",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3-int8",
]


def _is_table(chunk) -> bool:
    return chunk.table_id is not None or chunk.text.lstrip().startswith("|")


def _first_gold(hits, gold, doc_id):
    for i, h in enumerate(hits, 1):
        if h.chunk.doc_id == doc_id and h.chunk.page in gold:
            return i, h
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, nargs="+", default=[20, 30])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--out", default="docs/rerank_dev.json")
    args = ap.parse_args()
    cfg = load_config()

    from fastembed.rerank.cross_encoder import TextCrossEncoder

    from src.embeddings.embedder import Embedder
    from src.pipelines.index_pipeline import load_index
    from src.retrieval.retriever import HybridRetriever

    store, bm25 = load_index(cfg)
    retr = HybridRetriever(store, bm25, Embedder(cfg), cfg)
    docs = json.loads((Path(cfg.data.eval_dir) / "docs.json").read_text(encoding="utf-8"))
    known = sorted({d["company"] for d in docs.values()})
    rows = [json.loads(line) for line in open(Path(cfg.data.eval_dir) / "questions.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "dev"]
    k = cfg.eval.recall_k
    DEPTH = 100

    results = []
    for top_k in args.top_k:
        pools = []
        for r in rows:
            f = route(r["question"], known).filters or None
            pools.append(retr.retrieve(r["question"], top_k, f))
        base_ranks = []
        for r, pool in zip(rows, pools, strict=True):
            rk, _ = _first_gold(pool, r["gold_pages"], r["doc_id"])
            base_ranks.append(rk or DEPTH + 1)

        def summarize(name, ranks, lat=None, extra=None):
            res = {"variant": name, "top_k": top_k, f"recall@{k}": round(sum(x <= k for x in ranks) / len(ranks), 3),
                   "recall_n": sum(x <= k for x in ranks),
                   "mrr": round(sum(1 / x if x <= DEPTH else 0 for x in ranks) / len(ranks), 3),
                   "mean_rank": round(statistics.mean(ranks), 1),
                   "geo_rank": round(statistics.geometric_mean(ranks), 1),
                   "median_rank": statistics.median(ranks), "ranks": ranks}
            if lat:
                res["latency_s_mean"] = round(statistics.mean(lat), 2)
                res["latency_s_max"] = round(max(lat), 2)
            res.update(extra or {})
            print(json.dumps({a: b for a, b in res.items() if a != "ranks"}), flush=True)
            results.append(res)

        summarize(f"no rerank (top_k {top_k})", base_ranks)
        for name in args.models:
            model = TextCrossEncoder(model_name=name)
            ranks, lat, table_down, table_up, table_n = [], [], 0, 0, 0
            for r, pool, b in zip(rows, pools, base_ranks, strict=True):
                texts = [embedding_text(h.chunk) for h in pool]
                t0 = time.perf_counter()
                scores = list(model.rerank(r["question"], texts)) if pool else []
                lat.append(time.perf_counter() - t0)
                order = sorted(range(len(pool)), key=lambda i: -scores[i])
                reranked = [pool[i] for i in order]
                rk, gh = _first_gold(reranked, r["gold_pages"], r["doc_id"])
                ranks.append(rk or DEPTH + 1)
                _, base_hit = _first_gold(pool, r["gold_pages"], r["doc_id"])
                if base_hit is not None and _is_table(base_hit.chunk):
                    table_n += 1
                    table_down += (rk or DEPTH + 1) > b
                    table_up += (rk or DEPTH + 1) < b
            summarize(f"{name} (top_k {top_k})", ranks, lat,
                      {"table_gold_questions": table_n, "table_gold_worse": table_down,
                       "table_gold_better": table_up,
                       "improved": sum(a < b for a, b in zip(ranks, base_ranks)),
                       "worsened": sum(a > b for a, b in zip(ranks, base_ranks))})
            del model
    store.close()
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
