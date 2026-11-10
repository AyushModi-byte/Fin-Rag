"""Phase 5a/5e offline sweep on the dev split: query prefix on/off and fusion variants.
NO LLM calls. Candidate lists are retrieved once per question (production depth `retrieval.top_k`
per list, same router filters, same tiny-chunk filter) and then re-fused for every variant, so all
variants see identical candidates.

    python -m src.jobs.fusion_sweep --out docs/fusion_sweep.json
"""
import argparse
import json
import statistics
from pathlib import Path

from src.config import load_config
from src.eval.rag_metrics import mrr, recall_at_k
from src.query.query_router import route
from src.retrieval.retriever import rrf

PREFIX = "Represent this sentence for searching relevant passages: "
DEPTH = 100


def _rank(hits, gold, doc_id):
    for i, h in enumerate(hits, 1):
        if h.chunk.doc_id == doc_id and h.chunk.page in gold:
            return i
    return DEPTH + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="docs/fusion_sweep.json")
    args = ap.parse_args()
    cfg = load_config(args.config)

    from src.embeddings.embedder import Embedder
    from src.pipelines.index_pipeline import load_index
    from src.retrieval.retriever import HybridRetriever

    store, bm25 = load_index(cfg)
    emb = Embedder(cfg)
    retr = HybridRetriever(store, bm25, emb, cfg)
    docs = json.loads((Path(cfg.data.eval_dir) / "docs.json").read_text(encoding="utf-8"))
    known = sorted({d["company"] for d in docs.values()})
    rows = [json.loads(line) for line in open(Path(cfg.data.eval_dir) / "questions.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["split"] == "dev"]
    k, kk = cfg.eval.recall_k, cfg.retrieval.top_k

    cands = []
    for r in rows:
        filters = (route(r["question"], known).filters or None) if cfg.retrieval.use_router_filter else None
        entry = {"row": r}
        for tag, prefix in (("off", ""), ("on", PREFIX)):
            vec = emb.embed([prefix + r["question"]])[0]
            entry[f"dense_{tag}_{kk}"] = retr._drop_tiny(store.search(vec, kk, filters))
            entry[f"dense_{tag}_{DEPTH}"] = retr._drop_tiny(store.search(vec, DEPTH, filters))
        entry[f"bm25_{kk}"] = retr._drop_tiny(bm25.search(r["question"], kk, filters))
        entry[f"bm25_{DEPTH}"] = retr._drop_tiny(bm25.search(r["question"], DEPTH, filters))
        cands.append(entry)
    store.close()

    def evaluate(name, fuse):
        recs = []
        for c in cands:
            r = c["row"]
            top = fuse(c, kk)[:kk]
            deep = fuse(c, DEPTH)
            recs.append((recall_at_k(top, r["gold_pages"], k, r["doc_id"]), mrr(top, r["gold_pages"], r["doc_id"]),
                         _rank(deep, r["gold_pages"], r["doc_id"])))
        ranks = [x[2] for x in recs]
        res = {"variant": name, f"recall@{k}": round(sum(x[0] for x in recs) / len(recs), 3),
               "recall_n": sum(x[0] for x in recs), "mrr": round(sum(x[1] for x in recs) / len(recs), 3),
               "mean_rank": round(statistics.mean(ranks), 1), "geo_rank": round(statistics.geometric_mean(ranks), 1),
               "median_rank": statistics.median(ranks), "missing": sum(x > DEPTH for x in ranks), "ranks": ranks}
        print(json.dumps({k_: v for k_, v in res.items() if k_ != "ranks"}), flush=True)
        return res

    def hybrid(tag, wd, wb, rrf_k):
        def f(c, depth):
            return rrf([c[f"dense_{tag}_{depth}"], c[f"bm25_{depth}"]], k=rrf_k, weights=[wd, wb])
        return f

    def dense_only(tag):
        return lambda c, depth: c[f"dense_{tag}_{depth}"]

    out = []
    print("== 5a: query instruction (hybrid RRF k=60, weights 1:1; dense only)")
    out.append(evaluate("5a baseline: prefix OFF, hybrid 1:1 k60", hybrid("off", 1, 1, 60)))
    out.append(evaluate("5a prefix ON, hybrid 1:1 k60", hybrid("on", 1, 1, 60)))
    out.append(evaluate("5a prefix OFF, dense only", dense_only("off")))
    out.append(evaluate("5a prefix ON, dense only", dense_only("on")))
    print("== 5e: fusion (prefix OFF)")
    for wd, wb in ((1, 0.5), (1, 0.25), (1, 0.1)):
        out.append(evaluate(f"5e weighted dense {wd} : bm25 {wb}, k60", hybrid("off", wd, wb, 60)))
    for rk in (10, 30, 100):
        out.append(evaluate(f"5e RRF 1:1 rrf_k {rk}", hybrid("off", 1, 1, rk)))
    out.append(evaluate("5e BM25 only (reference)", lambda c, d: c[f"bm25_{d}"]))
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
