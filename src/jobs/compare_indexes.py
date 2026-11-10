"""Phase 4 offline retrieval check on the dev split. NO LLM calls.

    python -m src.jobs.compare_indexes run --config config.v1.yaml --out .cache/scratch/dev_v1.json
    python -m src.jobs.compare_indexes run --config config.yaml    --out .cache/scratch/dev_v2.json
    python -m src.jobs.compare_indexes diff .cache/scratch/dev_v1.json .cache/scratch/dev_v2.json

`run` uses the real router + retriever (+ rerank if the config enables it) exactly as the pipeline
does: recall@k / MRR from the production top_k list, and the gold-page rank from a deeper list.
"""
import argparse
import json
import statistics
from pathlib import Path

from src.config import load_config
from src.eval.rag_metrics import mrr, recall_at_k
from src.query.query_router import route

DEPTH = 100


def _dev_rows(cfg) -> list[dict]:
    rows = [json.loads(line) for line in open(Path(cfg.data.eval_dir) / "questions.jsonl", encoding="utf-8")]
    return [r for r in rows if r["split"] == "dev"]


def _rank(hits, gold_pages, doc_id) -> int:
    for i, h in enumerate(hits, 1):
        if h.chunk.doc_id == doc_id and h.chunk.page in gold_pages:
            return i
    return DEPTH + 1


def run(cfg, out: str) -> dict:
    from src.embeddings.embedder import Embedder
    from src.pipelines.index_pipeline import load_index
    from src.retrieval.retriever import HybridRetriever

    store, bm25 = load_index(cfg)
    try:
        retriever = HybridRetriever(store, bm25, Embedder(cfg), cfg)
        docs = json.loads((Path(cfg.data.eval_dir) / "docs.json").read_text(encoding="utf-8"))
        known = sorted({d["company"] for d in docs.values()})
        k = cfg.eval.recall_k
        recs = []
        for r in _dev_rows(cfg):
            filters = None
            if cfg.retrieval.use_router_filter:
                routed = route(r["question"], known)
                filters = routed.filters or None
            top = retriever.retrieve(r["question"], cfg.retrieval.top_k, filters)
            deep = retriever.retrieve(r["question"], DEPTH, filters)
            recs.append({
                "id": r["id"], "doc_id": r["doc_id"], "gold_pages": r["gold_pages"], "filters": filters,
                f"recall@{k}": recall_at_k(top, r["gold_pages"], k, r["doc_id"]),
                "mrr": mrr(top, r["gold_pages"], r["doc_id"]),
                "gold_rank": _rank(deep, r["gold_pages"], r["doc_id"]),
                "top5": [[h.chunk.doc_id, h.chunk.page, h.chunk.row_group] for h in top[:k]],
            })
    finally:
        store.close()
    ranks = [x["gold_rank"] for x in recs]
    summary = {
        "index_version": cfg.index.version, "n": len(recs),
        f"recall@{k}": round(sum(x[f"recall@{k}"] for x in recs) / len(recs), 3),
        "mrr": round(sum(x["mrr"] for x in recs) / len(recs), 3),
        "mean_gold_rank": round(statistics.mean(ranks), 1),
        "geo_mean_gold_rank": round(statistics.geometric_mean(ranks), 1),
        "median_gold_rank": statistics.median(ranks),
        "missing": sum(r > DEPTH for r in ranks),
    }
    Path(out).write_text(json.dumps({"summary": summary, "questions": recs}, indent=1), encoding="utf-8")
    print(json.dumps(summary))
    return summary


def diff(a_path: str, b_path: str, k: int) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print("v1:", json.dumps(a["summary"]))
    print("v2:", json.dumps(b["summary"]))
    wins = losses = 0
    print(f"{'question':14s} {'gold':>8s} {'rank v1':>8s} {'rank v2':>8s} {f'R@{k} v1':>7s} {f'R@{k} v2':>7s}")
    for qa, qb in zip(a["questions"], b["questions"], strict=True):
        assert qa["id"] == qb["id"]
        mark = ""
        if qb["gold_rank"] < qa["gold_rank"]:
            wins += 1
            mark = "  v2 better"
        elif qb["gold_rank"] > qa["gold_rank"]:
            losses += 1
            mark = "  v2 WORSE"
        if qa[f"recall@{k}"] != qb[f"recall@{k}"]:
            mark += f"  (recall@{k} {'gained' if qb[f'recall@{k}'] > qa[f'recall@{k}'] else 'LOST'})"
        print(f"{qa['id'][-9:]:14s} {str(qa['gold_pages']):>8s} {qa['gold_rank']:>8d} {qb['gold_rank']:>8d} "
              f"{qa[f'recall@{k}']:>7.0f} {qb[f'recall@{k}']:>7.0f}{mark}")
    print(f"gold rank: v2 better on {wins}, worse on {losses}, same on {len(a['questions']) - wins - losses}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--config", default="config.yaml")
    r.add_argument("--out", required=True)
    d = sp.add_parser("diff")
    d.add_argument("a")
    d.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "run":
        run(load_config(args.config), args.out)
    else:
        diff(args.a, args.b, load_config().eval.recall_k)


if __name__ == "__main__":
    main()
