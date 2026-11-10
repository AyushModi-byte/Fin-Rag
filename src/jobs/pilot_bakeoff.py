"""Phase 2: offline pilot bake-off on a few filings. NO LLM calls.

    python -m src.jobs.pilot_bakeoff retrieval --models bge-small bge-base arctic-m nomic mxbai-large
    python -m src.jobs.pilot_bakeoff rerank --embed-model bge-base

The router hard-filters by company + fiscal year, so retrieval only compares chunks inside one
filing. A small in-memory index of the filings that cover the most dev questions is therefore a
valid test for those questions. Here the filter is the gold doc_id (an "oracle router"; the real
router is right for 27/29 dev questions). Metrics are page level: rank of the first chunk that lies
on a gold page of the gold doc (1-based; depth+1 when missing).
"""
import argparse
import collections
import copy
import json
import statistics
import time
from pathlib import Path

import numpy as np

from src.chunking.chunker import chunk_pages, count_tokens, embedding_text
from src.config import load_config
from src.embeddings.embedder import Embedder
from src.embeddings.tokenizer import EmbedTokenizer
from src.ingestion.loader import load_corpus
from src.lexical.bm25_index import BM25Index
from src.retrieval.retriever import rrf
from src.schemas.models import Chunk, Hit

BGE_Q = "Represent this sentence for searching relevant passages: "
MODELS = {  # short name -> (fastembed id, query prefix, doc prefix)
    "bge-small": ("BAAI/bge-small-en-v1.5", BGE_Q, ""),
    "bge-base": ("BAAI/bge-base-en-v1.5", BGE_Q, ""),
    "arctic-m": ("snowflake/snowflake-arctic-embed-m", BGE_Q, ""),
    "nomic": ("nomic-ai/nomic-embed-text-v1.5", "search_query: ", "search_document: "),
    "mxbai-large": ("mixedbread-ai/mxbai-embed-large-v1", BGE_Q, ""),
    "gte-base": ("thenlper/gte-base", "", ""),
}
DEPTH = 100
CACHE = Path(".cache/pilot")


def dev_rows(cfg, n_docs: int) -> tuple[list[dict], list[str]]:
    rows = [json.loads(line) for line in open(Path(cfg.data.eval_dir) / "questions.jsonl", encoding="utf-8")]
    dev = [r for r in rows if r["split"] == "dev"]
    top = [d for d, _ in collections.Counter(r["doc_id"] for r in dev).most_common(n_docs)]
    return [r for r in dev if r["doc_id"] in top], top


def gold_rank(chunks: list[Chunk], doc_id: str, gold_pages: list[int]) -> int:
    for i, c in enumerate(chunks, 1):
        if c.doc_id == doc_id and c.page in gold_pages:
            return i
    return DEPTH + 1


def summarize(ranks: list[int]) -> dict:
    n = len(ranks)
    return {
        "mean_rank": round(statistics.mean(ranks), 1),
        "median_rank": statistics.median(ranks),
        "recall@5": round(sum(r <= 5 for r in ranks) / n, 3),
        "recall@20": round(sum(r <= 20 for r in ranks) / n, 3),
        "mrr": round(sum(1 / r if r <= DEPTH else 0 for r in ranks) / n, 3),
        "missing": sum(r > DEPTH for r in ranks),
    }


def pilot_cfg(base, short: str, child_tokens: int | None, max_embed: int | None = None,
              table_group: int | None = None):
    cfg = copy.deepcopy(base)
    if max_embed:
        cfg.chunking.max_embed_tokens = max_embed
    if table_group:
        cfg.chunking.table_group_tokens = table_group
    model, qp, dp = MODELS[short]
    cfg.embeddings.model, cfg.embeddings.query_prefix, cfg.embeddings.doc_prefix = model, qp, dp
    if child_tokens:
        cfg.chunking.child_tokens = child_tokens
    return cfg


def embed_cached(embedder: Embedder, chunks: list[Chunk], tag: str) -> tuple[np.ndarray, float]:
    """Embed documents (cached on disk by tag); returns (vectors, seconds spent embedding)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{tag}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return z["v"], float(z["secs"])
    t0 = time.perf_counter()
    vecs = embedder.embed_documents([embedding_text(c) for c in chunks])
    secs = time.perf_counter() - t0
    np.savez(f, v=vecs, secs=secs)
    return vecs, secs


def dense_rank(vecs, chunks, qvec, doc_id, min_tokens):
    qv = qvec / np.linalg.norm(qvec)
    v = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = v @ qv
    order = np.argsort(-sims)
    out = [Hit(chunk_id=chunks[i].id, score=float(sims[i]), chunk=chunks[i]) for i in order
           if chunks[i].doc_id == doc_id and count_tokens(chunks[i].text) >= min_tokens]
    return out[:DEPTH]


def bm25_rank(bm25: BM25Index, q, doc_id, min_tokens):
    hits = bm25.search(q, DEPTH * 2, {"doc_id": doc_id})
    return [h for h in hits if count_tokens(h.chunk.text) >= min_tokens][:DEPTH]


def evaluate(rows, chunks, vecs, embedder, cfg, use_qp: bool, bm25s: dict) -> dict:
    """Per-variant ranks for every question. bm25s: name -> BM25Index built on `chunks`."""
    mt = cfg.retrieval.min_chunk_tokens
    ranks = collections.defaultdict(list)
    for r in rows:
        q, d, g = r["question"], r["doc_id"], r["gold_pages"]
        qvec = embedder.embed([(embedder.query_prefix if use_qp else "") + q])[0]
        dense = dense_rank(vecs, chunks, qvec, d, mt)
        ranks["dense"].append(gold_rank([h.chunk for h in dense], d, g))
        for name, bm in bm25s.items():
            lex = bm25_rank(bm, q, d, mt)
            ranks[name].append(gold_rank([h.chunk for h in lex], d, g))
            fused = rrf([dense, lex], k=cfg.retrieval.rrf_k)
            ranks[f"hybrid+{name}"].append(gold_rank([h.chunk for h in fused], d, g))
    return {k: v for k, v in ranks.items()}


def load_docs(cfg, doc_ids):
    corpus = load_corpus(Path(cfg.data.raw_dir), Path(cfg.data.eval_dir) / "docs.json",
                         Path(cfg.data.parsed_dir), reuse_parsed=True)
    return [(m, p) for m, p in corpus if m.doc_id in doc_ids]


def v1_chunks(cfg, doc_ids) -> list[Chunk]:
    path = Path(cfg.index.dir) / "v1" / "bm25" / "chunks.jsonl"
    out = [Chunk(**json.loads(line)) for line in open(path, encoding="utf-8") if line.strip()]
    return [c for c in out if c.doc_id in doc_ids]


def run_retrieval(args) -> None:
    base = load_config()
    rows, docs = dev_rows(base, args.n_docs)
    corpus = load_docs(base, set(docs))
    print(f"docs={docs} dev questions={len(rows)}", flush=True)
    results: dict = {"docs": docs, "n_questions": len(rows), "depth": DEPTH, "variants": {}}
    out_path = Path(args.out)

    def record(name, ranks, extra):
        for sub, rk in ranks.items():
            results["variants"][f"{name} | {sub}"] = {**summarize(rk), **extra, "ranks": rk}
            print(f"{name:34s} | {sub:22s} {json.dumps(summarize(rk))}", flush=True)
        out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")

    for short in args.models:
        cfg = pilot_cfg(base, short, args.child_tokens, args.max_embed_tokens, args.table_group_tokens)
        emb = Embedder(cfg)
        tok = EmbedTokenizer(cfg.embeddings.model)
        chunks = [c for m, p in corpus for c in chunk_pages(p, m, cfg, tok=tok)[1]]
        tag_base = f"{short}_c{cfg.chunking.child_tokens}_{len(chunks)}"
        if args.max_embed_tokens or args.table_group_tokens:
            tag_base += f"_m{args.max_embed_tokens}_g{args.table_group_tokens}"
        label = f"c{cfg.chunking.child_tokens} m{cfg.chunking.max_embed_tokens} g{getattr(cfg.chunking, 'table_group_tokens', None)}"
        vecs, secs = embed_cached(emb, chunks, tag_base)
        cps = len(chunks) / secs
        extra = {"chunks": len(chunks), "embed_chunks_per_s": round(cps, 1),
                 "projected_full_hours": round(16101 / cps / 3600, 2)}
        bm = {"bm25v1": BM25Index("en", "v1"), "bm25v2": BM25Index("en", "v2", True)}
        for b in bm.values():
            b.build(chunks)
        variants = {"": False}
        if emb.query_prefix:
            variants = {" qprefix=off": False, " qprefix=on": True}
        for suffix, use_qp in variants.items():
            ranks = evaluate(rows, chunks, vecs, emb, cfg, use_qp, bm)
            record(f"v2 {short} {label}{suffix}", ranks, extra)
        if short == "bge-small" and not args.skip_v1:  # control: the v1 chunks, v1 BM25
            c1 = v1_chunks(base, set(docs))
            v1cfg = copy.deepcopy(cfg)
            v1cfg.embeddings.query_prefix = ""
            e1 = Embedder(v1cfg)
            vecs1, secs1 = embed_cached(e1, c1, f"v1ctl_bge-small_{len(c1)}")
            b1 = BM25Index("en", "v1")
            b1.build(c1)
            ranks = evaluate(rows, c1, vecs1, e1, v1cfg, False, {"bm25v1": b1})
            record("v1 chunks bge-small (control)", ranks,
                   {"chunks": len(c1), "embed_chunks_per_s": round(len(c1) / secs1, 1)})
    results["questions"] = [{"id": r["id"], "doc_id": r["doc_id"], "gold_pages": r["gold_pages"]} for r in rows]
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")


def run_rerank(args) -> None:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    base = load_config()
    rows, docs = dev_rows(base, args.n_docs)
    corpus = load_docs(base, set(docs))
    cfg = pilot_cfg(base, args.embed_model, args.child_tokens)
    emb = Embedder(cfg)
    tok = EmbedTokenizer(cfg.embeddings.model)
    chunks = [c for m, p in corpus for c in chunk_pages(p, m, cfg, tok=tok)[1]]
    vecs, _ = embed_cached(emb, chunks, f"{args.embed_model}_c{cfg.chunking.child_tokens}_{len(chunks)}")
    bm = BM25Index("en", "v2", True)
    bm.build(chunks)
    use_qp = args.qprefix
    cands = {}
    for r in rows:
        qvec = emb.embed([(emb.query_prefix if use_qp else "") + r["question"]])[0]
        dense = dense_rank(vecs, chunks, qvec, r["doc_id"], cfg.retrieval.min_chunk_tokens)
        lex = bm25_rank(bm, r["question"], r["doc_id"], cfg.retrieval.min_chunk_tokens)
        cands[r["id"]] = rrf([dense, lex], k=cfg.retrieval.rrf_k)
    before = [gold_rank([h.chunk for h in cands[r["id"]]], r["doc_id"], r["gold_pages"]) for r in rows]
    top_k = args.top_k
    print("no rerank:", json.dumps(summarize(before)), flush=True)
    results = {"embed_model": args.embed_model, "top_k": top_k, "no_rerank": summarize(before), "rerankers": {}}
    for name in args.rerankers:
        try:
            model = TextCrossEncoder(model_name=name)
        except Exception as e:  # unsupported / download failure: record and go on
            print(name, "FAILED to load:", repr(e)[:200], flush=True)
            continue
        after, lat, moved = [], [], []
        for r, b in zip(rows, before, strict=True):
            hits = cands[r["id"]][:top_k]
            docs_txt = [embedding_text(h.chunk) for h in hits]
            t0 = time.perf_counter()
            scores = list(model.rerank(r["question"], docs_txt))
            lat.append(time.perf_counter() - t0)
            order = sorted(range(len(hits)), key=lambda i: -scores[i])
            rk = gold_rank([hits[i].chunk for i in order], r["doc_id"], r["gold_pages"])
            # candidates beyond top_k keep their place after the reranked block
            after.append(rk if rk <= top_k else b)
            moved.append((b, rk))
        s = summarize(after)
        s.update(latency_s_mean=round(statistics.mean(lat), 2), latency_s_max=round(max(lat), 2),
                 gold_in_pool=sum(b <= top_k for b in before),
                 improved=sum(a < b for a, b in zip(after, before)),
                 worsened=sum(a > b for a, b in zip(after, before)), ranks=after)
        results["rerankers"][name] = s
        print(f"{name:45s} {json.dumps({k: v for k, v in s.items() if k != 'ranks'})}", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    results["before_ranks"] = before


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("retrieval")
    a.add_argument("--models", nargs="+", default=["bge-small"])
    a.add_argument("--n-docs", type=int, default=3)
    a.add_argument("--child-tokens", type=int, default=None)
    a.add_argument("--skip-v1", action="store_true")
    a.add_argument("--max-embed-tokens", type=int, default=None)
    a.add_argument("--table-group-tokens", type=int, default=None)
    a.add_argument("--out", default="docs/pilot_retrieval.json")
    b = sp.add_parser("rerank")
    b.add_argument("--embed-model", default="bge-small")
    b.add_argument("--qprefix", action="store_true")
    b.add_argument("--n-docs", type=int, default=3)
    b.add_argument("--child-tokens", type=int, default=None)
    b.add_argument("--top-k", type=int, default=20)
    b.add_argument("--rerankers", nargs="+", default=[
        "Xenova/ms-marco-MiniLM-L-6-v2", "Xenova/ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-base",
        "BAAI/bge-reranker-v2-m3-int8", "jinaai/jina-reranker-v1-turbo-en", "jinaai/jina-reranker-v1-tiny-en"])
    b.add_argument("--out", default="docs/pilot_rerank.json")
    args = ap.parse_args()
    run_retrieval(args) if args.cmd == "retrieval" else run_rerank(args)


if __name__ == "__main__":
    main()
