"""Phase 0: embedding speed benchmark on a random sample of v1 chunks. No LLM calls.

    python -m src.jobs.bench_embed --n 300 --variants all
    python -m src.jobs.bench_embed --n 60 --variants baseline

Each variant is (batch_size, onnxruntime threads, fastembed parallel, length-sorted, max_length).
Prints chunks/sec and tokens/sec per variant, and writes docs/bench_embed.json.
"""
import argparse
import json
import random
import time
from pathlib import Path

import psutil
from fastembed import TextEmbedding

from src.chunking.chunker import embedding_text
from src.config import load_config
from src.schemas.models import Chunk


def load_sample(cfg, n: int, seed: int) -> list[str]:
    path = Path(cfg.index.dir) / "v1" / "bm25" / "chunks.jsonl"
    with path.open(encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
    random.Random(seed).shuffle(chunks)
    return [embedding_text(c) for c in chunks[:n]]


def run_variant(model_id: str, texts: list[str], v: dict) -> dict:
    model = TextEmbedding(model_name=model_id, threads=v["threads"])
    tok = model.model.tokenizer if hasattr(model.model, "tokenizer") else None
    if v.get("max_length") and tok is not None:
        tok.enable_truncation(max_length=v["max_length"])
    docs = texts
    if v.get("sort"):
        docs = sorted(texts, key=len)
    list(model.embed(docs[:8], batch_size=8))  # warm-up (session init, allocator)
    t0 = time.perf_counter()
    cpu0 = psutil.cpu_percent(interval=None)
    out = list(model.embed(docs, batch_size=v["batch"], parallel=v.get("parallel")))
    dt = time.perf_counter() - t0
    assert len(out) == len(docs)
    return {
        **v,
        "seconds": round(dt, 2),
        "chunks_per_s": round(len(docs) / dt, 2),
        "cpu_pct_during": psutil.cpu_percent(interval=None),
    }


def variants(kind: str, cores: int, logical: int) -> list[dict]:
    base = {"batch": 32, "threads": None, "parallel": None, "sort": False, "max_length": None}
    if kind == "baseline":
        return [base]
    if kind == "refine":
        return [{**base, "batch": b, "sort": True} for b in (1, 2, 4)]
    vs = [base]
    for b in (8, 64):
        vs.append({**base, "batch": b})
    for t in (cores, logical, 4):
        vs.append({**base, "threads": t})
    vs.append({**base, "sort": True})
    vs.append({**base, "sort": True, "threads": cores})
    vs.append({**base, "max_length": 256})
    vs.append({**base, "parallel": 2, "threads": 1})
    vs.append({**base, "parallel": 4, "threads": 1})
    return vs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="docs/bench_embed.json")
    args = ap.parse_args()
    cfg = load_config()
    model_id = args.model or cfg.embeddings.model
    texts = load_sample(cfg, args.n, args.seed)
    cores = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    print(f"model={model_id} n={len(texts)} physical={cores} logical={logical}", flush=True)
    results = []
    for v in variants(args.variants, cores, logical):
        r = run_variant(model_id, texts, v)
        results.append(r)
        print(json.dumps(r), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
