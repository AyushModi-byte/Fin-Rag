"""Retrieval metrics, LLM judges and the eval runner.

Judges make exactly one LLM call per question each (accuracy, faithfulness) and none at all when
there is nothing to judge ("I don't know" answers).
"""
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.llm.llm_client import BudgetNearlyExhausted, LLMError
from src.prompts.prompt_templates import (
    IDK,
    JUDGE_CORRECT_SYSTEM,
    JUDGE_FAITHFUL_SYSTEM,
    build_judge_correct_prompt,
    build_judge_faithful_prompt,
)


# ---------------- retrieval metrics (page level) ----------------

def _on_gold_page(hit, gold_pages, doc_id) -> bool:
    return hit.chunk.page in gold_pages and (doc_id is None or hit.chunk.doc_id == doc_id)


def recall_at_k(hits, gold_pages, k: int, doc_id: str | None = None) -> float:
    """1.0 if any of the top-k hits is on a gold page (of the right doc, if given), else 0.0."""
    return float(any(_on_gold_page(h, gold_pages, doc_id) for h in hits[:k]))


def mrr(hits, gold_pages, doc_id: str | None = None) -> float:
    """Reciprocal rank of the first hit on a gold page; 0.0 if none."""
    for rank, hit in enumerate(hits, start=1):
        if _on_gold_page(hit, gold_pages, doc_id):
            return 1.0 / rank
    return 0.0


# ---------------- LLM judges ----------------

def _is_idk(text: str) -> bool:
    return text.strip().rstrip(".").strip().lower().replace("’", "'") == IDK.lower()


def judge_correct(q: str, gold: str, pred: str, llm) -> bool:
    """One LLM call. "I don't know" is wrong by definition and costs no call."""
    if _is_idk(pred):
        return False
    reply = llm.complete(build_judge_correct_prompt(q, gold, pred), system=JUDGE_CORRECT_SYSTEM)
    words = re.findall(r"[A-Za-z]+", reply)
    return bool(words) and words[0].upper() == "CORRECT"


def _parse_claims(reply: str) -> list[dict]:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("faithfulness judge did not return JSON")
    try:
        claims = json.loads(reply[start : end + 1])["claims"]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"faithfulness judge returned unparseable JSON: {e}") from None
    if not isinstance(claims, list):
        raise ValueError("faithfulness judge 'claims' is not a list")
    return claims


def judge_faithful(answer: str, contexts: list[str], llm, max_output_tokens: int | None = None) -> float | None:
    """supported_claims / total_claims, from one LLM call. None when there is nothing to judge
    (I don't know, no contexts, or no claims)."""
    if _is_idk(answer) or not contexts:
        return None
    extra = {"max_output_tokens": max_output_tokens} if max_output_tokens else {}
    claims = _parse_claims(
        llm.complete(build_judge_faithful_prompt(answer, contexts), system=JUDGE_FAITHFUL_SYSTEM, **extra)
    )
    if not claims:
        return None
    supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported") is True)
    return supported / len(claims)


# ---------------- eval runner ----------------

def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(math.ceil(p * len(ordered)) - 1, 0)]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _delta(before: Counter, after: Counter) -> dict:
    return {k: after[k] - before.get(k, 0) for k in after if after[k] - before.get(k, 0) > 0}


def _load_rows(cfg, split: str) -> list[dict]:
    path = Path(cfg.data.eval_dir) / "questions.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted((r for r in rows if r["split"] == split), key=lambda r: r["id"])


def _summarize(cfg, split, limit, records, served, stopped_early) -> dict:
    k = cfg.eval.recall_k
    n = len(records)
    ok = [r for r in records if not r.get("error")]
    faith = [r["faithfulness"] for r in ok if r.get("faithfulness") is not None]
    fresh = [r for r in ok if r["generation_source"] != "cache"]  # cached answers have no real latency
    lat = {s: [r["latency_ms"][s] for r in fresh] for s in ("total", "retrieve", "rerank", "llm")}
    return {
        "split": split,
        "limit": limit,
        "n": n,
        "completed": n,
        "stopped_early": stopped_early,
        "errors": sum(1 for r in records if r.get("error")),
        "idk": sum(1 for r in ok if r["answer"].strip().rstrip(".") == IDK),
        f"recall_at_{k}": _mean([r["recall"] for r in records]),
        "mrr": _mean([r["mrr"] for r in records]),
        "accuracy": _mean([1.0 if r["correct"] else 0.0 for r in records]),
        "faithfulness": _mean(faith),
        "faithfulness_n": len(faith),
        "latency_n": len(fresh),
        "latency_p50_ms": _pct(lat["total"], 0.5),
        "latency_p95_ms": _pct(lat["total"], 0.95),
        "stage_p50_ms": {s: _pct(v, 0.5) for s, v in lat.items()},
        "served_by": dict(served),
        "generation_source": dict(Counter(r["generation_source"] for r in ok)),
        "config": {
            "router": cfg.retrieval.use_router_filter,
            "rerank": cfg.rerank.enabled,
            "top_k": cfg.retrieval.top_k,
            "top_n": cfg.rerank.top_n,
            "child_tokens": cfg.chunking.child_tokens,
            "context_prefix": cfg.chunking.context_prefix,
            "embedding_model": cfg.embeddings.model,
            "primary": cfg.llm.primary.model,
            "fallback": cfg.llm.fallback.model if cfg.llm.fallback else None,
        },
    }


def _save(cfg, split, limit, summary, records) -> Path:
    runs = Path(cfg.eval.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = f"_limit{limit}" if limit else ""
    path = runs / f"{stamp}_{split}{suffix}.json"
    summary["run_file"] = str(path)
    path.write_text(json.dumps({"summary": summary, "questions": records}, indent=2), encoding="utf-8")
    return path


def run_eval(split: str, limit: int | None = None, cfg=None, pipeline=None, llm=None,
             allow_test: bool = False) -> dict:
    """Run the pipeline on a split and score it. Writes data/eval/runs/<timestamp>.json.

    The test split is refused unless allow_test (Task 11 only). A question-level failure
    (LLMError) is recorded and the run continues; BudgetNearlyExhausted saves the partial run
    and re-raises so the user is told.
    """
    if split == "test" and not allow_test:
        raise PermissionError("the test split is reserved for the final run (Task 11); tune on dev")
    from src.config import load_config

    cfg = cfg or load_config()
    own_pipeline = pipeline is None
    if own_pipeline:
        from src.pipelines.query_pipeline import build_pipeline

        pipeline = build_pipeline(cfg)
    llm = llm or pipeline.llm

    rows = _load_rows(cfg, split)
    if limit:
        rows = rows[:limit]
    k = cfg.eval.recall_k
    served_start = Counter(getattr(llm, "served", Counter()))
    records: list[dict] = []
    stopped_early = False
    try:
        for i, row in enumerate(rows, start=1):
            rec = {"id": row["id"], "question": row["question"], "gold_answer": row["gold_answer"],
                   "doc_id": row["doc_id"], "gold_pages": row["gold_pages"]}
            try:
                before = Counter(getattr(llm, "served", Counter()))
                out = pipeline.run(row["question"])
                gen = _delta(before, Counter(getattr(llm, "served", Counter())))
                result = out.result
                rec.update(
                    answer=result.answer,
                    citations=[c.model_dump() for c in result.citations],
                    retrieved=[{"doc_id": h.chunk.doc_id, "page": h.chunk.page} for h in out.ranked[:k]],
                    latency_ms=result.latency_ms,
                    generation_source=next(iter(gen), "none"),
                    recall=recall_at_k(out.ranked, row["gold_pages"], k, row["doc_id"]),
                    mrr=mrr(out.ranked, row["gold_pages"], row["doc_id"]),
                    correct=judge_correct(row["question"], row["gold_answer"], result.answer, llm),
                    faithfulness=judge_faithful(
                        result.answer, result.contexts, llm, cfg.eval.judge_max_output_tokens
                    ),
                )
            except BudgetNearlyExhausted:
                stopped_early = True
                raise
            except (LLMError, ValueError) as e:
                rec.update(error=f"{type(e).__name__}: {e}", recall=0.0, mrr=0.0, correct=False,
                           faithfulness=None, answer="", generation_source="none", latency_ms={})
            records.append(rec)
            print(f"[{i}/{len(rows)}] {row['id']} recall={rec['recall']:.0f} correct={rec['correct']}"
                  + (f" ERROR {rec['error']}" if rec.get("error") else ""), flush=True)
    except BudgetNearlyExhausted:
        summary = _summarize(cfg, split, limit, records, _delta(served_start, Counter(getattr(llm, "served", Counter()))), True)
        _save(cfg, split, limit, summary, records)
        raise
    finally:
        if own_pipeline and hasattr(pipeline, "close"):
            pipeline.close()

    served = _delta(served_start, Counter(getattr(llm, "served", Counter())))
    summary = _summarize(cfg, split, limit, records, served, stopped_early)
    _save(cfg, split, limit, summary, records)
    return summary
