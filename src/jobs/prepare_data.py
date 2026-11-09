"""Download FinanceBench 10-K subset and build the frozen dev/test eval set.

Usage: python -m src.jobs.prepare_data [--config config.yaml]
"""
import argparse
import collections
import json
import logging
import random
import re
import urllib.request
from pathlib import Path

from src.config import load_config

logger = logging.getLogger("finrag.prepare_data")

_TENK = re.compile(r"^(?P<ticker>.+)_(?P<year>\d{4})_10K$")


def _is_10k(doc_name: str) -> bool:
    return _TENK.match(doc_name) is not None


def select_docs(questions: list[dict], n_companies: int = 10) -> list[str]:
    """Pick the companies with the most 10-K questions; return their 10-K doc ids (sorted)."""
    tenk = [q for q in questions if _is_10k(q["doc_name"])]
    counts = collections.Counter(q["company"] for q in tenk)
    ranked = sorted(counts, key=lambda c: (-counts[c], c))[:n_companies]
    chosen = set(ranked)
    return sorted({q["doc_name"] for q in tenk if q["company"] in chosen})


def assign_splits(ids: list[str], dev_ratio: float, seed: int) -> dict[str, str]:
    """Deterministic dev/test split, independent of input order."""
    ordered = sorted(ids)
    random.Random(seed).shuffle(ordered)
    n_dev = round(len(ordered) * dev_ratio)
    return {i: ("dev" if k < n_dev else "test") for k, i in enumerate(ordered)}


def build_eval_rows(
    questions: list[dict], doc_ids: list[str], dev_ratio: float, seed: int
) -> tuple[list[dict], dict[str, dict]]:
    """Eval rows (page-level gold, 1-based pages) and doc_id -> DocMeta dict."""
    wanted = set(doc_ids)
    chosen = [q for q in questions if q["doc_name"] in wanted]
    splits = assign_splits([q["financebench_id"] for q in chosen], dev_ratio, seed)
    rows, docs = [], {}
    for q in sorted(chosen, key=lambda x: x["financebench_id"]):
        doc_id = q["doc_name"]
        pages = sorted(
            {e["evidence_page_num"] + 1 for e in q["evidence"] if e["doc_name"] == doc_id}
        )
        rows.append(
            {
                "id": q["financebench_id"],
                "question": q["question"],
                "gold_answer": q["answer"],
                "doc_id": doc_id,
                "gold_pages": pages,
                "split": splits[q["financebench_id"]],
            }
        )
        docs[doc_id] = {
            "doc_id": doc_id,
            "company": q["company"],
            "fiscal_year": int(_TENK.match(doc_id)["year"]),
            "doc_type": "10-K",
        }
    return rows, docs


def _download(url: str, dest: Path, timeout_s: int) -> None:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        dest.write_bytes(resp.read())


def main(config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)
    raw_dir, eval_dir = Path(cfg.data.raw_dir), Path(cfg.data.eval_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    qfile = eval_dir / "financebench_open_source.jsonl"
    if not qfile.exists():
        _download(cfg.data.questions_url, qfile, cfg.data.timeout_s)
    questions = [json.loads(l) for l in qfile.read_text(encoding="utf-8").splitlines() if l.strip()]

    doc_ids = select_docs(questions, cfg.eval.n_companies)
    for doc_id in doc_ids:
        dest = raw_dir / f"{doc_id}.pdf"
        if not dest.exists():
            logger.info("downloading %s", doc_id)
            _download(cfg.data.pdf_url.format(doc_id=doc_id), dest, cfg.data.timeout_s)

    rows, docs = build_eval_rows(questions, doc_ids, cfg.eval.dev_ratio, cfg.eval.seed)
    (eval_dir / "questions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (eval_dir / "docs.json").write_text(json.dumps(docs, indent=2), encoding="utf-8")
    per_split = collections.Counter(r["split"] for r in rows)
    summary = {"docs": len(docs), "questions": len(rows), **per_split}
    print(summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
