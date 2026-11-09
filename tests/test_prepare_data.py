import json
from pathlib import Path

from src.config import load_config
from src.jobs.prepare_data import assign_splits, build_eval_rows, select_docs

ROOT = Path(__file__).resolve().parent.parent


def _q(i, company, doc, pages=(0,), ev_doc=None):
    return {
        "financebench_id": f"q{i}",
        "company": company,
        "doc_name": doc,
        "question": f"question {i}",
        "answer": f"answer {i}",
        "evidence": [{"doc_name": ev_doc or doc, "evidence_page_num": p} for p in pages],
    }


def test_select_docs_picks_top_companies_10k_only():
    qs = [_q(i, "A", "A_2020_10K") for i in range(3)]
    qs += [_q(10 + i, "B", "B_2019_10K") for i in range(2)]
    qs += [_q(20, "B", "B_2023Q2_10Q"), _q(21, "B", "B_2023Q2_10Q"), _q(22, "B", "B_2023Q2_10Q")]
    qs += [_q(30, "C", "C_2021_10K")]
    assert select_docs(qs, n_companies=2) == ["A_2020_10K", "B_2019_10K"]


def test_split_is_deterministic_and_disjoint():
    ids = [f"q{i}" for i in range(60)]
    a = assign_splits(ids, dev_ratio=0.5, seed=42)
    b = assign_splits(list(reversed(ids)), dev_ratio=0.5, seed=42)
    assert a == b
    dev = {i for i, s in a.items() if s == "dev"}
    test = {i for i, s in a.items() if s == "test"}
    assert not dev & test
    assert dev | test == set(ids)
    assert len(dev) == 30


def test_build_rows_uses_one_based_gold_pages_for_own_doc_only():
    qs = [_q(1, "A", "A_2020_10K", pages=(4, 9), ev_doc="A_2020_10K")]
    qs[0]["evidence"].append({"doc_name": "OTHER", "evidence_page_num": 1})
    rows, docs = build_eval_rows(qs, ["A_2020_10K"], dev_ratio=0.5, seed=42)
    assert rows[0]["gold_pages"] == [5, 10]
    assert rows[0]["doc_id"] == "A_2020_10K"
    assert rows[0]["gold_answer"] == "answer 1"
    assert docs["A_2020_10K"] == {
        "doc_id": "A_2020_10K", "company": "A", "fiscal_year": 2020, "doc_type": "10-K",
    }


def _rows():
    path = ROOT / "data/eval/questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_every_eval_doc_exists_on_disk():
    cfg = load_config()
    docs = json.loads((ROOT / cfg.data.eval_dir / "docs.json").read_text(encoding="utf-8"))
    assert docs
    for doc_id in docs:
        assert (ROOT / cfg.data.raw_dir / f"{doc_id}.pdf").exists(), doc_id
    assert {r["doc_id"] for r in _rows()} <= set(docs)


def test_real_split_is_deterministic_and_disjoint():
    cfg = load_config()
    rows = _rows()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    expected = assign_splits(ids, cfg.eval.dev_ratio, cfg.eval.seed)
    assert {r["id"]: r["split"] for r in rows} == expected
    assert {r["split"] for r in rows} == {"dev", "test"}


def test_min_questions():
    cfg = load_config()
    assert len(_rows()) >= cfg.eval.min_questions
