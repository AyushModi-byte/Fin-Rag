import json
import logging
from pathlib import Path

import pymupdf

from src.ingestion.loader import load_corpus
from src.ingestion.parsers import parse_pdf

FIXTURE = Path(__file__).parent / "fixtures" / "sample.pdf"


def test_parse_returns_one_page_per_pdf_page():
    pages = parse_pdf(FIXTURE, doc_id="sample")
    assert [p.page for p in pages] == [1, 2]
    assert all(p.doc_id == "sample" for p in pages)
    assert "Page one" in pages[0].markdown


def test_table_kept_as_markdown_rows():
    pages = parse_pdf(FIXTURE, doc_id="sample")
    assert "|Safety|3,000|2,900|" in pages[1].markdown.replace(" ", "")


def test_empty_page_skipped_not_raised(tmp_path, caplog):
    pdf = tmp_path / "blank_middle.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "first page text")
    doc.new_page()  # blank
    doc.new_page().insert_text((72, 72), "third page text")
    doc.save(pdf)
    with caplog.at_level(logging.WARNING):
        pages = parse_pdf(pdf, doc_id="d")
    assert [p.page for p in pages] == [1, 3]
    assert "skipp" in caplog.text.lower()


def test_load_corpus_writes_parsed_jsonl(tmp_path):
    raw, parsed = tmp_path / "raw", tmp_path / "parsed"
    raw.mkdir()
    (raw / "SAMPLE_2018_10K.pdf").write_bytes(FIXTURE.read_bytes())
    docs_json = tmp_path / "docs.json"
    docs_json.write_text(json.dumps({"SAMPLE_2018_10K": {
        "doc_id": "SAMPLE_2018_10K", "company": "Sample", "fiscal_year": 2018, "doc_type": "10-K"}}))
    corpus = load_corpus(raw, docs_json, parsed_dir=parsed)
    assert len(corpus) == 1
    meta, pages = corpus[0]
    assert meta.company == "Sample" and len(pages) == 2
    lines = (parsed / "SAMPLE_2018_10K.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["page"] for l in lines] == [1, 2]
