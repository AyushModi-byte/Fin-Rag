from types import SimpleNamespace

from src.chunking.chunker import chunk_pages, count_tokens, embedding_text
from src.schemas.models import DocMeta, Page

META = DocMeta(doc_id="ACME_2020_10K", company="Acme", fiscal_year=2020, doc_type="10-K")


def cfg(child_tokens=40, overlap_tokens=5, context_prefix=True, max_embed_tokens=400, min_tokens=3):
    return SimpleNamespace(
        chunking=SimpleNamespace(
            child_tokens=child_tokens, overlap_tokens=overlap_tokens, context_prefix=context_prefix,
            max_embed_tokens=max_embed_tokens, min_tokens=min_tokens, max_header_rows=3,
            max_heading_chars=80,
        )
    )


TABLE = "\n".join(
    ["|Segment|2020|2019|", "|---|---|---|"]
    + [f"|Product line number {i}|{1000 + i}|{900 + i}|" for i in range(12)]
)


def _pages():
    prose = " ".join(f"word{i}" for i in range(30))
    return [
        Page(doc_id=META.doc_id, page=1, markdown=f"{prose}\n\n{prose}"),
        Page(doc_id=META.doc_id, page=2, markdown=f"Intro paragraph here.\n\n{TABLE}\n\nClosing remarks."),
    ]


def test_table_not_split():
    c = cfg(child_tokens=40)
    assert count_tokens(TABLE) > c.chunking.child_tokens  # table alone exceeds the budget
    _, chunks = chunk_pages(_pages(), META, c)
    holders = [ch for ch in chunks if "Product line number" in ch.text]
    assert len(holders) == 1
    assert TABLE in holders[0].text


def test_every_chunk_has_parent_and_page():
    parents, chunks = chunk_pages(_pages(), META, cfg())
    by_id = {p.id: p for p in parents}
    assert chunks
    for ch in chunks:
        parent = by_id[ch.parent_id]
        assert parent.doc_id == ch.doc_id
        assert parent.page_start <= ch.page <= parent.page_end
        assert ch.company == "Acme" and ch.fiscal_year == 2020


def test_parent_is_page_level_span():
    parents, _ = chunk_pages(_pages(), META, cfg())
    assert [(p.page_start, p.page_end) for p in parents] == [(1, 1), (2, 2)]
    assert parents[1].text == _pages()[1].markdown


def test_context_prefix_contains_company_and_year():
    _, chunks = chunk_pages(_pages(), META, cfg())
    ch = next(c for c in chunks if c.page == 2)
    assert ch.context == "Acme FY2020 10-K | page 2"
    assert embedding_text(ch) == "Acme FY2020 10-K | page 2\n" + ch.text


def test_context_prefix_can_be_disabled():
    _, chunks = chunk_pages(_pages(), META, cfg(context_prefix=False))
    assert all(c.context == "" for c in chunks)
    assert embedding_text(chunks[0]) == chunks[0].text


def test_chunk_ids_unique_and_stable():
    _, a = chunk_pages(_pages(), META, cfg())
    _, b = chunk_pages(_pages(), META, cfg())
    ids = [c.id for c in a]
    assert len(ids) == len(set(ids))
    assert ids == [c.id for c in b]


def test_prose_chunks_respect_token_budget_and_overlap():
    words = " ".join(f"w{i}" for i in range(100))
    page = Page(doc_id=META.doc_id, page=1, markdown=words)
    c = cfg(child_tokens=30, overlap_tokens=5)
    _, chunks = chunk_pages([page], META, c)
    assert len(chunks) > 1
    assert all(count_tokens(ch.text) <= 30 for ch in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-5:] == second[:5]  # overlap carried between windows
    assert {w for ch in chunks for w in ch.text.split()} == set(words.split())  # nothing lost


def test_small_blocks_are_packed_into_one_chunk():
    page = Page(doc_id=META.doc_id, page=1, markdown="Short one.\n\nShort two.\n\nShort three.")
    _, chunks = chunk_pages([page], META, cfg(child_tokens=40))
    assert len(chunks) == 1
    assert "Short one." in chunks[0].text and "Short three." in chunks[0].text
