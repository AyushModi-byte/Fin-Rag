from types import SimpleNamespace

import pytest

from src.chunking.chunker import RegexTokenizer, chunk_pages, embedding_text
from src.schemas.models import DocMeta, Page

META = DocMeta(doc_id="ADBE_2022_10K", company="Adobe", fiscal_year=2022, doc_type="10-K")
TOK = RegexTokenizer()


def cfg(child_tokens=100, overlap_tokens=8, max_embed_tokens=200, min_tokens=5, prefix=True):
    return SimpleNamespace(
        chunking=SimpleNamespace(
            child_tokens=child_tokens, overlap_tokens=overlap_tokens,
            max_embed_tokens=max_embed_tokens, min_tokens=min_tokens, context_prefix=prefix,
            max_header_rows=3, max_heading_chars=80,
        )
    )


HEADER = [
    "|||**Years Ended**||",
    "|---|---|---|---|",
    "||**December 2,**<br>**2022**|**December 3,**<br>**2021**|**November 27,**<br>**2020**|",
]
ROWS = [f"|Line item number {i}|{1000 + i}|{900 + i}|{800 + i}|" for i in range(40)]
TABLE = "\n".join(HEADER + ROWS)

STATEMENT_PAGE = (
    "<u>Table of Contents</u>\n\n#### **ADOBE INC.**\n\n"
    "#### **CONSOLIDATED STATEMENTS OF INCOME**\n\n**(In millions, except per share data)**\n\n"
    + TABLE
    + "\n\n_See accompanying Notes._\n\n54"
)


def _chunks(pages, c=None):
    return chunk_pages(pages, META, c or cfg(), tok=TOK)


def _page(md, n=54):
    return Page(doc_id=META.doc_id, page=n, markdown=md)


def _full(ch):
    return embedding_text(ch)


def test_no_chunk_exceeds_the_embedding_cap():
    prose = " ".join(f"word{i}" for i in range(400))
    pages = [_page(STATEMENT_PAGE, 54), _page(prose, 55), _page(prose + "\n\n" + TABLE, 56)]
    c = cfg()
    _, chunks = _chunks(pages, c)
    assert chunks
    assert all(TOK.count(_full(ch)) <= c.chunking.max_embed_tokens for ch in chunks)


def test_large_table_split_into_row_groups_with_repeated_header():
    _, chunks = _chunks([_page(STATEMENT_PAGE)])
    groups = [ch for ch in chunks if ch.row_group is not None]
    assert len(groups) > 1
    assert [g.row_group for g in groups] == list(range(len(groups)))
    for g in groups:
        lines = g.text.split("\n")
        assert lines[:3] == HEADER  # both header rows repeated, continuation row included
    body = [ln for g in groups for ln in g.text.split("\n")[3:]]
    assert body == ROWS  # every row once, in order, none split


def test_row_group_chunks_point_at_the_full_table_parent():
    parents, chunks = _chunks([_page(STATEMENT_PAGE)])
    groups = [ch for ch in chunks if ch.row_group is not None]
    by_id = {p.id: p for p in parents}
    table_ids = {g.table_id for g in groups}
    assert len(table_ids) == 1
    parent = by_id[table_ids.pop()]
    assert parent.text == TABLE
    assert all(g.parent_id == parent.id for g in groups)
    assert parent.page_start == parent.page_end == 54


def test_table_that_fits_stays_whole_and_has_table_id():
    small = "\n".join(HEADER + ROWS[:3])
    _, chunks = _chunks([_page(f"Intro text here for the page.\n\n{small}")], cfg(child_tokens=20))
    holders = [c for c in chunks if "Line item number 0" in c.text]
    assert len(holders) == 1 and holders[0].row_group is None
    assert small in holders[0].text and holders[0].table_id


def test_prefix_has_section_and_units():
    _, chunks = _chunks([_page(STATEMENT_PAGE)])
    for ch in (c for c in chunks if c.row_group is not None):
        assert ch.context == (
            "Adobe FY2022 10-K | page 54 | CONSOLIDATED STATEMENTS OF INCOME | USD millions"
        )


def test_heading_carries_over_from_previous_page():
    page2 = _page("The following discussion covers segment results in some detail here.", 55)
    _, chunks = _chunks([_page(STATEMENT_PAGE), page2])
    ch = next(c for c in chunks if c.page == 55)
    assert "CONSOLIDATED STATEMENTS OF INCOME" in ch.context


@pytest.mark.parametrize(
    "line,expected",
    [
        ("(In thousands)", "USD thousands"),
        ("($ in millions)", "USD millions"),
        ("Dollars in billions", "USD billions"),
        ("(Millions of Dollars)", "USD millions"),
        ("$ millions", "USD millions"),
    ],
)
def test_units_patterns(line, expected):
    page = _page(f"## Balance Sheet\n\n{line}\n\n{TABLE}")
    _, chunks = _chunks([page])
    assert chunks[0].context.endswith(expected)


def test_tiny_page_is_dropped():
    parents, chunks = _chunks([_page("54"), _page("ii", 3)])
    assert chunks == []
    assert len(parents) == 2  # the pages still exist as parents


def test_ids_deterministic_across_runs():
    pages = [_page(STATEMENT_PAGE), _page(" ".join(f"w{i}" for i in range(300)), 55)]
    _, a = _chunks(pages)
    _, b = _chunks(pages)
    ids = [c.id for c in a]
    assert ids == [c.id for c in b]
    assert len(ids) == len(set(ids))


def test_row_wider_than_the_cap_is_windowed_and_still_fits():
    wide = "|" + " ".join(f"tok{i}" for i in range(300)) + "|1|"
    table = "\n".join(HEADER[:2] + [ROWS[0], wide, ROWS[1]])
    c = cfg()
    _, chunks = _chunks([_page(table)], c)
    assert chunks
    assert all(TOK.count(_full(ch)) <= c.chunking.max_embed_tokens for ch in chunks)


def test_embedder_tokenizer_respects_cap_on_a_real_statement():
    from src.embeddings.tokenizer import EmbedTokenizer

    tok = EmbedTokenizer("BAAI/bge-small-en-v1.5")
    big = "\n".join(HEADER + [f"|Revenue category {i}|$ {10000 + i:,}|{9000 + i:,}|{8000 + i:,}|" for i in range(120)])
    page = _page(f"#### **CONSOLIDATED BALANCE SHEETS**\n\n(In millions)\n\n{big}", 58)
    c = cfg(child_tokens=300, overlap_tokens=40, max_embed_tokens=480, min_tokens=15)
    parents, chunks = chunk_pages([page], META, c, tok=tok)
    assert len([ch for ch in chunks if ch.row_group is not None]) >= 2
    assert all(tok.count(_full(ch)) <= 480 for ch in chunks)


def test_table_group_tokens_makes_smaller_row_groups_without_losing_rows():
    big = cfg(child_tokens=100, max_embed_tokens=200)
    small = cfg(child_tokens=100, max_embed_tokens=200)
    small.chunking.table_group_tokens = 185
    _, a = _chunks([_page(STATEMENT_PAGE)], big)
    _, b = _chunks([_page(STATEMENT_PAGE)], small)
    ga = [c for c in a if c.row_group is not None]
    gb = [c for c in b if c.row_group is not None]
    assert len(gb) > len(ga)
    assert [ln for g in gb for ln in g.text.split("\n")[3:]] == ROWS
    assert all(TOK.count(_full(c)) <= 185 for c in gb)  # target respected (prefix incl.)
