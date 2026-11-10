from src.lexical.bm25_index import BM25Index
from src.lexical.tokenize import tokenize
from src.schemas.models import Chunk


def test_stemming_joins_inflections():
    assert tokenize("Revenues increased") == tokenize("revenue increases")


def test_identifiers_and_numbers_stay_searchable():
    toks = tokenize("3M reported 10-K figures of $1,234.5 and 62% growth (1,577)")
    for expected in ("3m", "10-k", "10", "1,234.5", "1234.5", "62%", "62", "1,577", "1577"):
        assert expected in toks


def test_number_matches_with_or_without_commas():
    assert set(tokenize("16,388")) & set(tokenize("16388"))


def test_stopwords_removed_and_stemming_can_be_disabled():
    assert "the" not in tokenize("the margin")
    assert "margins" in tokenize("margins", stem=False)


def _chunk(i, text):
    return Chunk(id=f"c{i}", doc_id="d", parent_id="p", text=text, context="Acme FY2020 10-K | page 1",
                 page=1, company="Acme", fiscal_year=2020)


def test_bm25_v2_finds_stemmed_and_comma_free_number(tmp_path):
    idx = BM25Index("en", "v2", True)
    idx.build([_chunk(0, "Net revenues were 16,388 million."), _chunk(1, "Unrelated prose about weather.")])
    assert idx.search("revenue 16388", 3)[0].chunk_id == "c0"
    idx.save(tmp_path)
    loaded = BM25Index.load(tmp_path, "en", "v2", True)
    assert loaded.search("revenue 16388", 3)[0].chunk_id == "c0"
