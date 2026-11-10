from types import SimpleNamespace

from src.config import load_config
from src.generation.context_expansion import expand_context
from src.pipelines.query_pipeline import QueryPipeline
from src.schemas.models import Chunk, Hit, Parent

TABLE_T1 = "|H|2022|2021|\n|---|---|---|\n" + "\n".join(f"|row {i}|{i}|{i + 1}|" for i in range(30))
TABLE_T2 = "|H|2022|\n|---|---|\n|other row|5|"


def hit(cid, text="frag", page=5, table_id=None, row_group=None, doc="ACME_2022_10K"):
    c = Chunk(id=cid, doc_id=doc, parent_id=table_id or "page", text=text, context="ctx", page=page,
              company="Acme", fiscal_year=2022, table_id=table_id, row_group=row_group)
    return Hit(chunk_id=cid, score=1.0, chunk=c)


def parents():
    return {
        "T1": Parent(id="T1", doc_id="ACME_2022_10K", page_start=5, page_end=5, text=TABLE_T1),
        "T2": Parent(id="T2", doc_id="ACME_2022_10K", page_start=7, page_end=7, text=TABLE_T2),
    }


def texts(hits):
    return [h.chunk.text for h in hits]


def test_row_group_hit_is_replaced_by_the_full_table():
    out = expand_context([hit("a", "frag", table_id="T1", row_group=1)], parents(), max_tokens=10_000)
    assert texts(out) == [TABLE_T1]
    assert (out[0].chunk.doc_id, out[0].chunk.page, out[0].chunk.id) == ("ACME_2022_10K", 5, "a")


def test_two_row_groups_of_one_table_are_deduplicated_in_first_position():
    hits = [hit("a", "f1", table_id="T1", row_group=0), hit("b", "prose"),
            hit("c", "f3", table_id="T1", row_group=2)]
    out = expand_context(hits, parents(), max_tokens=10_000)
    assert texts(out) == [TABLE_T1, "prose"]


def test_hits_without_a_split_table_parent_are_untouched():
    whole = hit("w", "whole table text", table_id="NOT_A_PARENT")  # whole table chunk: nothing to expand
    prose = hit("p", "prose")
    assert texts(expand_context([whole, prose], parents(), max_tokens=10_000)) == ["whole table text", "prose"]


def test_cap_keeps_the_fragment_when_the_table_does_not_fit():
    h1 = hit("a", "frag1", table_id="T1", row_group=0)
    h2 = hit("b", "frag2", page=7, table_id="T2", row_group=0)
    out = expand_context([h1, h2], parents(), max_tokens=60)  # T1 is far over 60; T2 fits
    assert texts(out) == ["frag1", TABLE_T2]


def test_cap_counts_what_is_already_in_the_context():
    h1 = hit("a", "x " * 50)
    h2 = hit("b", "frag2", page=7, table_id="T2", row_group=0)
    out = expand_context([h1, h2], parents(), max_tokens=60)  # 50 tokens used, T2 would push over
    assert texts(out) == ["x " * 50, "frag2"]


def test_no_parents_returns_hits_unchanged():
    hits = [hit("a", "frag", table_id="T1", row_group=0)]
    assert expand_context(hits, {}, max_tokens=10_000) == hits


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, q, k, filters=None):
        return list(self.hits)


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete(self, prompt, system=None, **kw):
        self.calls.append(prompt)
        return "ok [ACME_2022_10K p.5]"


def _pipe(expand, parent_map):
    cfg = load_config()
    cfg.retrieval.use_router_filter = False
    cfg.rerank.enabled = False
    cfg.generation.expand_tables = expand
    llm = FakeLLM()
    p = QueryPipeline(cfg, FakeRetriever([hit("a", "FRAGMENT", table_id="T1", row_group=1)]), None, llm,
                      ["Acme"], parents=parent_map)
    return p, llm


def test_pipeline_gives_the_llm_the_full_table_when_enabled():
    p, llm = _pipe(True, parents())
    out = p.run("what is row 3?")
    assert "row 29" in llm.calls[0] and "FRAGMENT" not in llm.calls[0]
    assert TABLE_T1 in out.result.contexts  # faithfulness judge sees what the LLM saw
    assert out.ranked[0].chunk.text == "FRAGMENT"  # retrieval ranking (metrics) is untouched


def test_pipeline_keeps_fragments_when_disabled():
    p, llm = _pipe(False, parents())
    p.run("what is row 3?")
    assert "FRAGMENT" in llm.calls[0] and "row 29" not in llm.calls[0]
