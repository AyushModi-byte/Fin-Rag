import pytest

from src.config import load_config
from src.generation.grounded_answer import generate
from src.pipelines.query_pipeline import QueryPipeline
from src.prompts.prompt_templates import IDK, SYSTEM_GROUNDED, system_prompt
from src.schemas.models import Chunk, Hit


def hit():
    c = Chunk(id="a", doc_id="ACME_2022_10K", parent_id="p", text="Net AR 1,616", context="ctx", page=5,
              company="Acme", fiscal_year=2022)
    return Hit(chunk_id="a", score=1.0, chunk=c)


class FakeLLM:
    def __init__(self):
        self.systems = []

    def complete(self, prompt, system=None, **kw):
        self.systems.append(system)
        return "ok [ACME_2022_10K p.5]"


def test_v1_is_the_original_prompt_unchanged():
    assert system_prompt("grounded_v1") == SYSTEM_GROUNDED


def test_v2_asks_for_formula_units_fiscal_year_and_restricts_idk():
    s = system_prompt("grounded_v2").lower()
    for needle in ("formula", "units", "fiscal year", "parentheses", "closest matching line item"):
        assert needle in s
    assert IDK in system_prompt("grounded_v2")
    assert "only if" in s  # "I don't know" only when a needed number is absent


def test_unknown_version_is_an_error():
    with pytest.raises(KeyError):
        system_prompt("grounded_v9")


def test_generate_passes_the_given_system_prompt():
    llm = FakeLLM()
    generate("q", [hit()], llm, system="CUSTOM")
    generate("q", [hit()], llm)
    assert llm.systems == ["CUSTOM", SYSTEM_GROUNDED]


def test_pipeline_uses_the_configured_prompt_version():
    class R:
        def retrieve(self, q, k, filters=None):
            return [hit()]

    for version in ("grounded_v1", "grounded_v2"):
        cfg = load_config()
        cfg.retrieval.use_router_filter = False
        cfg.rerank.enabled = False
        cfg.generation.prompt_version = version
        llm = FakeLLM()
        QueryPipeline(cfg, R(), None, llm, ["Acme"]).run("q")
        assert llm.systems == [system_prompt(version)]
