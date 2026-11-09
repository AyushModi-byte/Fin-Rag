import pytest
from pydantic import ValidationError

from src.config import load_config
from src.schemas.models import Chunk


def test_load_config_has_required_keys():
    cfg = load_config()
    assert cfg.embeddings.model
    assert cfg.retrieval.top_k == 20
    assert cfg.rerank.top_n == 5
    assert cfg.index.version


def test_chunk_requires_page():
    with pytest.raises(ValidationError):
        Chunk(
            id="c1", doc_id="d1", parent_id="p1", text="t",
            context="ctx", company="3M", fiscal_year=2018,
        )
