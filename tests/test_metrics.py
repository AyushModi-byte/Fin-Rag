import pytest

from src.eval.rag_metrics import judge_correct, judge_faithful, mrr, recall_at_k
from src.schemas.models import Chunk, Hit


def make_hit(cid, doc_id="D", page=1):
    chunk = Chunk(
        id=cid, doc_id=doc_id, parent_id="p", text=f"text {cid}", context="ctx",
        page=page, company="C", fiscal_year=2020,
    )
    return Hit(chunk_id=cid, score=1.0, chunk=chunk)


HITS = [make_hit("a", page=3), make_hit("b", page=7), make_hit("c", page=9)]


# ---------- retrieval metrics ----------

def test_recall_at_k_hit_and_miss():
    assert recall_at_k(HITS, gold_pages=[7], k=5) == 1.0
    assert recall_at_k(HITS, gold_pages=[100], k=5) == 0.0


def test_recall_at_k_only_counts_top_k():
    assert recall_at_k(HITS, gold_pages=[9], k=2) == 0.0
    assert recall_at_k(HITS, gold_pages=[9], k=3) == 1.0


def test_recall_requires_the_right_doc_when_given():
    assert recall_at_k(HITS, gold_pages=[3], k=5, doc_id="OTHER") == 0.0
    assert recall_at_k(HITS, gold_pages=[3], k=5, doc_id="D") == 1.0


def test_mrr_rank_two_is_half():
    assert mrr(HITS, gold_pages=[7]) == 0.5


def test_mrr_first_gold_hit_wins_and_miss_is_zero():
    assert mrr(HITS, gold_pages=[3, 9]) == 1.0
    assert mrr(HITS, gold_pages=[100]) == 0.0
    assert mrr([], gold_pages=[1]) == 0.0


def test_mrr_respects_doc_id():
    hits = [make_hit("x", doc_id="OTHER", page=7), make_hit("y", doc_id="D", page=7)]
    assert mrr(hits, gold_pages=[7], doc_id="D") == 0.5


# ---------- judges (fake LLM; no real calls) ----------

class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete(self, prompt, system=None, **kwargs):
        self.calls.append({"prompt": prompt, "system": system})
        return self.reply


@pytest.mark.parametrize(
    "reply, expected",
    [("CORRECT", True), ("correct.", True), ("CORRECT - same value", True),
     ("INCORRECT", False), ("Incorrect: units differ", False), ("unsure", False)],
)
def test_judge_correct_parses_verdict(reply, expected):
    llm = FakeLLM(reply)
    assert judge_correct("q", "$1577.00", "1,577 million", llm) is expected
    assert len(llm.calls) == 1  # exactly one call per question


def test_judge_correct_prompt_has_question_gold_and_prediction():
    llm = FakeLLM("CORRECT")
    judge_correct("What was capex?", "GOLD-ANSWER", "PREDICTED-ANSWER", llm)
    prompt = llm.calls[0]["prompt"]
    assert "What was capex?" in prompt and "GOLD-ANSWER" in prompt and "PREDICTED-ANSWER" in prompt


def test_judge_correct_idk_is_incorrect_without_spending_a_call():
    llm = FakeLLM("CORRECT")
    assert judge_correct("q", "gold", "I don't know", llm) is False
    assert llm.calls == []


def test_judge_faithful_score_is_supported_over_total():
    reply = '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}, ' \
            '{"claim": "c", "supported": true}, {"claim": "d", "supported": true}]}'
    llm = FakeLLM(reply)
    assert judge_faithful("answer text", ["ctx one", "ctx two"], llm) == pytest.approx(0.75)
    assert len(llm.calls) == 1
    assert "ctx one" in llm.calls[0]["prompt"] and "answer text" in llm.calls[0]["prompt"]


def test_judge_faithful_accepts_code_fenced_json():
    llm = FakeLLM('```json\n{"claims": [{"claim": "a", "supported": true}]}\n```')
    assert judge_faithful("ans", ["ctx"], llm) == 1.0


def test_judge_faithful_none_when_nothing_to_judge():
    assert judge_faithful("I don't know", ["ctx"], FakeLLM("{}")) is None
    assert judge_faithful("answer", [], FakeLLM("{}")) is None
    assert judge_faithful("answer", ["ctx"], FakeLLM('{"claims": []}')) is None


def test_judge_faithful_unparseable_reply_raises_not_guesses():
    with pytest.raises(ValueError):
        judge_faithful("answer", ["ctx"], FakeLLM("sorry, I cannot do that"))
