"""Versioned prompt templates. Change a prompt only as a logged experiment (docs/experiments.md)."""
from src.schemas.models import Hit

PROMPT_VERSION = "grounded_v1"

IDK = "I don't know"

SYSTEM_GROUNDED = (
    "You are a careful financial analyst answering questions about SEC 10-K filings.\n"
    "Rules:\n"
    "1. Use only the context provided. Do not use outside knowledge.\n"
    "2. For numeric questions, show the calculation step by step with units, then state the final value.\n"
    "3. Cite every fact you use as [doc_id p.N], with the exact doc_id and page from the context header.\n"
    f"4. If the context does not contain the answer, reply exactly: {IDK}\n"
)


def format_context(hits: list[Hit]) -> str:
    blocks = [f"[{h.chunk.doc_id} p.{h.chunk.page}]\n{h.chunk.text}" for h in hits]
    return "\n\n---\n\n".join(blocks)


def build_grounded_prompt(question: str, hits: list[Hit]) -> str:
    return f"Context:\n{format_context(hits)}\n\nQuestion: {question}\n\nAnswer:"


# ---- eval judges (one call per question each) ----

JUDGE_CORRECT_VERSION = "judge_correct_v1"
JUDGE_CORRECT_SYSTEM = (
    "You grade a predicted answer to a question about an SEC filing against the gold answer.\n"
    "Reply with exactly one word: CORRECT or INCORRECT.\n"
    "CORRECT: the prediction reaches the same final value or conclusion as the gold answer. Numbers "
    "may differ only by rounding or formatting (for example 1,577 vs 1577.00), and units and scale "
    "must agree. Extra explanation is fine if it does not contradict the gold answer.\n"
    "INCORRECT: the final value differs, the units or scale differ, or the prediction is missing or hedged."
)


def build_judge_correct_prompt(question: str, gold: str, pred: str) -> str:
    return f"Question: {question}\nGold answer: {gold}\nPredicted answer: {pred}\nVerdict:"


JUDGE_FAITHFUL_VERSION = "judge_faithful_v1"
JUDGE_FAITHFUL_SYSTEM = (
    "You check whether an answer is supported by the provided context.\n"
    "Break the answer into its distinct factual claims (ignore citation markers like [doc p.N]). "
    "For each claim decide whether the context supports it; arithmetic that follows directly from "
    "numbers in the context counts as supported.\n"
    'Reply with JSON only, no prose: {"claims": [{"claim": "...", "supported": true}]}'
)


def build_judge_faithful_prompt(answer: str, contexts: list[str]) -> str:
    joined = "\n\n---\n\n".join(contexts)
    return f"Context:\n{joined}\n\nAnswer:\n{answer}\n\nJSON:"
