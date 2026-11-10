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
