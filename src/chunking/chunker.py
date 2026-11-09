"""Parent/child chunking.

Parent = one page. Children = contiguous slices of the page markdown, packed up to
`cfg.chunking.child_tokens`. Markdown tables are atomic: never split, even if they exceed the
budget. Oversized prose is split into sliding windows with `cfg.chunking.overlap_tokens` overlap.

Token counting is a cheap regex approximation (words and punctuation); the embedder's real
tokenizer is not needed to keep chunks near the target size.
"""
import hashlib
import re
from dataclasses import dataclass

from src.schemas.models import Chunk, DocMeta, Page, Parent

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def _stable_id(*parts: object) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def embedding_text(chunk: Chunk) -> str:
    """Text that gets embedded: context prefix + chunk text."""
    return f"{chunk.context}\n{chunk.text}" if chunk.context else chunk.text


@dataclass
class _Block:
    start: int
    end: int
    is_table: bool
    tokens: int


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _blocks(md: str) -> list[_Block]:
    """Split page markdown into prose paragraphs and whole tables, with char offsets."""
    blocks: list[_Block] = []
    cur_start = cur_end = None
    cur_table = False

    def close():
        nonlocal cur_start, cur_end
        if cur_start is not None:
            blocks.append(
                _Block(cur_start, cur_end, cur_table, count_tokens(md[cur_start:cur_end]))
            )
        cur_start = cur_end = None

    pos = 0
    for line in md.splitlines(keepends=True):
        stripped = line.strip()
        line_start = pos + (len(line) - len(line.lstrip()))
        line_end = pos + len(line.rstrip())
        pos += len(line)
        if not stripped:
            close()
            continue
        table = _is_table_line(line)
        if cur_start is not None and table != cur_table:
            close()
        if cur_start is None:
            cur_start, cur_table = line_start, table
        cur_end = line_end
    close()
    return blocks


def _windows(md: str, block: _Block, size: int, overlap: int) -> list[tuple[int, int]]:
    """Sliding token windows over an oversized prose block, as (start, end) char offsets."""
    spans = [
        (block.start + m.start(), block.start + m.end())
        for m in _TOKEN.finditer(md[block.start : block.end])
    ]
    step = max(size - overlap, 1)
    out = []
    for s in range(0, len(spans), step):
        e = min(s + size, len(spans))
        out.append((spans[s][0], spans[e - 1][1]))
        if e == len(spans):
            break
    return out


def _child_spans(md: str, size: int, overlap: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    group: list[_Block] = []
    group_tokens = 0

    def flush():
        nonlocal group, group_tokens
        if group:
            spans.append((group[0].start, group[-1].end))
        group, group_tokens = [], 0

    for b in _blocks(md):
        if b.tokens > size:
            flush()
            if b.is_table:
                spans.append((b.start, b.end))  # tables are never split
            else:
                spans.extend(_windows(md, b, size, overlap))
            continue
        if group_tokens + b.tokens > size:
            flush()
        group.append(b)
        group_tokens += b.tokens
    flush()
    return spans


def chunk_pages(pages: list[Page], meta: DocMeta, cfg) -> tuple[list[Parent], list[Chunk]]:
    size, overlap = cfg.chunking.child_tokens, cfg.chunking.overlap_tokens
    parents: list[Parent] = []
    chunks: list[Chunk] = []
    for page in pages:
        parent = Parent(
            id=_stable_id(meta.doc_id, page.page, "parent"),
            doc_id=meta.doc_id,
            page_start=page.page,
            page_end=page.page,
            text=page.markdown,
        )
        parents.append(parent)
        context = (
            f"{meta.company} {meta.fiscal_year} {meta.doc_type}, page {page.page}"
            if cfg.chunking.context_prefix
            else ""
        )
        for start, end in _child_spans(page.markdown, size, overlap):
            chunks.append(
                Chunk(
                    id=_stable_id(meta.doc_id, page.page, start),
                    doc_id=meta.doc_id,
                    parent_id=parent.id,
                    text=page.markdown[start:end],
                    context=context,
                    page=page.page,
                    company=meta.company,
                    fiscal_year=meta.fiscal_year,
                )
            )
    return parents, chunks
