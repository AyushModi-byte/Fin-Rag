"""Parent/child chunking, v2.

Parent = one page (plus, for split tables, the whole table). Children = contiguous slices of the
page markdown packed up to `chunking.child_tokens`, measured with the EMBEDDER'S tokenizer and
counting the context prefix too, so no chunk ever exceeds `chunking.max_embed_tokens` (the
embedder would silently truncate it).

* Markdown tables that fit stay whole. A table over the cap is split into ROW GROUPS: a row is never
  split, and the header row(s) are repeated in every group. Each group chunk has the whole table as
  its parent (`parent_id == table_id`) so generation can expand to it.
* Every chunk gets a deterministic context prefix:
  `{company} FY{year} 10-K | page {n} | {nearest preceding heading} | {units}`.
* Chunks under `chunking.min_tokens` (page numbers, "ii") are not indexed.
* Ids are a hash of (doc, page, start offset [, row group]) so rebuilds are stable.

`tok` is any object with `count(text)`, `offsets(text)` and `special_overhead`; the real pipeline
passes `EmbedTokenizer`, tests may use the regex stand-in `RegexTokenizer`.
"""
import hashlib
import math
import re

from src.schemas.models import Chunk, DocMeta, Page, Parent

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Cheap regex token count (retrieval-time tiny-chunk filter, offline tests)."""
    return len(_TOKEN.findall(text))


class RegexTokenizer:
    """Offline stand-in with the same interface as EmbedTokenizer."""

    special_overhead = 0

    def count(self, text: str) -> int:
        return len(_TOKEN.findall(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in _TOKEN.finditer(text)]


def _stable_id(*parts: object) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def embedding_text(chunk: Chunk) -> str:
    """Text that gets embedded: context prefix + chunk text."""
    return f"{chunk.context}\n{chunk.text}" if chunk.context else chunk.text


# --------------------------------------------------------------------------- headings and units

_MD_HEADING = re.compile(r"^\s*#{1,6}\s*(.+?)\s*#*\s*$")
_BOLD_LINE = re.compile(r"^\s*(?:\*\*|__)(.+?)(?:\*\*|__)\s*$")
_ALL_CAPS = re.compile(r"^[A-Z][A-Z0-9 ,.&'()/:\-]{5,}$")
_ITEM_NOTE = re.compile(r"^(?:item\s+\d+[a-c]?\b|note\s+\d+\b)", re.I)
_NOT_HEADING = re.compile(r"^(?:table of contents|\d+|[ivxlc]+|\(.*\))$", re.I)
_UNITS = re.compile(
    r"(?:\bin|\$|\busd|dollars\s+in|\()\s*\$?\s*(millions|thousands|billions)\b"
    r"|\b(millions|thousands|billions)\s+of\s+(?:u\.?s\.?\s+)?dollars",
    re.I,
)


def _clean_heading(raw: str) -> str:
    text = re.sub(r"</?u>|[*_#`]", "", raw)
    return re.sub(r"\s+", " ", text).strip(" .:|")


def _headings(md: str, max_chars: int) -> list[tuple[int, str]]:
    """(offset, heading text) for markdown headings, bold-only lines, ALL-CAPS lines and
    'Item N' / 'Note N' lines. Table rows and units lines are never headings."""
    out: list[tuple[int, str]] = []
    pos = 0
    for line in md.splitlines(keepends=True):
        start = pos + (len(line) - len(line.lstrip()))
        pos += len(line)
        stripped = line.strip()
        if not stripped or stripped.startswith("|"):
            continue
        m = _MD_HEADING.match(line) or _BOLD_LINE.match(line)
        raw = m.group(1) if m else stripped
        text = _clean_heading(raw)
        if not m and not (_ALL_CAPS.match(text) or _ITEM_NOTE.match(text)):
            continue
        if (
            len(text) < 3
            or len(text) > 2 * max_chars
            or _NOT_HEADING.match(text)
            or _UNITS.search(text)
            or not re.search(r"[A-Za-z]{3}", text)
        ):
            continue
        out.append((start, text[:max_chars].rstrip()))
    return out


def _units(md: str) -> list[tuple[int, str]]:
    """(offset, 'USD millions') for 'in millions', '$ millions', '(Millions of Dollars)' ..."""
    out: list[tuple[int, str]] = []
    pos = 0
    for line in md.splitlines(keepends=True):
        start = pos
        pos += len(line)
        if len(line) > 200:
            continue
        m = _UNITS.search(line)
        if m:
            unit = (m.group(1) or m.group(2)).lower()
            out.append((start, f"USD {unit}"))
    return out


# --------------------------------------------------------------------------- blocks

class _Block:
    __slots__ = ("start", "end", "is_table", "tokens")

    def __init__(self, start: int, end: int, is_table: bool, tokens: int):
        self.start, self.end, self.is_table, self.tokens = start, end, is_table, tokens


def _blocks(md: str, tok) -> list[_Block]:
    """Split page markdown into prose paragraphs and whole tables, with char offsets."""
    spans: list[tuple[int, int, bool]] = []
    cur_start = cur_end = None
    cur_table = False
    pos = 0
    for line in md.splitlines(keepends=True):
        stripped = line.strip()
        line_start = pos + (len(line) - len(line.lstrip()))
        line_end = pos + len(line.rstrip())
        pos += len(line)
        if not stripped:
            if cur_start is not None:
                spans.append((cur_start, cur_end, cur_table))
            cur_start = cur_end = None
            continue
        table = stripped.startswith("|")
        if cur_start is not None and table != cur_table:
            spans.append((cur_start, cur_end, cur_table))
            cur_start = None
        if cur_start is None:
            cur_start, cur_table = line_start, table
        cur_end = line_end
    if cur_start is not None:
        spans.append((cur_start, cur_end, cur_table))
    return [
        _Block(s, e, t, max(tok.count(md[s:e]) - tok.special_overhead, 0)) for s, e, t in spans
    ]


_SEPARATOR_ROW = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _is_header_continuation(line: str) -> bool:
    """A row right after the separator that still belongs to the header: first cell empty and
    every non-empty cell bold (e.g. `||**December 2,**<br>**2022**|...`)."""
    parts = line.strip().split("|")
    cells = [c.strip() for c in parts[1:-1]]  # drop only the outer pipes, keep empty first cell
    if not cells or cells[0] != "":
        return False
    filled = [c for c in cells[1:] if c]
    return bool(filled) and all(c.startswith("**") for c in filled)


def _header_rows(lines: list[str], max_rows: int) -> int:
    """Number of leading lines that form the table header (0 when no `|---|` separator)."""
    if len(lines) < 2 or not _SEPARATOR_ROW.match(lines[1]):
        return 0
    n = 2
    while n < min(len(lines), max_rows) and _is_header_continuation(lines[n]):
        n += 1
    return n


# --------------------------------------------------------------------------- the chunker

class _PageChunker:
    def __init__(self, md: str, page: Page, meta: DocMeta, cfg, tok, carry_heading: str):
        self.md, self.page, self.meta, self.tok = md, page, meta, tok
        self.c = cfg.chunking
        self.overhead = tok.special_overhead
        self.headings = _headings(md, self.c.max_heading_chars)
        self.units = _units(md)
        self.carry_heading = carry_heading
        self.chunks: list[Chunk] = []
        self.table_parents: list[Parent] = []
        longest_h = max([h for _, h in self.headings] + [carry_heading], key=len)
        longest_u = max([u for _, u in self.units] + [""], key=len)
        self.page_ptoks = tok.count(self._join_prefix(longest_h, longest_u)) if self.c.context_prefix else 0
        floor = self.c.child_tokens // 2
        self.child_budget = max(self.c.child_tokens - self.page_ptoks, floor)
        self.cap_budget = self.c.max_embed_tokens - self.page_ptoks

    # ---- prefix
    def _join_prefix(self, heading: str, units: str) -> str:
        m = self.meta
        parts = [f"{m.company} FY{m.fiscal_year} {m.doc_type}", f"page {self.page.page}"]
        if heading:
            parts.append(heading)
        if units:
            parts.append(units)
        return " | ".join(parts)

    def prefix_at(self, offset: int, end: int | None = None) -> str:
        """Prefix for a chunk starting at `offset`. Units default to the nearest one before the
        chunk; if none, the first one inside the chunk (`end`), e.g. a heading + units + table."""
        if not self.c.context_prefix:
            return ""
        heading = self.carry_heading
        for off, text in self.headings:
            if off <= offset:
                heading = text
        units = ""
        for off, text in self.units:
            if off <= offset:
                units = text
        if not units and end is not None:
            units = next((t for off, t in self.units if offset <= off < end), "")
        return self._join_prefix(heading, units)

    def last_heading(self) -> str:
        return self.headings[-1][1] if self.headings else self.carry_heading

    # ---- emit
    def _fits(self, prefix: str, text: str) -> bool:
        full = f"{prefix}\n{text}" if prefix else text
        return self.tok.count(full) <= self.c.max_embed_tokens

    def _add(self, cid: str, start: int, text: str, parent_id: str, **extra) -> None:
        if self.tok.count(text) - self.overhead < self.c.min_tokens:
            return  # page numbers, "ii", stray fragments
        self.chunks.append(
            Chunk(
                id=cid,
                doc_id=self.meta.doc_id,
                parent_id=parent_id,
                text=text,
                context=self.prefix_at(start, start + len(text)),
                page=self.page.page,
                company=self.meta.company,
                fiscal_year=self.meta.fiscal_year,
                **extra,
            )
        )

    @property
    def page_parent_id(self) -> str:
        return _stable_id(self.meta.doc_id, self.page.page, "parent")

    def emit_span(self, start: int, end: int, **extra) -> None:
        """One chunk for md[start:end]; if it would exceed the cap, window it into pieces."""
        text = self.md[start:end]
        if self._fits(self.prefix_at(start, end), text):
            self._add(_stable_id(self.meta.doc_id, self.page.page, start), start, text,
                      self.page_parent_id, **extra)
            return
        self.emit_windows(start, end)

    def emit_windows(self, start: int, end: int) -> None:
        """Sliding token windows of child_budget (shrunk until every window fits the cap)."""
        size = self.child_budget
        while True:
            pieces = self._windows(start, end, size)
            if all(self._fits(self.prefix_at(s, e), self.md[s:e]) for s, e in pieces) or size <= 8:
                break
            size = max(int(size * 0.9), 8)
        for s, e in pieces:
            self._add(_stable_id(self.meta.doc_id, self.page.page, s), s, self.md[s:e],
                      self.page_parent_id)

    def _windows(self, start: int, end: int, size: int) -> list[tuple[int, int]]:
        offs = [(start + s, start + e) for s, e in self.tok.offsets(self.md[start:end])]
        if not offs:
            return []
        step = max(size - self.c.overlap_tokens, 1)
        out = []
        for s in range(0, len(offs), step):
            e = min(s + size, len(offs))
            out.append((offs[s][0], offs[e - 1][1]))
            if e == len(offs):
                break
        return out

    # ---- tables
    def emit_table(self, b: _Block) -> None:
        text = self.md[b.start : b.end]
        table_id = _stable_id(self.meta.doc_id, self.page.page, b.start, "table")
        if b.tokens <= self.cap_budget and self._fits(self.prefix_at(b.start), text):
            self.emit_span(b.start, b.end, table_id=table_id)
            return
        self.table_parents.append(
            Parent(id=table_id, doc_id=self.meta.doc_id, page_start=self.page.page,
                   page_end=self.page.page, text=text)
        )
        lines = text.split("\n")
        header_n = _header_rows(lines, self.c.max_header_rows)
        header, body = lines[:header_n], lines[header_n:]
        prefix = self.prefix_at(b.start)
        ptoks = self.tok.count(prefix) if prefix else self.overhead
        # row groups aim at table_group_tokens (default: fill up to the cap)
        group_cap = min(getattr(self.c, "table_group_tokens", None) or self.c.max_embed_tokens,
                        self.c.max_embed_tokens)
        avail = group_cap - ptoks
        line_toks = [max(self.tok.count(ln) - self.overhead, 0) for ln in lines]
        htoks = sum(line_toks[:header_n])
        if htoks > avail // 2:  # header too big to repeat
            header, htoks = [], 0
        body_toks = line_toks[header_n:]
        limit = avail - htoks
        groups = self._group_rows(body, body_toks, limit, header, prefix)
        for i, rows in enumerate(groups):
            gtext = "\n".join(header + rows)
            if not self._fits(prefix, gtext):  # one row alone is wider than the cap: window it
                self._emit_wide_row(b, table_id, i, "\n".join(rows))
                continue
            self._add(_stable_id(self.meta.doc_id, self.page.page, b.start, "rg", i), b.start,
                      gtext, table_id, table_id=table_id, row_group=i)

    def _group_rows(self, rows, toks, limit, header, prefix) -> list[list[str]]:
        """Greedy row packing, then rebalanced so groups have similar sizes. Never splits a row."""
        def greedy(lim: int, shrink: float) -> list[list[str]]:
            while True:
                out: list[list[str]] = []
                cur: list[str] = []
                cur_t = 0
                for row, t in zip(rows, toks, strict=True):
                    if cur and cur_t + t > lim:
                        out.append(cur)
                        cur, cur_t = [], 0
                    cur.append(row)
                    cur_t += t
                if cur:
                    out.append(cur)
                if all(self._fits(prefix, "\n".join(header + g)) or len(g) == 1 for g in out):
                    return out
                lim = max(int(lim * shrink), 1)

        first = greedy(limit, 0.95)
        n = len(first)
        if n <= 1:
            return first
        target = math.ceil(sum(toks) / n)
        lim = max(target, max(toks, default=0))
        while lim < limit:
            balanced = greedy(lim, 0.95)
            if len(balanced) <= n:
                return balanced
            lim = min(int(lim * 1.05) + 1, limit)
        return first

    def _emit_wide_row(self, b: _Block, table_id: str, group: int, text: str) -> None:
        size = self.child_budget
        prefix = self.prefix_at(b.start)
        while True:
            offs = self.tok.offsets(text)
            step = max(size - self.c.overlap_tokens, 1)
            pieces = []
            for s in range(0, len(offs), step):
                e = min(s + size, len(offs))
                pieces.append(text[offs[s][0] : offs[e - 1][1]])
                if e == len(offs):
                    break
            if all(self._fits(prefix, p) for p in pieces) or size <= 8:
                break
            size = max(int(size * 0.9), 8)
        for j, piece in enumerate(pieces):
            self._add(_stable_id(self.meta.doc_id, self.page.page, b.start, "rg", group, j),
                      b.start, piece, table_id, table_id=table_id, row_group=group)

    # ---- main loop
    def run(self) -> None:
        group: list[_Block] = []
        group_tokens = 0

        def flush():
            nonlocal group, group_tokens
            if group:
                self.emit_span(group[0].start, group[-1].end)
            group, group_tokens = [], 0

        for b in _blocks(self.md, self.tok):
            if b.tokens > self.child_budget:
                flush()
                if b.is_table:
                    self.emit_table(b)
                else:
                    self.emit_windows(b.start, b.end)
                continue
            if group_tokens + b.tokens > self.child_budget:
                flush()
            group.append(b)
            group_tokens += b.tokens
        flush()


def chunk_pages(pages: list[Page], meta: DocMeta, cfg, tok=None) -> tuple[list[Parent], list[Chunk]]:
    tok = tok or RegexTokenizer()
    parents: list[Parent] = []
    chunks: list[Chunk] = []
    carry_heading = ""
    for page in pages:
        pc = _PageChunker(page.markdown, page, meta, cfg, tok, carry_heading)
        parents.append(
            Parent(id=pc.page_parent_id, doc_id=meta.doc_id, page_start=page.page,
                   page_end=page.page, text=page.markdown)
        )
        pc.run()
        parents.extend(pc.table_parents)
        chunks.extend(pc.chunks)
        carry_heading = pc.last_heading()
    return parents, chunks
