from pydantic import BaseModel


class Page(BaseModel):
    doc_id: str
    page: int
    markdown: str


class DocMeta(BaseModel):
    doc_id: str
    company: str
    fiscal_year: int
    doc_type: str


class Parent(BaseModel):
    id: str
    doc_id: str
    page_start: int
    page_end: int
    text: str


class Chunk(BaseModel):
    id: str
    doc_id: str
    parent_id: str
    text: str
    context: str
    page: int
    company: str
    fiscal_year: int
    table_id: str | None = None  # set when the chunk holds (part of) a table; the id of its parent
    row_group: int | None = None  # 0-based index of the row group within a split table


class Hit(BaseModel):
    chunk_id: str
    score: float
    chunk: Chunk


class Citation(BaseModel):
    doc_id: str
    page: int


class QueryResult(BaseModel):
    answer: str
    citations: list[Citation]
    contexts: list[str]
    latency_ms: dict[str, float]
