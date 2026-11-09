"""Rule-based query routing: detect company and fiscal year in the question text.

No LLM fallback (project rule: no extra LLM calls in pipelines without asking). A question that names
no known company is treated as out of scope.
"""
import re
from dataclasses import dataclass, field

# 4-digit years 1990-2099. Fiscal-year markers ("FY2018", "fiscal year 2018") win over bare years.
_FY = re.compile(r"\b(?:fy|fiscal(?:\s+year)?)\s*[-']?\s*((?:19[9]|20\d)\d)\b", re.IGNORECASE)
_BARE_YEAR = re.compile(r"\b((?:199|20\d)\d)\b")


@dataclass
class RoutedQuery:
    filters: dict = field(default_factory=dict)
    query: str = ""
    in_scope: bool = False


def _company_pattern(name: str) -> re.Pattern:
    parts = [re.escape(p) for p in re.split(r"[\s_\-]+", name.strip()) if p]
    return re.compile(r"(?<!\w)" + r"[\s_\-]*".join(parts) + r"(?!\w)", re.IGNORECASE)


def _detect_companies(q: str, known_companies: list[str]) -> list[str]:
    return [c for c in known_companies if _company_pattern(c).search(q)]


def _detect_year(q: str) -> int | None:
    years = {int(y) for y in _FY.findall(q)} or {int(y) for y in _BARE_YEAR.findall(q)}
    return years.pop() if len(years) == 1 else None  # none or ambiguous -> no year filter


def route(q: str, known_companies: list[str]) -> RoutedQuery:
    companies = _detect_companies(q, known_companies)
    if not companies:
        return RoutedQuery(filters={}, query=q, in_scope=False)
    filters: dict = {}
    if len(companies) == 1:
        filters["company"] = companies[0]
    year = _detect_year(q)
    if year is not None:
        filters["fiscal_year"] = year
    return RoutedQuery(filters=filters, query=q, in_scope=True)
