"""BM25 v2 tokenizer: stemming, but numbers and identifiers stay searchable.

* "3M" -> `3m`; "10-K" -> `10-k` and `10`; "1,234.5" -> `1,234.5` and `1234.5`; "62%" -> `62%`
  and `62`; "(1,577)" -> `1,577` and `1577`. A query written either way matches either form.
* Alphabetic words are lower-cased, stop-worded, then stemmed (Snowball English), so "revenues"
  and "revenue" meet. Mixed alphanumerics such as `3m` or `q4` are never stemmed.
"""
import re

import bm25s.stopwords as _sw
from py_rust_stemmers import SnowballStemmer

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-.,'][A-Za-z0-9]+)*%?")
_STEMMER = SnowballStemmer("english")


def _stopword_set(name: str) -> frozenset[str]:
    if not name:
        return frozenset()
    return frozenset(getattr(_sw, f"STOPWORDS_{name.upper()}"))


def _number_forms(tok: str) -> list[str]:
    pct = tok.endswith("%")
    core = tok[:-1] if pct else tok
    forms = [tok]
    if pct:
        forms.append(core)
    if "," in core:
        forms.append(core.replace(",", ""))
    if "-" in core:
        forms.extend(p for p in core.split("-") if len(p) >= 2)
    return forms


def _word_forms(tok: str, stops: frozenset[str], stem: bool) -> list[str]:
    word = tok.replace("'s", "").replace("'", "").replace(".", "")
    pieces = [word] + (word.split("-") if "-" in word else [])
    out = []
    for piece in dict.fromkeys(pieces):
        if len(piece) < 2 or piece in stops:
            continue
        out.append(_STEMMER.stem_word(piece) if stem and piece.isalpha() and len(piece) > 3 else piece)
    return out


def tokenize(text: str, stopwords: str = "en", stem: bool = True) -> list[str]:
    stops = _stopword_set(stopwords)
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        tok = raw.lower()
        if any(ch.isdigit() for ch in tok):
            out.extend(_number_forms(tok))
        else:
            out.extend(_word_forms(tok, stops, stem))
    return out
