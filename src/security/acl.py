"""Per-user company ACL and PII redaction.

ACL comes from `config.yaml` (`acl.default`, `acl.users`). `"*"` means every company. A user not
listed in `acl.users` gets `acl.default`.
"""
import re

from src.config import load_config

ALL = "*"

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
# Phones need separators between digit groups (415-555-0123, (415) 555 0123, +1 415.555.0123), or a
# leading "+" with a long digit run. Plain numbers such as 1,234,567.89, 2018 or 60-61 never match.
_PHONE = re.compile(
    r"(?<![\w.])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\w])"
    r"|(?<![\w.])\+\d{1,3}(?:[\s.-]?\d{2,4}){3,5}(?![\w])"
)


def redact_pii(text: str) -> str:
    """Replace e-mail addresses and phone numbers with [EMAIL] / [PHONE]."""
    return _PHONE.sub("[PHONE]", _EMAIL.sub("[EMAIL]", text))


def _as_set(value) -> set[str]:
    if isinstance(value, str):
        return {value}
    return {str(v) for v in value}


def allowed_companies(user: str | None, cfg=None) -> set[str]:
    """Companies `user` may query. `{"*"}` means no restriction."""
    cfg = cfg or load_config()
    users = vars(cfg.acl.users) if cfg.acl.users is not None else {}
    if user is not None and user in users:
        return _as_set(users[user])
    return _as_set(cfg.acl.default)


def is_allowed(allowed: set[str], companies: list[str]) -> bool:
    """True if every company named in the question is permitted."""
    return ALL in allowed or all(c in allowed for c in companies)
