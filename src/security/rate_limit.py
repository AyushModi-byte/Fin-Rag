"""Abuse protection for the public API: per-client sliding-window limits, a daily global cap and
client identification. In memory (one process), no extra dependency; limits reset on restart, the
daily cap is also persisted to a small JSON file so a restart does not hand out a fresh day.
"""
import hashlib
import json
import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MINUTE_S = 60
DAY_S = 86_400


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str | None = None  # "minute" | "day" when blocked
    retry_after: int = 0  # seconds until the client may try again


class RateLimiter:
    """Sliding windows per client: `per_minute` over 60 s and `per_day` over 24 h.

    Only accepted requests are recorded, so hammering a blocked client does not keep it blocked.
    At most `max_clients` clients are tracked (least recently seen is dropped), so memory is bounded.
    """

    def __init__(self, per_minute: int, per_day: int, max_clients: int, clock=time.time):
        self.per_minute, self.per_day, self.max_clients, self.clock = per_minute, per_day, max_clients, clock
        self._hits: OrderedDict[str, deque] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, client: str) -> Decision:
        now = self.clock()
        with self._lock:
            hits = self._hits.setdefault(client, deque())
            self._hits.move_to_end(client)
            while hits and now - hits[0] >= DAY_S:
                hits.popleft()
            if len(hits) >= self.per_day:
                decision = Decision(False, "day", max(1, math.ceil(hits[0] + DAY_S - now)))
            else:
                recent = [t for t in hits if now - t < MINUTE_S]
                if len(recent) >= self.per_minute:
                    decision = Decision(False, "minute", max(1, math.ceil(recent[0] + MINUTE_S - now)))
                else:
                    hits.append(now)
                    decision = Decision(True)
            while len(self._hits) > self.max_clients:
                self._hits.popitem(last=False)
            return decision

    def tracked(self) -> int:
        return len(self._hits)


class DailyCap:
    """At most `limit` accepted calls per UTC day, across all clients; persisted to `path`."""

    def __init__(self, limit: int, path: str, clock=time.time):
        self.limit, self.path, self.clock = limit, Path(path), clock
        self._lock = threading.Lock()
        self._day, self._count = self._load()

    def _today(self) -> str:
        return datetime.fromtimestamp(self.clock(), timezone.utc).date().isoformat()

    def _load(self) -> tuple[str, int]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return str(data["date"]), int(data["count"])
        except (OSError, ValueError, KeyError, TypeError):
            return "", 0  # missing or corrupt file: start a fresh day

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps({"date": self._day, "count": self._count}), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass  # a read-only disk must not take the API down; the in-memory count still holds

    def try_acquire(self) -> bool:
        with self._lock:
            today = self._today()
            if today != self._day:
                self._day, self._count = today, 0
            if self._count >= self.limit:
                return False
            self._count += 1
            self._save()
            return True

    def seconds_to_reset(self) -> int:
        now = self.clock()
        return max(1, int(DAY_S - (now % DAY_S)))  # UTC midnight


def client_ip(headers, peer: str | None, hops: int) -> str:
    """The client address. Behind `hops` trusted proxies, each appends the address it saw to the right of
    X-Forwarded-For, so the entry `hops` from the right is the one our own proxy recorded; anything to
    its left is client-supplied and can be forged. With hops=0 or a short header, use the socket peer."""
    peer = peer or "unknown"
    if hops <= 0:
        return peer
    raw = headers.get("x-forwarded-for", "")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts[-hops] if len(parts) >= hops else peer


def client_label(ip: str, salt: str) -> str:
    """Stable pseudonym for logs: the same client maps to the same label, the address is not recoverable."""
    return hashlib.sha256(f"{salt}|{ip}".encode()).hexdigest()[:12]
