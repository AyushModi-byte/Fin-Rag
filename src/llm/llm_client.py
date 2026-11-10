"""LLMClient: the only place an LLM API is ever called.

Order per request: disk cache (primary, then fallback entries) -> primary provider (throttled,
retried with exponential backoff on 429/5xx/timeout) -> fallback provider. Every real request
increments a per-provider daily counter; at `budget_warn_ratio` of `daily_budget` the client stops
and raises BudgetNearlyExhausted so the user is told before anything else is spent.

API keys travel only in request headers and are never logged or put in error messages.
"""
import hashlib
import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

logger = logging.getLogger("finrag.llm")

_RETRYABLE_STATUS = {408, 429}  # plus every 5xx
_MAX_ERROR_CHARS = 300
_USER_AGENT = "finrag/0.1"  # some edges reject urllib's default agent


class LLMError(RuntimeError):
    """No provider could answer (all failed, or none configured)."""


class BudgetNearlyExhausted(RuntimeError):
    """A provider reached budget_warn_ratio of its daily call budget; stop and tell the user."""


class _ProviderFailed(Exception):
    """One provider gave up (retries exhausted or non-retryable); try the next one."""


@dataclass
class HttpResponse:
    status: int
    body: dict


@dataclass
class LLMCallInfo:
    provider: str
    model: str
    cached: bool


def urllib_transport(url: str, headers: dict, payload: dict, timeout_s: float) -> HttpResponse:
    """Default HTTP layer. Returns HttpResponse for any HTTP status; raises OSError on network
    problems and TimeoutError on timeouts."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json", "User-Agent": _USER_AGENT},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return HttpResponse(resp.status, json.loads(resp.read().decode("utf-8")))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except (ValueError, OSError):
            body = {}
        return HttpResponse(e.code, body if isinstance(body, dict) else {})
    except urllib.error.URLError as e:
        if isinstance(e.reason, socket.timeout):
            raise TimeoutError("request timed out") from None
        raise ConnectionError("network error") from None


@dataclass
class _Provider:
    name: str
    model: str
    rpm: float
    daily_budget: int
    key_env: str


def _gemini_request(url_tpl, p, key, prompt, system, cfg):
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": cfg.llm.temperature,
            "maxOutputTokens": cfg.llm.max_output_tokens,
        },
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    return url_tpl.format(model=p.model), {"x-goog-api-key": key}, payload


def _gemini_parse(body: dict) -> str:
    try:
        parts = body["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return ""
    return "".join(part.get("text", "") for part in parts if isinstance(part, dict))


def _groq_request(url_tpl, p, key, prompt, system, cfg):
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": p.model,
        "messages": messages,
        "temperature": cfg.llm.temperature,
        "max_tokens": cfg.llm.max_output_tokens,
    }
    return url_tpl, {"Authorization": f"Bearer {key}"}, payload


def _groq_parse(body: dict) -> str:
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


_ADAPTERS = {
    "gemini": (_gemini_request, _gemini_parse),
    "groq": (_groq_request, _groq_parse),
}


class LLMClient:
    def __init__(
        self,
        cfg,
        transport: Callable[..., HttpResponse] = urllib_transport,
        api_keys: dict[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], str] = lambda: date.today().isoformat(),
    ):
        self.cfg = cfg
        self._transport, self._sleep, self._clock, self._today = transport, sleep, clock, today
        self.providers = []
        for role in (cfg.llm.primary, cfg.llm.fallback):
            if role.provider not in _ADAPTERS:
                raise ValueError(f"unsupported LLM provider: {role.provider}")
            self.providers.append(
                _Provider(role.provider, role.model, role.rpm, role.daily_budget, role.api_key_env)
            )
        self._api_keys = api_keys
        self._cache_dir = Path(cfg.llm.cache_dir)
        self._last_request: dict[str, float] = {}
        self.last_call: LLMCallInfo | None = None
        self.served: Counter = Counter()

    # ---- public ----

    def complete(self, prompt: str, system: str | None = None, only: str | None = None) -> str:
        """Answer a prompt. `only` restricts to one provider by name (e.g. "groq")."""
        providers = [p for p in self.providers if only in (None, p.name)]
        if not providers:
            raise LLMError(f"unknown provider {only!r}")
        for p in providers:
            cached = self._cache_get(p, system, prompt)
            if cached is not None:
                self.last_call = LLMCallInfo(p.name, p.model, True)
                self.served["cache"] += 1
                return cached

        failures = []
        for i, p in enumerate(providers):
            key = self._key(p)
            if not key:
                failures.append(f"{p.name}: {p.key_env} is not set")
                continue
            try:
                text = self._call_with_retries(p, key, prompt, system)
            except _ProviderFailed as e:
                failures.append(f"{p.name}: {e}")
                if i + 1 < len(providers):
                    logger.warning("%s failed (%s); falling back to %s", p.name, e, providers[i + 1].name)
                continue
            self._cache_put(p, system, prompt, text)
            self.last_call = LLMCallInfo(p.name, p.model, False)
            self.served[p.name] += 1
            return text
        raise LLMError("all providers failed: " + "; ".join(failures))

    # ---- providers ----

    def _key(self, p: _Provider) -> str | None:
        keys = self._api_keys if self._api_keys is not None else os.environ
        value = keys.get(p.key_env)
        return value.strip() if value and value.strip() else None

    def _call_with_retries(self, p, key, prompt, system) -> str:
        build, parse = _ADAPTERS[p.name]
        url_tpl = self.cfg.llm.endpoints.__dict__[p.name]
        url, headers, payload = build(url_tpl, p, key, prompt, system, self.cfg)
        attempts = 1 + self.cfg.llm.max_retries
        reason = "unknown"
        for attempt in range(attempts):
            self._check_budget(p)
            self._throttle(p)
            self._count_call(p)
            logger.debug("request provider=%s model=%s attempt=%d/%d", p.name, p.model, attempt + 1, attempts)
            try:
                resp = self._transport(url, headers, payload, self.cfg.llm.timeout_s)
            except OSError as e:  # includes TimeoutError / ConnectionError
                reason, retryable = type(e).__name__, True
            else:
                if resp.status == 200:
                    text = parse(resp.body).strip()
                    if text:
                        return text
                    raise _ProviderFailed("empty response")
                reason = f"HTTP {resp.status}" + self._error_detail(resp.body)
                retryable = resp.status in _RETRYABLE_STATUS or resp.status >= 500
                if not retryable:
                    raise _ProviderFailed(reason)
            if attempt + 1 < attempts:
                self._sleep(self.cfg.llm.backoff_base_s * 2**attempt)
        raise _ProviderFailed(f"{reason} after {attempts} attempts")

    def _error_detail(self, body: dict) -> str:
        """Provider error message for diagnostics: keys redacted, truncated."""
        error = body.get("error") if isinstance(body, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if not isinstance(message, str) or not message.strip():
            return ""
        for p in self.providers:
            key = self._key(p)
            if key:
                message = message.replace(key, "[REDACTED]")
        return ": " + " ".join(message.split())[:_MAX_ERROR_CHARS]

    def _throttle(self, p: _Provider) -> None:
        last = self._last_request.get(p.name)
        if last is not None:
            wait = last + 60.0 / p.rpm - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request[p.name] = self._clock()

    # ---- daily budget ----

    @property
    def _budget_file(self) -> Path:
        return self._cache_dir / "budget.json"

    def _read_budget(self) -> dict:
        try:
            return json.loads(self._budget_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def calls_today(self, name: str) -> int:
        entry = self._read_budget().get(name, {})
        return entry.get("calls", 0) if entry.get("date") == self._today() else 0

    def _check_budget(self, p: _Provider) -> None:
        used, limit = self.calls_today(p.name), p.daily_budget
        if used >= limit * self.cfg.llm.budget_warn_ratio:
            raise BudgetNearlyExhausted(
                f"{p.name} has used {used}/{limit} of today's calls "
                f"(>= {self.cfg.llm.budget_warn_ratio:.0%}). Stopping; tell the user before continuing."
            )

    def _count_call(self, p: _Provider) -> None:
        data = self._read_budget()
        data[p.name] = {"date": self._today(), "calls": self.calls_today(p.name) + 1}
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._budget_file.write_text(json.dumps(data), encoding="utf-8")

    # ---- disk cache (key includes provider and model) ----

    def _cache_path(self, p: _Provider, system, prompt) -> Path:
        raw = json.dumps([p.name, p.model, system or "", prompt], ensure_ascii=False)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return self._cache_dir / "responses" / f"{digest}.json"

    def _cache_get(self, p, system, prompt) -> str | None:
        path = self._cache_path(p, system, prompt)
        try:
            return json.loads(path.read_text(encoding="utf-8"))["text"]
        except (OSError, ValueError, KeyError):
            return None

    def _cache_put(self, p, system, prompt, text) -> None:
        path = self._cache_path(p, system, prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"provider": p.name, "model": p.model, "text": text}, ensure_ascii=False),
            encoding="utf-8",
        )
