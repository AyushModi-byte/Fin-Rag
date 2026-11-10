"""LLMClient: the only place an LLM API is ever called.

Order per request: disk cache (every configured provider's entry) -> primary provider (throttled,
retried with exponential backoff on 429/5xx/timeout) -> fallback provider, if one is configured.

A provider may have several API keys. A 429 on one key moves to the next key immediately and puts
the limited key on a cooldown. Every real request increments a per-key daily counter; a key at
`budget_warn_ratio` of `daily_budget` is skipped, and when every key is there the client stops and
raises BudgetNearlyExhausted so the user is told before anything else is spent.

API keys travel only in request headers and are never logged or put in error messages.
"""
import hashlib
import json
import logging
import os
import re
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


_RETRY_IN = re.compile(r"try again in\s+((?:\d+(?:\.\d+)?\s*(?:ms|s|m|h)\s*)+)", re.IGNORECASE)
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h)", re.IGNORECASE)
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_retry_after(message: str) -> float | None:
    """Seconds from a provider hint like "Please try again in 1m3.5s"; None if there is none."""
    m = _RETRY_IN.search(message or "")
    if not m:
        return None
    return sum(float(n) * _UNIT_SECONDS[u.lower()] for n, u in _DURATION_PART.findall(m.group(1)))


class LLMError(RuntimeError):
    """No provider could answer (all failed, or none configured)."""


class BudgetNearlyExhausted(RuntimeError):
    """Every key of a provider reached budget_warn_ratio of its daily budget; stop and tell the user."""


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
    key_envs: list[str]


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


def _key_envs(role) -> list[str]:
    envs = getattr(role, "api_key_envs", None)
    return list(envs) if envs else [role.api_key_env]


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
        for role in (cfg.llm.primary, getattr(cfg.llm, "fallback", None)):
            if role is None:
                continue
            if role.provider not in _ADAPTERS:
                raise ValueError(f"unsupported LLM provider: {role.provider}")
            self.providers.append(
                _Provider(role.provider, role.model, role.rpm, role.daily_budget, _key_envs(role))
            )
        self._api_keys = api_keys
        self._cache_dir = Path(cfg.llm.cache_dir)
        self._last_request: dict[str, float] = {}  # per key env
        self._cooldown_until: dict[str, float] = {}  # per key env
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
            if not self._usable_envs(p):
                failures.append(f"{p.name}: {' / '.join(p.key_envs)} is not set")
                continue
            try:
                text = self._call_with_retries(p, prompt, system)
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

    # ---- keys ----

    def _key_value(self, env: str) -> str | None:
        keys = self._api_keys if self._api_keys is not None else os.environ
        value = keys.get(env)
        return value.strip() if value and value.strip() else None

    def _usable_envs(self, p: _Provider) -> list[str]:
        """Configured key env vars that actually have a value."""
        return [e for e in p.key_envs if self._key_value(e)]

    def _threshold(self, p: _Provider) -> float:
        return p.daily_budget * self.cfg.llm.budget_warn_ratio

    def _under_budget(self, p: _Provider) -> list[str]:
        return [e for e in self._usable_envs(p) if self.calls_today(p.name, e) < self._threshold(p)]

    def _is_fresh(self, env: str) -> bool:
        return self._cooldown_until.get(env, 0.0) <= self._clock()

    def _pick_key(self, p: _Provider) -> str:
        """Next key to use: first one under budget that is not cooling down; if every eligible key is
        cooling, the one that cools down soonest. Raises BudgetNearlyExhausted if all are over budget."""
        eligible = self._under_budget(p)
        if not eligible:
            counts = ", ".join(
                f"{e}={self.calls_today(p.name, e)}/{p.daily_budget}" for e in self._usable_envs(p)
            )
            raise BudgetNearlyExhausted(
                f"{p.name}: every API key has used >= {self.cfg.llm.budget_warn_ratio:.0%} of its daily "
                f"budget ({counts}). Stopping; tell the user before continuing."
            )
        fresh = [e for e in eligible if self._is_fresh(e)]
        if fresh:
            return fresh[0]
        return min(eligible, key=lambda e: self._cooldown_until.get(e, 0.0))

    def _has_other_fresh_key(self, p: _Provider, exclude: str) -> bool:
        return any(e != exclude and self._is_fresh(e) for e in self._under_budget(p))

    # ---- one provider ----

    def _call_with_retries(self, p: _Provider, prompt: str, system: str | None) -> str:
        build, parse = _ADAPTERS[p.name]
        url_tpl = self.cfg.llm.endpoints.__dict__[p.name]
        max_attempts = 1 + self.cfg.llm.max_retries
        attempts, reason = 0, "unknown"
        hint: float | None = None
        while attempts < max_attempts:
            hint = None
            env = self._pick_key(p)
            url, headers, payload = build(url_tpl, p, self._key_value(env), prompt, system, self.cfg)
            self._throttle(env, p.rpm)
            self._count_call(p, env)
            logger.debug(
                "request provider=%s model=%s key=%s attempt=%d/%d", p.name, p.model, env, attempts + 1, max_attempts
            )
            try:
                resp = self._transport(url, headers, payload, self.cfg.llm.timeout_s)
            except OSError as e:  # includes TimeoutError / ConnectionError
                reason = type(e).__name__
            else:
                if resp.status == 200:
                    text = parse(resp.body).strip()
                    if text:
                        return text
                    raise _ProviderFailed("empty response")
                reason = f"HTTP {resp.status}" + self._error_detail(resp.body)
                if resp.status == 429:
                    hint = parse_retry_after(self._raw_error_message(resp.body))
                    self._cooldown_until[env] = self._clock() + self.cfg.llm.key_cooldown_s
                    if self._has_other_fresh_key(p, env):
                        logger.warning("%s key %s rate limited; rotating to the next key", p.name, env)
                        continue  # rotate immediately: no backoff, no attempt used (bounded by key count)
                elif resp.status not in _RETRYABLE_STATUS and resp.status < 500:
                    raise _ProviderFailed(reason)
            attempts += 1
            if attempts < max_attempts:
                wait = self.cfg.llm.backoff_base_s * 2 ** (attempts - 1)
                if hint is not None:  # honor the provider's own "try again in" (capped, never shorter)
                    wait = min(max(wait, hint), self.cfg.llm.max_retry_wait_s)
                self._sleep(wait)
        raise _ProviderFailed(f"{reason} after {attempts} attempts")

    @staticmethod
    def _raw_error_message(body: dict) -> str:
        error = body.get("error") if isinstance(body, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        return message if isinstance(message, str) else ""

    def _error_detail(self, body: dict) -> str:
        """Provider error message for diagnostics: keys redacted, truncated."""
        message = self._raw_error_message(body)
        if not message.strip():
            return ""
        for p in self.providers:
            for env in p.key_envs:
                key = self._key_value(env)
                if key:
                    message = message.replace(key, "[REDACTED]")
        return ": " + " ".join(message.split())[:_MAX_ERROR_CHARS]

    def _throttle(self, env: str, rpm: float) -> None:
        last = self._last_request.get(env)
        if last is not None:
            wait = last + 60.0 / rpm - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request[env] = self._clock()

    # ---- daily budget (per key) ----

    @property
    def _budget_file(self) -> Path:
        return self._cache_dir / "budget.json"

    def _read_budget(self) -> dict:
        try:
            return json.loads(self._budget_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def calls_today(self, provider: str, env: str | None = None) -> int:
        """Real requests made today: for one key, or summed over all keys of the provider."""
        data = self._read_budget()
        total = 0
        for name, entry in data.items():
            prov, _, key_env = name.partition(":")
            if prov != provider or (env not in (None, "") and key_env != env):
                continue
            if entry.get("date") == self._today():
                total += entry.get("calls", 0)
        return total

    def _count_call(self, p: _Provider, env: str) -> None:
        data = self._read_budget()
        name = f"{p.name}:{env}"
        data[name] = {"date": self._today(), "calls": self.calls_today(p.name, env) + 1}
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._budget_file.write_text(json.dumps(data), encoding="utf-8")

    # ---- disk cache (key includes provider and model, never the API key) ----

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
