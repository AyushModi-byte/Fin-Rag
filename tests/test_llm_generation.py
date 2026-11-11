import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import load_config
from src.generation.grounded_answer import generate
from src.prompts.prompt_templates import IDK
from src.llm.llm_client import BudgetNearlyExhausted, HttpResponse, LLMClient, LLMError
from src.schemas.models import Chunk, Hit

GEM_KEY, GROQ_KEY = "gem-secret-123", "groq-secret-456"
KEYS = {"GEMINI_API_KEY": GEM_KEY, "GROQ_API_KEY": GROQ_KEY}


def gemini_ok(text="gemini answer"):
    return HttpResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]})


def groq_ok(text="groq answer"):
    return HttpResponse(200, {"choices": [{"message": {"content": text}}]})


class FakeTransport:
    """Scripted HTTP layer. Responses are consumed per provider (matched by URL)."""

    def __init__(self, gemini=(), groq=()):
        self.script = {"generativelanguage": list(gemini), "groq.com": list(groq)}
        self.calls = []

    def __call__(self, url, headers, payload, timeout_s):
        self.calls.append({"url": url, "headers": headers, "payload": payload, "timeout_s": timeout_s})
        key = "generativelanguage" if "generativelanguage" in url else "groq.com"
        queue = self.script[key]
        if not queue:
            raise AssertionError(f"unexpected extra call to {key}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def count(self, which):
        return sum(1 for c in self.calls if which in c["url"])


class FakeTime:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _provider(provider, model, key_env=None, key_envs=None, daily_budget=200):
    p = SimpleNamespace(provider=provider, model=model, rpm=600000, daily_budget=daily_budget)
    if key_envs:
        p.api_key_envs = key_envs
    else:
        p.api_key_env = key_env
    return p  # rpm is effectively unlimited unless a test sets it


@pytest.fixture
def cfg(tmp_path):
    """Tests build their own two-provider setup so they do not depend on the production config."""
    c = load_config()
    c.llm.cache_dir = str(tmp_path / "llm")
    c.llm.primary = _provider("gemini", "gemini-test", key_env="GEMINI_API_KEY")
    c.llm.fallback = _provider("groq", "groq-test", key_env="GROQ_API_KEY", daily_budget=1000)
    return c


@pytest.fixture
def tm():
    return FakeTime()


def make_client(cfg, tm, transport, keys=None, today="2026-11-10"):
    return LLMClient(
        cfg, transport=transport, api_keys=KEYS if keys is None else keys,
        sleep=tm.sleep, clock=tm.clock, today=lambda: today,
    )


# ---------------- LLMClient ----------------

def test_retries_on_429_then_succeeds(cfg, tm):
    t = FakeTransport(gemini=[HttpResponse(429, {}), gemini_ok()])
    out = make_client(cfg, tm, t).complete("question")
    assert out == "gemini answer"
    assert t.count("generativelanguage") == 2
    assert t.count("groq.com") == 0
    assert tm.sleeps == [cfg.llm.backoff_base_s]


@pytest.mark.parametrize("failure", [HttpResponse(503, {}), TimeoutError("timed out")])
def test_retries_on_5xx_and_timeout(cfg, tm, failure):
    t = FakeTransport(gemini=[failure, gemini_ok()])
    assert make_client(cfg, tm, t).complete("question") == "gemini answer"
    assert t.count("generativelanguage") == 2


def test_falls_back_to_groq_after_gemini_retries_exhausted(cfg, tm):
    attempts = 1 + cfg.llm.max_retries
    t = FakeTransport(gemini=[HttpResponse(429, {})] * attempts, groq=[groq_ok()])
    client = make_client(cfg, tm, t)
    assert client.complete("question") == "groq answer"
    assert t.count("generativelanguage") == attempts
    assert t.count("groq.com") == 1
    base = cfg.llm.backoff_base_s
    assert tm.sleeps == [base, base * 2]  # exponential backoff, no sleep after the last attempt
    assert client.last_call.provider == "groq"
    assert client.last_call.cached is False


def test_non_retryable_status_falls_through_without_retry(cfg, tm):
    t = FakeTransport(gemini=[HttpResponse(401, {})], groq=[groq_ok()])
    assert make_client(cfg, tm, t).complete("question") == "groq answer"
    assert t.count("generativelanguage") == 1
    assert tm.sleeps == []


def test_empty_gemini_response_falls_through(cfg, tm):
    blocked = HttpResponse(200, {"candidates": [{"finishReason": "SAFETY"}]})
    t = FakeTransport(gemini=[blocked], groq=[groq_ok()])
    assert make_client(cfg, tm, t).complete("question") == "groq answer"
    assert t.count("generativelanguage") == 1


def test_all_providers_failing_raises_clean_error(cfg, tm):
    attempts = 1 + cfg.llm.max_retries
    t = FakeTransport(gemini=[HttpResponse(500, {})] * attempts, groq=[TimeoutError()] * attempts)
    with pytest.raises(LLMError) as exc:
        make_client(cfg, tm, t).complete("question")
    assert GEM_KEY not in str(exc.value) and GROQ_KEY not in str(exc.value)
    assert t.count("generativelanguage") == attempts and t.count("groq.com") == attempts


def test_timeout_from_config_is_passed_to_transport(cfg, tm):
    t = FakeTransport(gemini=[gemini_ok()])
    make_client(cfg, tm, t).complete("question")
    assert t.calls[0]["timeout_s"] == cfg.llm.timeout_s


def test_cache_hit_makes_no_second_call(cfg, tm):
    t = FakeTransport(gemini=[gemini_ok()])
    client = make_client(cfg, tm, t)
    assert client.complete("question", system="sys") == "gemini answer"
    assert client.complete("question", system="sys") == "gemini answer"
    assert t.count("generativelanguage") == 1
    assert client.last_call.cached is True
    assert client.served["cache"] == 1


def test_cache_survives_a_new_client_and_serves_fallback_answers(cfg, tm):
    attempts = 1 + cfg.llm.max_retries
    t = FakeTransport(gemini=[HttpResponse(429, {})] * attempts, groq=[groq_ok()])
    make_client(cfg, tm, t).complete("question")
    calls_before = len(t.calls)
    again = make_client(cfg, tm, t)  # fresh instance, same cache dir; script is empty
    assert again.complete("question") == "groq answer"
    assert len(t.calls) == calls_before


def test_cache_key_includes_model(cfg, tm):
    t = FakeTransport(gemini=[gemini_ok("first"), gemini_ok("second")])
    assert make_client(cfg, tm, t).complete("question") == "first"
    cfg.llm.primary.model = "another-model"
    assert make_client(cfg, tm, t).complete("question") == "second"
    assert t.count("generativelanguage") == 2


def test_budget_counter_blocks_at_80_percent(cfg, tm):
    cfg.llm.primary.daily_budget = 10  # 80% = 8 calls
    t = FakeTransport(gemini=[gemini_ok(f"a{i}") for i in range(20)], groq=[groq_ok()])
    client = make_client(cfg, tm, t)
    for i in range(8):
        client.complete(f"q{i}")
    with pytest.raises(BudgetNearlyExhausted) as exc:
        client.complete("q8")
    assert "gemini" in str(exc.value)
    assert t.count("generativelanguage") == 8
    assert t.count("groq.com") == 0  # stops and tells the user instead of silently switching


def test_budget_counts_retries_and_persists_across_instances(cfg, tm):
    cfg.llm.primary.daily_budget = 10
    t = FakeTransport(gemini=[HttpResponse(429, {}), gemini_ok()] + [gemini_ok()] * 6)
    first = make_client(cfg, tm, t)
    first.complete("q0")  # 2 requests (one retry)
    for i in range(1, 7):
        first.complete(f"q{i}")  # 6 more -> 8 requests total
    second = make_client(cfg, tm, t)  # reads the counter from disk
    with pytest.raises(BudgetNearlyExhausted):
        second.complete("q-new")


def test_budget_resets_on_a_new_day(cfg, tm):
    cfg.llm.primary.daily_budget = 10
    t = FakeTransport(gemini=[gemini_ok(f"a{i}") for i in range(9)])
    day1 = make_client(cfg, tm, t, today="2026-11-10")
    for i in range(8):
        day1.complete(f"q{i}")
    day2 = make_client(cfg, tm, t, today="2026-11-11")
    assert day2.complete("q-next-day") == "a8"


def test_cache_hits_do_not_consume_budget(cfg, tm):
    cfg.llm.primary.daily_budget = 10
    t = FakeTransport(gemini=[gemini_ok()])
    client = make_client(cfg, tm, t)
    for _ in range(20):
        client.complete("same question")
    assert t.count("generativelanguage") == 1


def test_throttle_waits_to_respect_rpm(cfg, tm):
    cfg.llm.primary.rpm = 10  # one request every 6 seconds
    t = FakeTransport(gemini=[gemini_ok("a"), gemini_ok("b")])
    client = make_client(cfg, tm, t)
    client.complete("q1")
    assert tm.sleeps == []
    client.complete("q2")
    assert tm.sleeps == [pytest.approx(6.0)]


def test_gemini_request_shape(cfg, tm):
    t = FakeTransport(gemini=[gemini_ok()])
    make_client(cfg, tm, t).complete("the prompt", system="the system")
    call = t.calls[0]
    assert cfg.llm.primary.model in call["url"]
    assert GEM_KEY not in call["url"]  # key travels in a header, never in the URL
    assert call["headers"]["x-goog-api-key"] == GEM_KEY
    body = call["payload"]
    assert body["contents"][0]["parts"][0]["text"] == "the prompt"
    assert body["systemInstruction"]["parts"][0]["text"] == "the system"
    assert body["generationConfig"]["temperature"] == cfg.llm.temperature
    assert body["generationConfig"]["maxOutputTokens"] == cfg.llm.max_output_tokens


def test_groq_request_shape_and_only_param(cfg, tm):
    t = FakeTransport(groq=[groq_ok()])
    out = make_client(cfg, tm, t).complete("the prompt", system="the system", only="groq")
    assert out == "groq answer"
    call = t.calls[0]
    assert call["headers"]["Authorization"] == f"Bearer {GROQ_KEY}"
    body = call["payload"]
    assert body["model"] == cfg.llm.fallback.model
    assert body["messages"] == [
        {"role": "system", "content": "the system"},
        {"role": "user", "content": "the prompt"},
    ]
    assert body["temperature"] == cfg.llm.temperature


def test_missing_key_skips_provider_and_never_calls_it(cfg, tm):
    t = FakeTransport(groq=[groq_ok()])
    out = make_client(cfg, tm, t, keys={"GROQ_API_KEY": GROQ_KEY}).complete("question")
    assert out == "groq answer"
    assert t.count("generativelanguage") == 0


def test_no_keys_raises_error_naming_env_vars_only(cfg, tm):
    with pytest.raises(LLMError) as exc:
        make_client(cfg, tm, FakeTransport(), keys={}).complete("question")
    assert "GEMINI_API_KEY" in str(exc.value)


def test_api_keys_never_appear_in_logs(cfg, tm, caplog):
    attempts = 1 + cfg.llm.max_retries
    t = FakeTransport(gemini=[HttpResponse(429, {})] * attempts, groq=[groq_ok()])
    with caplog.at_level(logging.DEBUG):
        make_client(cfg, tm, t).complete("question")
    assert GEM_KEY not in caplog.text and GROQ_KEY not in caplog.text
    assert "groq" in caplog.text.lower()  # but the provider switch is logged


def test_budget_file_lives_in_cache_dir(cfg, tm):
    t = FakeTransport(gemini=[gemini_ok()])
    make_client(cfg, tm, t).complete("question")
    data = json.loads((Path(cfg.llm.cache_dir) / "budget.json").read_text())
    assert data["gemini:GEMINI_API_KEY"] == {"date": "2026-11-10", "calls": 1}


# ---------------- generate() ----------------

class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def complete(self, prompt, system=None, **kwargs):
        self.calls.append({"prompt": prompt, "system": system})
        return self.answer


def make_hit(doc_id="3M_2018_10K", page=60, text="Purchases of PP&E (1,577)"):
    chunk = Chunk(
        id=f"{doc_id}-{page}", doc_id=doc_id, parent_id="p", text=text,
        context=f"3M 2018 10-K, page {page}", page=page, company="3M", fiscal_year=2018,
    )
    return Hit(chunk_id=chunk.id, score=1.0, chunk=chunk)


def test_generate_no_context_says_i_dont_know():
    llm = FakeLLM("should never be used")
    result = generate("What was capex?", [], llm)
    assert result.answer == "I don't know"
    assert result.citations == [] and result.contexts == []
    assert llm.calls == []  # no context -> no LLM call spent


def test_citations_parsed_into_page_numbers():
    hits = [make_hit(page=60), make_hit(page=61), make_hit(page=62)]
    llm = FakeLLM("Capex was $1,577 million [3M_2018_10K p.60]. See also [3M_2018_10K p.61, p.62].")
    result = generate("What was capex?", hits, llm)
    assert [(c.doc_id, c.page) for c in result.citations] == [
        ("3M_2018_10K", 60), ("3M_2018_10K", 61), ("3M_2018_10K", 62),
    ]


def test_citation_to_page_not_in_context_is_dropped():
    llm = FakeLLM("It was 5 [3M_2018_10K p.60] and 6 [3M_2018_10K p.999].")
    result = generate("q", [make_hit(page=60)], llm)
    assert [(c.doc_id, c.page) for c in result.citations] == [("3M_2018_10K", 60)]


def test_duplicate_citations_are_collapsed():
    llm = FakeLLM("A [3M_2018_10K p.60]. B [3M_2018_10K p.60].")
    result = generate("q", [make_hit(page=60)], llm)
    assert len(result.citations) == 1


# ---- citation formats the model really produces (measured on the saved eval answers) ----

from src.generation.grounded_answer import parse_citations  # noqa: E402

D = "3M_2018_10K"


def cites(answer, pages=(60, 61, 62), doc=D):
    return [(c.doc_id, c.page) for c in parse_citations(answer, [make_hit(doc_id=doc, page=p) for p in pages])]


@pytest.mark.parametrize(
    "answer",
    [
        "Capex was 5 【" + D + " p.60】.",               # full-width corner brackets (177 of ~410 in saved runs)
        "Capex was 5 ［" + D + " p.60］.",               # full-width square brackets
        "Capex was 5 [" + D + " p.60].",                    # narrow no-break space before the bracket
        "Capex was 5 [" + D + " p.60 ].",                          # space before the closing bracket
        "Capex was 5 [ " + D + " p.60].",                          # space after the opening bracket
        "Capex was 5 (" + D + " p.60).",                    # parentheses
        "Capex was 5 [" + D + " p. 60].",                          # space after "p."
        "Capex was 5 [" + D + " p60].",                            # no dot
        "| Capex | 5 | [" + D + " p.60] |",                       # inside a markdown table cell
        "**Capex 5** 【" + D + " p.60】",               # inside bold text
    ],
)
def test_citation_bracket_variants_are_parsed(answer):
    assert cites(answer) == [(D, 60)]


def test_page_list_in_one_bracket_with_fullwidth_brackets():
    assert cites("x 【" + D + " p.60, p.61】") == [(D, 60), (D, 61)]
    assert cites("x [" + D + " p.60; p.62]") == [(D, 60), (D, 62)]
    assert cites("x [" + D + " pp. 60, 61]") == [(D, 60), (D, 61)]


def test_page_range_expands_to_the_pages_that_were_in_context():
    # non-breaking hyphen (U+2011), en dash and a plain hyphen all appear in real answers
    for dash in ("‑", "–", "-"):
        assert cites("x [" + D + " p.60" + dash + "62]") == [(D, 60), (D, 61), (D, 62)]
    assert cites("x [" + D + " p.60-62]", pages=(60, 62)) == [(D, 60), (D, 62)]  # 61 was not retrieved


def test_huge_page_range_does_not_loop_or_invent_pages():
    assert cites("x [" + D + " p.1-99999999999]", pages=(60,)) == [(D, 60)]


def test_two_documents_in_one_bracket():
    a, b = "AMERICANEXPRESS_2022_10K", D
    hits = [make_hit(doc_id=b, page=60), make_hit(doc_id=a, page=96)]
    got = [(c.doc_id, c.page) for c in parse_citations(f"x [{b} p.60; {a} p.96]", hits)]
    assert got == [(b, 60), (a, 96)]


def test_mixed_formats_keep_order_and_collapse_duplicates():
    ans = "A 【" + D + " p.61】 B [" + D + " p.60] C 【" + D + " p.61】"
    assert cites(ans) == [(D, 61), (D, 60)]


def test_text_that_only_looks_like_a_citation_is_ignored():
    assert cites("See page p.60 of the filing, or [note p.60], or (p.61).") == []
    assert cites("Revenue was [" + D + "] in total, p.60") == []  # doc id without a page
    assert cites("x [OTHER_2018_10K p.60]") == []                  # doc not in context


def test_answer_text_is_not_rewritten_by_citation_parsing():
    raw = "Capex was 5 【" + D + " p.60】."
    result = generate("q", [make_hit(page=60)], FakeLLM(raw))
    assert result.answer == raw  # eval judges the text as produced; only `citations` changes
    assert [(c.doc_id, c.page) for c in result.citations] == [(D, 60)]


@pytest.mark.parametrize("raw", ["I don't know", "I don't know.", "  I don't know \n"])
def test_i_dont_know_answer_is_exact_with_no_citations(raw):
    result = generate("q", [make_hit()], FakeLLM(raw))
    assert result.answer == IDK == "I don't know"
    assert result.citations == []


def test_prompt_carries_context_question_and_rules():
    llm = FakeLLM("ok [3M_2018_10K p.60]")
    hit = make_hit(page=60, text="Purchases of PP&E (1,577)")
    result = generate("What was 3M capex in FY2018?", [hit], llm)
    call = llm.calls[0]
    assert "[3M_2018_10K p.60]" in call["prompt"]
    assert "Purchases of PP&E (1,577)" in call["prompt"]
    assert "What was 3M capex in FY2018?" in call["prompt"]
    for rule in ("only the context", "I don't know", "calculation", "[doc_id p.N]"):
        assert rule in (call["system"] or "") + call["prompt"]
    assert result.contexts == ["Purchases of PP&E (1,577)"]
    assert result.latency_ms["llm"] >= 0


# ---------------- error messages ----------------

def test_provider_error_message_is_surfaced_with_keys_redacted(cfg, tm):
    body = {"error": {"message": f"models/x is not found for API version v1beta (key {GEM_KEY})"}}
    t = FakeTransport(gemini=[HttpResponse(404, body)], groq=[HttpResponse(404, {})])
    with pytest.raises(LLMError) as exc:
        make_client(cfg, tm, t).complete("question")
    text = str(exc.value)
    assert "HTTP 404" in text and "models/x is not found" in text
    assert GEM_KEY not in text and "[REDACTED]" in text


def test_long_error_messages_are_truncated(cfg, tm):
    t = FakeTransport(gemini=[HttpResponse(400, {"error": {"message": "x" * 5000}})], groq=[groq_ok()])
    client = make_client(cfg, tm, t)
    client.complete("question")  # falls back to groq; failure text is only in the log
    t2 = FakeTransport(gemini=[HttpResponse(400, {"error": {"message": "x" * 5000}})], groq=[HttpResponse(400, {})])
    with pytest.raises(LLMError) as exc:
        make_client(cfg, tm, t2).complete("another question")
    assert len(str(exc.value)) < 1000


# ---------------- multiple keys per provider ----------------

G1, G2, G3 = "groq-key-one", "groq-key-two", "groq-key-three"
GROQ_ENVS = ["GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]
GROQ_KEYS = dict(zip(GROQ_ENVS, [G1, G2, G3]))


@pytest.fixture
def gcfg(cfg):
    """Groq only, three keys, no fallback provider (the production shape)."""
    cfg.llm.primary = _provider("groq", "groq-test", key_envs=GROQ_ENVS, daily_budget=10)
    cfg.llm.fallback = None
    return cfg


def bearer(call):
    return call["headers"]["Authorization"].removeprefix("Bearer ")


def groq_client(cfg, tm, t, keys=None, today="2026-11-10"):
    return make_client(cfg, tm, t, keys=GROQ_KEYS if keys is None else keys, today=today)


def write_budget(cfg, counts, today="2026-11-10"):
    path = Path(cfg.llm.cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "budget.json").write_text(json.dumps(
        {f"groq:{env}": {"date": today, "calls": n} for env, n in counts.items()}))


def test_429_rotates_to_next_key_immediately(gcfg, tm):
    t = FakeTransport(groq=[HttpResponse(429, {}), groq_ok("from key two")])
    out = groq_client(gcfg, tm, t).complete("question")
    assert out == "from key two"
    assert [bearer(c) for c in t.calls] == [G1, G2]
    assert tm.sleeps == []  # rotation is immediate, no backoff


def test_rate_limited_key_is_avoided_until_cooldown_expires(gcfg, tm):
    t = FakeTransport(groq=[HttpResponse(429, {}), groq_ok("a"), groq_ok("b"), groq_ok("c")])
    client = groq_client(gcfg, tm, t)
    client.complete("q1")  # key1 429 -> key2
    client.complete("q2")  # key1 still cooling -> key2
    tm.now += gcfg.llm.key_cooldown_s + 1
    client.complete("q3")  # cooldown over -> back to key1
    assert [bearer(c) for c in t.calls] == [G1, G2, G2, G1]


def test_all_keys_rate_limited_is_bounded_and_ends_in_clean_error(gcfg, tm):
    t = FakeTransport(groq=[HttpResponse(429, {})] * 20)
    with pytest.raises(LLMError):
        groq_client(gcfg, tm, t).complete("question")
    # 3 keys tried at once, then (1 + max_retries) backoff rounds; never an unbounded loop
    assert len(t.calls) == 5
    assert all(k not in str(LLMError) for k in GROQ_KEYS.values())


def test_key_at_budget_is_skipped_for_the_next_key(gcfg, tm):
    write_budget(gcfg, {"GROQ_API_KEY_1": 8})  # 80% of 10
    t = FakeTransport(groq=[groq_ok()])
    groq_client(gcfg, tm, t).complete("question")
    assert [bearer(c) for c in t.calls] == [G2]


def test_all_keys_over_budget_stops_with_budget_error(gcfg, tm):
    write_budget(gcfg, {"GROQ_API_KEY_1": 8, "GROQ_API_KEY_2": 9, "GROQ_API_KEY_3": 8})
    t = FakeTransport(groq=[groq_ok()])
    with pytest.raises(BudgetNearlyExhausted) as exc:
        groq_client(gcfg, tm, t).complete("question")
    assert t.calls == []
    assert "groq" in str(exc.value) and "GROQ_API_KEY_1" in str(exc.value)
    assert not any(k in str(exc.value) for k in GROQ_KEYS.values())


def test_budget_is_counted_per_key(gcfg, tm):
    t = FakeTransport(groq=[HttpResponse(429, {}), groq_ok()])
    client = groq_client(gcfg, tm, t)
    client.complete("question")
    data = json.loads((Path(gcfg.llm.cache_dir) / "budget.json").read_text())
    assert data["groq:GROQ_API_KEY_1"]["calls"] == 1
    assert data["groq:GROQ_API_KEY_2"]["calls"] == 1
    assert client.calls_today("groq") == 2
    assert client.calls_today("groq", "GROQ_API_KEY_2") == 1


def test_only_configured_keys_that_are_set_are_used(gcfg, tm):
    t = FakeTransport(groq=[groq_ok()])
    groq_client(gcfg, tm, t, keys={"GROQ_API_KEY_3": G3}).complete("question")
    assert [bearer(c) for c in t.calls] == [G3]


def test_no_fallback_provider_means_clean_error_when_groq_fails(gcfg, tm):
    t = FakeTransport(groq=[HttpResponse(500, {})] * 3)
    with pytest.raises(LLMError):
        groq_client(gcfg, tm, t, keys={"GROQ_API_KEY_1": G1}).complete("question")
    assert t.count("generativelanguage") == 0


def test_production_config_is_groq_only_with_three_keys():
    prod = load_config()
    client = LLMClient(prod, transport=FakeTransport(), api_keys={})
    assert [p.name for p in client.providers] == ["groq"]
    assert client.providers[0].key_envs == GROQ_ENVS


# ---------------- provider-suggested retry wait ----------------

from src.llm.llm_client import parse_retry_after  # noqa: E402


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Limit 8000, Used 6066. Please try again in 15.704999999s. Need more tokens?", 15.704999999),
        ("Please try again in 1m3.5s.", 63.5),
        ("Please try again in 850ms", 0.85),
        ("Please try again in 2h1m", 7260.0),
        ("quota exceeded", None),
        ("", None),
    ],
)
def test_parse_retry_after(message, expected):
    got = parse_retry_after(message)
    assert got == pytest.approx(expected) if expected is not None else got is None


def rate_limit_429(seconds):
    return HttpResponse(429, {"error": {"message": f"Rate limit reached (TPM). Please try again in {seconds}s. Need more tokens?"}})


def test_429_waits_the_provider_suggested_time_on_a_single_key(cfg, tm):
    t = FakeTransport(gemini=[rate_limit_429("15.7"), gemini_ok()])
    assert make_client(cfg, tm, t).complete("question") == "gemini answer"
    assert tm.sleeps == [pytest.approx(15.7)]  # not the 2 s default backoff


def test_suggested_wait_is_capped_by_config(cfg, tm):
    t = FakeTransport(gemini=[rate_limit_429("600"), gemini_ok()])
    make_client(cfg, tm, t).complete("question")
    assert tm.sleeps == [cfg.llm.max_retry_wait_s]


def test_short_hint_never_shortens_the_exponential_backoff(cfg, tm):
    t = FakeTransport(gemini=[rate_limit_429("0.1"), gemini_ok()])
    make_client(cfg, tm, t).complete("question")
    assert tm.sleeps == [cfg.llm.backoff_base_s]


# ---------------- truncated output / per-call token budget ----------------

def groq_truncated(text="partial {"):
    return HttpResponse(200, {"choices": [{"message": {"content": text}, "finish_reason": "length"}]})


def gemini_truncated(text="partial {"):
    return HttpResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "MAX_TOKENS"}]})


def test_truncated_groq_reply_is_an_error_and_is_not_cached(cfg, tm):
    t = FakeTransport(groq=[groq_truncated(), groq_ok("complete answer")])
    client = make_client(cfg, tm, t)
    with pytest.raises(LLMError) as exc:
        client.complete("question", only="groq")
    assert "truncated" in str(exc.value)
    assert client.complete("question", only="groq") == "complete answer"  # not served from cache
    assert t.count("groq.com") == 2


def test_truncated_gemini_reply_is_an_error_and_is_not_cached(cfg, tm):
    t = FakeTransport(gemini=[gemini_truncated(), gemini_ok("complete answer")])
    client = make_client(cfg, tm, t)
    with pytest.raises(LLMError):
        client.complete("question", only="gemini")
    assert client.complete("question", only="gemini") == "complete answer"


def test_truncated_reply_is_not_retried_with_the_same_budget(cfg, tm):
    t = FakeTransport(groq=[groq_truncated()])
    with pytest.raises(LLMError):
        make_client(cfg, tm, t).complete("question", only="groq")
    assert t.count("groq.com") == 1 and tm.sleeps == []


def test_per_call_max_output_tokens_reaches_both_payloads(cfg, tm):
    t = FakeTransport(gemini=[gemini_ok()], groq=[groq_ok()])
    client = make_client(cfg, tm, t)
    client.complete("p1", only="gemini", max_output_tokens=4096)
    client.complete("p2", only="groq", max_output_tokens=4096)
    assert t.calls[0]["payload"]["generationConfig"]["maxOutputTokens"] == 4096
    assert t.calls[1]["payload"]["max_tokens"] == 4096


def test_default_max_output_tokens_comes_from_config(cfg, tm):
    t = FakeTransport(groq=[groq_ok()])
    make_client(cfg, tm, t).complete("p", only="groq")
    assert t.calls[0]["payload"]["max_tokens"] == cfg.llm.max_output_tokens


# ---------------- daily TOKEN budget (Groq has 200,000 tokens/day per account) ----------------

def groq_tokens(total, text="groq answer"):
    return HttpResponse(200, {"choices": [{"message": {"content": text}}], "usage": {"total_tokens": total}})


def budget_data(cfg):
    return json.loads((Path(cfg.llm.cache_dir) / "budget.json").read_text())


def write_usage(cfg, entries, today="2026-11-10"):
    path = Path(cfg.llm.cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "budget.json").write_text(json.dumps({f"groq:{env}": {"date": today, **e} for env, e in entries.items()}))


@pytest.fixture
def tcfg(gcfg):
    gcfg.llm.primary.daily_budget = 1000
    gcfg.llm.primary.daily_token_budget = 1000  # 80% = 800 tokens
    return gcfg


def test_tokens_are_recorded_per_key_and_calls_are_kept(tcfg, tm):
    t = FakeTransport(groq=[groq_tokens(300), groq_tokens(250)])
    client = groq_client(tcfg, tm, t)
    client.complete("q1")
    client.complete("q2")
    entry = budget_data(tcfg)["groq:GROQ_API_KEY_1"]
    assert entry["tokens"] == 550 and entry["calls"] == 2
    assert client.tokens_today("groq") == 550
    assert client.tokens_today("groq", "GROQ_API_KEY_2") == 0


def test_gemini_usage_is_recorded_too(cfg, tm):
    body = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}], "usageMetadata": {"totalTokenCount": 77}}
    t = FakeTransport(gemini=[HttpResponse(200, body)])
    client = make_client(cfg, tm, t)
    client.complete("q")
    assert client.tokens_today("gemini") == 77


def test_missing_usage_is_not_an_error(tcfg, tm):
    t = FakeTransport(groq=[groq_ok()])  # no usage field
    groq_client(tcfg, tm, t).complete("q")
    assert budget_data(tcfg)["groq:GROQ_API_KEY_1"].get("tokens", 0) == 0


def test_key_at_80_percent_of_token_budget_is_skipped(tcfg, tm):
    write_usage(tcfg, {"GROQ_API_KEY_1": {"calls": 1, "tokens": 800}})
    t = FakeTransport(groq=[groq_ok()])
    groq_client(tcfg, tm, t).complete("q")
    assert [bearer(c) for c in t.calls] == [G2]


def test_all_keys_over_token_budget_stops_and_mentions_tokens(tcfg, tm):
    write_usage(tcfg, {e: {"calls": 1, "tokens": 900} for e in GROQ_ENVS})
    t = FakeTransport(groq=[groq_ok()])
    with pytest.raises(BudgetNearlyExhausted) as exc:
        groq_client(tcfg, tm, t).complete("q")
    assert t.calls == [] and "token" in str(exc.value).lower()


def test_token_budget_off_when_not_configured(gcfg, tm):
    write_usage(gcfg, {"GROQ_API_KEY_1": {"calls": 1, "tokens": 10**9}})
    t = FakeTransport(groq=[groq_ok()])
    groq_client(gcfg, tm, t).complete("q")
    assert [bearer(c) for c in t.calls] == [G1]


def test_tokens_reset_on_a_new_day(tcfg, tm):
    write_usage(tcfg, {"GROQ_API_KEY_1": {"calls": 1, "tokens": 900}}, today="2026-11-09")
    t = FakeTransport(groq=[groq_ok()])
    groq_client(tcfg, tm, t, today="2026-11-10").complete("q")
    assert [bearer(c) for c in t.calls] == [G1]


def test_cache_hit_consumes_no_tokens(tcfg, tm):
    t = FakeTransport(groq=[groq_tokens(100)])
    client = groq_client(tcfg, tm, t)
    client.complete("same")
    client.complete("same")
    assert client.tokens_today("groq") == 100


TPD_BODY = {"error": {"message": "Rate limit reached for model `x` in organization `org_1` service tier "
            "`on_demand` on tokens per day (TPD): Limit 200000, Used 199087, Requested 1951. "
            "Please try again in 7m28.416s."}}
TPM_BODY = {"error": {"message": "Rate limit reached for model `x` on tokens per minute (TPM): "
            "Limit 8000, Used 7900, Requested 500. Please try again in 3.2s."}}


def test_per_day_429_marks_the_key_done_for_the_day_and_rotates_without_sleeping(gcfg, tm):
    t = FakeTransport(groq=[HttpResponse(429, TPD_BODY), groq_ok("two"), groq_ok("three")])
    client = groq_client(gcfg, tm, t)
    assert client.complete("q1") == "two"
    tm.now += gcfg.llm.key_cooldown_s * 10  # cooldown long over: key 1 must STILL be skipped today
    client.complete("q2")
    assert [bearer(c) for c in t.calls] == [G1, G2, G2]
    assert tm.sleeps == []


def test_per_minute_429_does_not_retire_the_key(gcfg, tm):
    t = FakeTransport(groq=[HttpResponse(429, TPM_BODY), groq_ok("two"), groq_ok("three")])
    client = groq_client(gcfg, tm, t)
    client.complete("q1")
    tm.now += gcfg.llm.key_cooldown_s + 1
    client.complete("q2")
    assert [bearer(c) for c in t.calls] == [G1, G2, G1]  # back to key 1 after the cooldown


def test_per_day_429_on_every_key_ends_in_a_clean_budget_stop_without_retry_loops(gcfg, tm):
    t = FakeTransport(groq=[HttpResponse(429, TPD_BODY)] * 10)
    with pytest.raises(BudgetNearlyExhausted):
        groq_client(gcfg, tm, t).complete("q")
    assert len(t.calls) == 3 and tm.sleeps == []  # one try per key, no backoff waits
