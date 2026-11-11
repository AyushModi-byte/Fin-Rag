"""Public-API hardening: per-IP rate limit, daily global cap, API key, CORS, per-client request log."""
import json
import logging

import pytest
from fastapi.testclient import TestClient

from src.api.routes import create_app
from src.security.rate_limit import DailyCap, RateLimiter, client_ip, client_label
from tests.test_api import cfg_for, make_pipeline

Q = {"question": "3M FY2018 capex?"}
KEY = "owner-secret"


class Clock:
    def __init__(self, t=1_700_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def hcfg(tmp_path, **api):
    cfg = cfg_for(tmp_path)
    cfg.cache.enabled = False
    cfg.api.rate_limit.per_minute = 3
    cfg.api.rate_limit.per_day = 10
    cfg.api.daily_global_cap = 100
    cfg.api.usage_file = str(tmp_path / "usage.json")
    cfg.api.cors_origins = []
    cfg.api.require_api_key = False
    cfg.api.trusted_proxy_hops = 1
    for k, v in api.items():
        setattr(cfg.api, k, v)
    return cfg


def make_client(tmp_path, clock=None, env=None, **api):
    cfg = hcfg(tmp_path, **api)
    pipe, _, llm = make_pipeline(cfg)
    app = create_app(pipeline=pipe, cfg=cfg, clock=clock or Clock(), env=env or {})
    return TestClient(app), llm


def post(client, ip="1.1.1.1", body=Q, **headers):
    return client.post("/query", json=body, headers={"x-forwarded-for": ip, **headers})


# ---------- building blocks ----------


def test_limiter_blocks_over_per_minute_and_reports_retry_after():
    clock = Clock()
    lim = RateLimiter(per_minute=2, per_day=100, max_clients=100, clock=clock)
    assert lim.check("a").allowed and lim.check("a").allowed
    third = lim.check("a")
    assert not third.allowed and third.reason == "minute" and 1 <= third.retry_after <= 60
    assert lim.check("b").allowed  # another client is unaffected


def test_limiter_window_slides():
    clock = Clock()
    lim = RateLimiter(per_minute=2, per_day=100, max_clients=100, clock=clock)
    lim.check("a"); lim.check("a")
    assert not lim.check("a").allowed
    clock.t += 61
    assert lim.check("a").allowed


def test_limiter_blocks_over_per_day():
    clock = Clock()
    lim = RateLimiter(per_minute=100, per_day=3, max_clients=100, clock=clock)
    for _ in range(3):
        clock.t += 61
        assert lim.check("a").allowed
    clock.t += 61
    blocked = lim.check("a")
    assert not blocked.allowed and blocked.reason == "day" and blocked.retry_after > 60
    clock.t += 86_400
    assert lim.check("a").allowed


def test_blocked_requests_do_not_extend_the_window():
    clock = Clock()
    lim = RateLimiter(per_minute=1, per_day=100, max_clients=100, clock=clock)
    lim.check("a")
    for _ in range(20):
        clock.t += 2
        assert not lim.check("a").allowed
    clock.t += 30  # 70 s after the one accepted request
    assert lim.check("a").allowed


def test_limiter_memory_is_bounded():
    lim = RateLimiter(per_minute=1, per_day=1, max_clients=50, clock=Clock())
    for i in range(500):
        lim.check(f"ip{i}")
    assert lim.tracked() <= 50


def test_daily_cap_counts_persists_and_resets_next_day(tmp_path):
    clock, path = Clock(), tmp_path / "usage.json"
    cap = DailyCap(2, str(path), clock=clock)
    assert cap.try_acquire() and cap.try_acquire() and not cap.try_acquire()
    assert not DailyCap(2, str(path), clock=clock).try_acquire()  # a restart does not reset the day
    clock.t += 86_400
    assert DailyCap(2, str(path), clock=clock).try_acquire()


def test_daily_cap_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("{oops", encoding="utf-8")
    assert DailyCap(1, str(path), clock=Clock()).try_acquire()


def test_client_ip_takes_the_entry_added_by_the_trusted_proxy():
    # a client can prepend fake entries; only the last hop is added by our own proxy
    assert client_ip({"x-forwarded-for": "6.6.6.6, 9.9.9.9, 2.2.2.2"}, "10.0.0.1", hops=1) == "2.2.2.2"
    assert client_ip({"x-forwarded-for": "6.6.6.6, 9.9.9.9, 2.2.2.2"}, "10.0.0.1", hops=2) == "9.9.9.9"


def test_client_ip_ignores_the_header_when_no_proxy_is_trusted_or_it_is_missing():
    assert client_ip({"x-forwarded-for": "6.6.6.6"}, "10.0.0.1", hops=0) == "10.0.0.1"
    assert client_ip({}, "10.0.0.1", hops=1) == "10.0.0.1"
    assert client_ip({"x-forwarded-for": "6.6.6.6"}, "10.0.0.1", hops=3) == "10.0.0.1"  # fewer hops than trusted


def test_client_label_is_a_salted_hash_not_the_ip():
    a, b = client_label("1.2.3.4", "salt1"), client_label("1.2.3.4", "salt2")
    assert "1.2.3.4" not in a and a != b and a == client_label("1.2.3.4", "salt1") and len(a) == 12


# ---------- API: rate limit ----------


def test_rate_limit_returns_429_with_retry_after(tmp_path):
    client, llm = make_client(tmp_path)
    assert [post(client).status_code for _ in range(3)] == [200, 200, 200]
    r = post(client)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert "detail" in r.json() and len(llm.calls) == 3  # the blocked request never reached the LLM


def test_rate_limit_is_per_client_ip(tmp_path):
    client, _ = make_client(tmp_path)
    for _ in range(3):
        post(client, ip="1.1.1.1")
    assert post(client, ip="1.1.1.1").status_code == 429
    assert post(client, ip="2.2.2.2").status_code == 200


def test_spoofed_forwarded_for_entries_do_not_dodge_the_limit(tmp_path):
    client, _ = make_client(tmp_path)
    codes = [post(client, ip=f"{i}.{i}.{i}.{i}, 7.7.7.7").status_code for i in range(5)]
    assert codes == [200, 200, 200, 429, 429]  # always the same real client (last hop)


def test_invalid_requests_also_count_against_the_limit(tmp_path):
    client, _ = make_client(tmp_path)
    assert [post(client, body={"question": ""}).status_code for _ in range(3)] == [422, 422, 422]
    assert post(client, body=Q).status_code == 429


def test_health_is_never_rate_limited(tmp_path):
    client, _ = make_client(tmp_path)
    assert all(client.get("/health").status_code == 200 for _ in range(20))


def test_rate_limit_window_reopens(tmp_path):
    clock = Clock()
    client, _ = make_client(tmp_path, clock=clock)
    for _ in range(3):
        post(client)
    assert post(client).status_code == 429
    clock.t += 61
    assert post(client).status_code == 200


# ---------- API: global daily cap ----------


def test_global_cap_returns_503_and_429s_do_not_consume_it(tmp_path):
    client, llm = make_client(tmp_path, daily_global_cap=2)
    assert post(client, ip="1.1.1.1").status_code == 200
    assert post(client, ip="2.2.2.2").status_code == 200
    r = post(client, ip="3.3.3.3")
    assert r.status_code == 503 and "capacity" in r.json()["detail"].lower() and len(llm.calls) == 2
    assert int(r.headers["retry-after"]) >= 1


def test_rate_limited_requests_do_not_use_the_global_cap(tmp_path):
    client, _ = make_client(tmp_path, daily_global_cap=4)
    for _ in range(3):
        post(client, ip="1.1.1.1")
    for _ in range(10):
        assert post(client, ip="1.1.1.1").status_code == 429  # blocked by the per-IP limit, free
    assert post(client, ip="2.2.2.2").status_code == 200  # 4th and last unit of the global cap


def test_global_cap_resets_on_a_new_day(tmp_path):
    clock = Clock()
    client, _ = make_client(tmp_path, clock=clock, daily_global_cap=1)
    assert post(client).status_code == 200
    assert post(client, ip="2.2.2.2").status_code == 503
    clock.t += 86_400
    assert post(client, ip="3.3.3.3").status_code == 200


# ---------- API: key ----------


def test_open_mode_without_key_env_allows_anonymous(tmp_path):
    client, _ = make_client(tmp_path)
    assert post(client).status_code == 200


def test_require_api_key_rejects_missing_and_wrong_keys(tmp_path):
    client, llm = make_client(tmp_path, env={"FINRAG_API_KEY": KEY}, require_api_key=True)
    assert post(client).status_code == 401
    assert post(client, **{"x-api-key": "wrong"}).status_code == 401
    assert post(client, **{"x-api-key": KEY}).status_code == 200
    assert len(llm.calls) == 1


def test_unauthorised_requests_do_not_use_the_cap_or_the_limit(tmp_path):
    client, _ = make_client(tmp_path, env={"FINRAG_API_KEY": KEY}, require_api_key=True, daily_global_cap=1)
    for _ in range(10):
        assert post(client).status_code == 401
    assert post(client, **{"x-api-key": KEY}).status_code == 200


def test_require_api_key_without_a_configured_key_fails_closed(tmp_path):
    client, _ = make_client(tmp_path, env={}, require_api_key=True)
    assert post(client, **{"x-api-key": ""}).status_code == 401
    assert post(client).status_code == 401


def test_owner_key_skips_the_per_ip_limit_but_not_the_global_cap(tmp_path):
    client, _ = make_client(tmp_path, env={"FINRAG_API_KEY": KEY}, daily_global_cap=5)
    owner = [post(client, **{"x-api-key": KEY}).status_code for _ in range(5)]
    assert owner == [200] * 5
    assert post(client, **{"x-api-key": KEY}).status_code == 503  # global cap still applies


def test_anonymous_callers_are_still_limited_when_a_key_exists(tmp_path):
    client, _ = make_client(tmp_path, env={"FINRAG_API_KEY": KEY})
    assert [post(client).status_code for _ in range(4)] == [200, 200, 200, 429]
    assert post(client, **{"x-api-key": "wrong"}).status_code == 429  # a wrong key is just anonymous in open mode


# ---------- API: CORS ----------

ORIGIN = "https://finrag.vercel.app"


def preflight(client, origin):
    return client.options(
        "/query",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type,x-api-key"},
    )


def test_cors_allows_only_configured_origins(tmp_path):
    client, _ = make_client(tmp_path, cors_origins=[ORIGIN])
    ok = preflight(client, ORIGIN)
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == ORIGIN
    assert "POST" in ok.headers["access-control-allow-methods"]
    bad = preflight(client, "https://evil.example")
    assert "access-control-allow-origin" not in bad.headers


def test_cors_headers_on_the_real_response_and_on_a_429(tmp_path):
    client, _ = make_client(tmp_path, cors_origins=[ORIGIN])
    r = post(client, Origin=ORIGIN)
    assert r.headers["access-control-allow-origin"] == ORIGIN
    for _ in range(3):
        post(client)
    limited = post(client, Origin=ORIGIN)  # a browser must be able to read the 429, not just see a network error
    assert limited.status_code == 429 and limited.headers["access-control-allow-origin"] == ORIGIN
    assert "retry-after" in limited.headers["access-control-expose-headers"].lower()


def test_cors_preflight_is_not_rate_limited(tmp_path):
    client, _ = make_client(tmp_path, cors_origins=[ORIGIN])
    for _ in range(3):
        post(client)
    assert preflight(client, ORIGIN).status_code == 200


def test_no_cors_headers_when_no_origins_are_configured(tmp_path):
    client, _ = make_client(tmp_path)
    r = post(client, Origin=ORIGIN)
    assert "access-control-allow-origin" not in r.headers


def test_cors_origins_can_come_from_the_environment(tmp_path):
    client, _ = make_client(tmp_path, env={"FINRAG_CORS_ORIGINS": f" {ORIGIN} , https://b.example "})
    assert preflight(client, ORIGIN).headers["access-control-allow-origin"] == ORIGIN
    assert preflight(client, "https://b.example").headers["access-control-allow-origin"] == "https://b.example"


# ---------- API: per-client request log ----------


def request_lines(caplog):
    return [json.loads(r.getMessage()) for r in caplog.records if '"event": "request"' in r.getMessage()]


def test_request_log_has_client_label_status_and_latency_but_no_ip_or_question(tmp_path, caplog):
    client, _ = make_client(tmp_path)
    with caplog.at_level(logging.INFO, logger="finrag"):
        post(client, ip="203.0.113.77", body={"question": "3M FY2018 secret-marker capex?"})
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "203.0.113.77" not in text
    line = request_lines(caplog)[0]
    assert line["path"] == "/query" and line["status"] == 200 and line["ms"] >= 0
    assert len(line["client"]) == 12 and line["auth"] is False and line["limited"] is None
    assert "question" not in line


def test_request_log_marks_limited_and_authenticated_requests(tmp_path, caplog):
    client, _ = make_client(tmp_path, env={"FINRAG_API_KEY": KEY})
    with caplog.at_level(logging.INFO, logger="finrag"):
        for _ in range(4):
            post(client)
        post(client, **{"x-api-key": KEY})
    lines = request_lines(caplog)
    assert [x["status"] for x in lines] == [200, 200, 200, 429, 200]
    assert lines[3]["limited"] == "minute" and lines[4]["auth"] is True
    assert len({x["client"] for x in lines}) == 1


def test_same_client_gets_the_same_label_different_clients_differ(tmp_path, caplog):
    client, _ = make_client(tmp_path)
    with caplog.at_level(logging.INFO, logger="finrag"):
        post(client, ip="1.1.1.1"); post(client, ip="1.1.1.1"); post(client, ip="2.2.2.2")
    a, b, c = (x["client"] for x in request_lines(caplog))
    assert a == b != c


def test_health_requests_are_not_logged_as_client_requests(tmp_path, caplog):
    client, _ = make_client(tmp_path)
    with caplog.at_level(logging.INFO, logger="finrag"):
        client.get("/health")
    assert request_lines(caplog) == []
