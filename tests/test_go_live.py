"""go_live: the pure parts (tunnel URL parsing, config.js rewriting, origin validation)."""
import pytest

from src.jobs.go_live import extract_tunnel_url, normalise_origin, set_api_base

CF_LOG = """\
2026-11-11T10:00:00Z INF Thank you for trying Cloudflare Tunnel. Doing so, without a Cloudflare account, is a quick way to experiment.
2026-11-11T10:00:01Z INF Requesting new quick Tunnel on trycloudflare.com...
2026-11-11T10:00:03Z INF +--------------------------------------------------------------------------------------------+
2026-11-11T10:00:03Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2026-11-11T10:00:03Z INF |  https://quiet-river-demo-1234.trycloudflare.com                                           |
2026-11-11T10:00:03Z INF +--------------------------------------------------------------------------------------------+
"""

CONFIG_JS = """\
// Deployment settings.
window.FINRAG_CONFIG = {
  apiBase: "http://localhost:8000",
  maxQuestionChars: 2000,
};
"""


# ---------- tunnel URL ----------


def test_extracts_the_quick_tunnel_url_from_cloudflared_output():
    assert extract_tunnel_url(CF_LOG) == "https://quiet-river-demo-1234.trycloudflare.com"


def test_returns_none_until_the_url_appears():
    assert extract_tunnel_url("INF Requesting new quick Tunnel on trycloudflare.com...") is None
    assert extract_tunnel_url("") is None


def test_ignores_the_cloudflare_api_host_that_shows_up_in_errors():
    log = 'ERR failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": EOF\n'
    assert extract_tunnel_url(log) is None
    assert extract_tunnel_url(log + CF_LOG) == "https://quiet-river-demo-1234.trycloudflare.com"


def test_takes_the_first_tunnel_url_when_the_log_repeats_it():
    log = CF_LOG + "INF Registered tunnel connection ... https://other-name-5678.trycloudflare.com\n"
    assert extract_tunnel_url(log) == "https://quiet-river-demo-1234.trycloudflare.com"


# ---------- config.js ----------


def test_set_api_base_replaces_only_the_value():
    out = set_api_base(CONFIG_JS, "https://quiet-river-demo-1234.trycloudflare.com")
    assert 'apiBase: "https://quiet-river-demo-1234.trycloudflare.com",' in out
    assert "http://localhost:8000" not in out
    assert out.replace("https://quiet-river-demo-1234.trycloudflare.com", "http://localhost:8000") == CONFIG_JS


def test_set_api_base_is_idempotent():
    once = set_api_base(CONFIG_JS, "https://a-b-c.trycloudflare.com")
    assert set_api_base(once, "https://d-e-f.trycloudflare.com").count("apiBase") == 1


def test_set_api_base_refuses_a_url_that_could_break_out_of_the_js_string():
    for bad in ['https://x.trycloudflare.com", evil: "1', "https://x.trycloudflare.com/\n", "javascript:alert(1)",
                "http://plain-http.example.com", "https://x.example.com/path", ""]:
        with pytest.raises(ValueError):
            set_api_base(CONFIG_JS, bad)


def test_set_api_base_fails_loudly_if_the_config_has_no_apibase():
    with pytest.raises(ValueError):
        set_api_base("window.FINRAG_CONFIG = {};", "https://x-y.trycloudflare.com")


# ---------- CORS origin ----------


@pytest.mark.parametrize("given, want", [
    ("https://fin-rag.vercel.app", "https://fin-rag.vercel.app"),
    ("https://fin-rag.vercel.app/", "https://fin-rag.vercel.app"),
    ("  https://fin-rag.vercel.app  ", "https://fin-rag.vercel.app"),
    ("http://localhost:5173", "http://localhost:5173"),
])
def test_normalise_origin_accepts_an_origin(given, want):
    assert normalise_origin(given) == want


@pytest.mark.parametrize("bad", ["", "fin-rag.vercel.app", "https://fin-rag.vercel.app/some/path", "ftp://x.com",
                                 "https://a.com, https://b.com", "*", "https://a.com?x=1"])
def test_normalise_origin_rejects_anything_that_is_not_one_origin(bad):
    with pytest.raises(ValueError):
        normalise_origin(bad)
