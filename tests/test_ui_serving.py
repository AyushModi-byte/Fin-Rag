"""The API serves the web UI itself (one process, one origin) and puts security headers on every response."""
from fastapi.testclient import TestClient

from src.api.routes import create_app
from tests.test_api import cfg_for, make_pipeline


def make_ui(tmp_path):
    ui = tmp_path / "ui"
    (ui / "js").mkdir(parents=True)
    (ui / "index.html").write_text("<!doctype html><title>FinRag</title><p>hello ui</p>", encoding="utf-8")
    (ui / "js" / "app.js").write_text("export const x = 1;\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside the ui folder", encoding="utf-8")
    return ui


def client(tmp_path, frontend_dir="__default__"):
    cfg = cfg_for(tmp_path)
    cfg.api.frontend_dir = str(make_ui(tmp_path)) if frontend_dir == "__default__" else frontend_dir
    pipe, _, _ = make_pipeline(cfg)
    return TestClient(create_app(pipeline=pipe, cfg=cfg))


def test_root_serves_the_ui_and_static_assets(tmp_path):
    c = client(tmp_path)
    r = c.get("/")
    assert r.status_code == 200 and "hello ui" in r.text and r.headers["content-type"].startswith("text/html")
    js = c.get("/js/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]


def test_api_routes_still_win_over_the_static_mount(tmp_path):
    c = client(tmp_path)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/query", json={"question": "3M FY2018 capex?"})
    assert r.status_code == 200 and "answer" in r.json()


def test_paths_outside_the_ui_folder_are_not_reachable(tmp_path):
    c = client(tmp_path)
    for path in ["/../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt", "/js/../../secret.txt"]:
        r = c.get(path)
        assert "outside the ui folder" not in r.text, path


def test_unknown_static_path_is_404(tmp_path):
    assert client(tmp_path).get("/nope.html").status_code == 404


def test_no_ui_is_served_when_the_folder_is_missing_or_unset(tmp_path):
    assert client(tmp_path, frontend_dir=str(tmp_path / "does-not-exist")).get("/").status_code == 404
    assert client(tmp_path, frontend_dir="").get("/").status_code == 404
    assert client(tmp_path, frontend_dir=None).get("/health").status_code == 200


def test_security_headers_on_ui_and_api_responses(tmp_path):
    c = client(tmp_path)
    for r in (c.get("/"), c.get("/health"), c.post("/query", json={"question": "3M FY2018 capex?"})):
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["referrer-policy"] == "no-referrer"
        assert r.headers["x-frame-options"] == "DENY"
        csp = r.headers["content-security-policy"]
        assert "default-src 'none'" in csp and "script-src 'self'" in csp and "connect-src 'self'" in csp
        assert "unsafe-inline" not in csp and "unsafe-eval" not in csp


def test_security_headers_are_on_rate_limit_responses_too(tmp_path):
    cfg = cfg_for(tmp_path)
    cfg.api.frontend_dir = None
    cfg.api.rate_limit.per_minute = 1
    pipe, _, _ = make_pipeline(cfg)
    c = TestClient(create_app(pipeline=pipe, cfg=cfg))
    c.post("/query", json={"question": "3M FY2018 capex?"})
    r = c.post("/query", json={"question": "3M FY2018 capex?"})
    assert r.status_code == 429 and r.headers["x-content-type-options"] == "nosniff"


def test_interactive_api_docs_stay_usable(tmp_path):
    # Swagger UI loads its script from a CDN, which the strict CSP would block: docs are exempt from the CSP only
    r = client(tmp_path).get("/docs")
    assert r.status_code == 200 and "content-security-policy" not in r.headers
    assert r.headers["x-content-type-options"] == "nosniff"
