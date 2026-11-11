"""Put the API online from this machine through a free Cloudflare quick tunnel (no account, no card).

    python -m src.jobs.go_live --origin https://<your-project>.vercel.app [--push] [--build]

What it does, in order:
  1. starts the Docker container with FINRAG_CORS_ORIGINS set to the UI's origin;
  2. starts `cloudflared tunnel --url http://localhost:8000` and reads the https URL it prints;
  3. checks that URL answers /health;
  4. writes the URL into frontend/config.js (`apiBase`), and with --push commits and pushes that file so Vercel
     redeploys the UI;
  5. keeps running. Ctrl+C stops the tunnel and the container.

The API is reachable only while this process, Docker and the PC are running. The quick-tunnel URL changes every run
(that is why step 4 exists). Cloudflare offers quick tunnels for experiments, with no uptime guarantee.
Needs `cloudflared` on PATH (Windows: `winget install Cloudflare.cloudflared`) and Docker.
"""
import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CONFIG_JS = Path("frontend/config.js")
USER_AGENT = "finrag-go-live/1"  # Cloudflare answers the default python-urllib agent with a 403

_TUNNEL_URL = re.compile(r"https://([a-z0-9-]+)\.trycloudflare\.com")
_API_BASE = re.compile(r'(apiBase:\s*")[^"]*(")')
_ALLOWED_API_BASE = re.compile(r"^(https://[a-z0-9-]+\.trycloudflare\.com|http://localhost:\d+)$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d+)?$")


def extract_tunnel_url(output: str) -> str | None:
    """The first https://<name>.trycloudflare.com in cloudflared's output. The api.* host that appears in
    error messages is the Cloudflare control plane, not our tunnel."""
    for m in _TUNNEL_URL.finditer(output):
        if m.group(1) != "api":
            return m.group(0)
    return None


def set_api_base(config_js: str, url: str) -> str:
    """Return config.js with apiBase set to `url`. Only a tunnel URL or localhost is accepted, so nothing that
    could break out of the JS string can be written."""
    if not _ALLOWED_API_BASE.match(url):
        raise ValueError(f"refusing to write {url!r}: expected https://<name>.trycloudflare.com or http://localhost:<port>")
    if not _API_BASE.search(config_js):
        raise ValueError("frontend/config.js has no apiBase setting to update")
    return _API_BASE.sub(lambda m: f"{m.group(1)}{url}{m.group(2)}", config_js, count=1)


def normalise_origin(value: str) -> str:
    """One browser origin (scheme://host[:port], no path), https unless it is localhost."""
    origin = value.strip().rstrip("/")
    if not _ORIGIN.match(origin):
        raise ValueError(f"{value!r} is not a single origin like https://my-project.vercel.app")
    if origin.startswith("http://") and not re.match(r"^http://(localhost|127\.0\.0\.1)(:\d+)?$", origin):
        raise ValueError("a non-local origin must use https")
    return origin


# ---------------------------------------------------------------------------------------------
# Orchestration (needs Docker, cloudflared and a network; exercised by hand, see docs/DEPLOYMENT.md)
# ---------------------------------------------------------------------------------------------
def _get(url: str, timeout: float = 5.0) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def _wait_health(url: str, seconds: float) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if _get(url) == 200:
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2)
    return False


def _run(cmd: list[str], env: dict | None = None) -> int:
    return subprocess.run(cmd, env=env).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--origin", required=True, help="origin of the web UI allowed to call the API (CORS)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--build", action="store_true", help="rebuild the Docker image first")
    ap.add_argument("--push", action="store_true", help="commit and push frontend/config.js so Vercel redeploys")
    args = ap.parse_args(argv)

    try:
        origin = normalise_origin(args.origin)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        print("error: cloudflared not found. Windows: winget install Cloudflare.cloudflared (then open a new terminal)",
              file=sys.stderr)
        return 2
    if not shutil.which("docker"):
        print("error: docker not found", file=sys.stderr)
        return 2

    import os

    env = {**os.environ, "FINRAG_CORS_ORIGINS": origin}
    up = ["docker", "compose", "up", "-d"] + (["--build"] if args.build else [])
    if _run(up, env) != 0:
        return 1
    local = f"http://localhost:{args.port}"
    tunnel = None
    try:
        if not _wait_health(f"{local}/health", 120):
            print("error: the container did not become healthy", file=sys.stderr)
            return 1
        print(f"container healthy, CORS origin {origin}")

        tunnel = subprocess.Popen([cloudflared, "tunnel", "--no-autoupdate", "--url", local],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines: list[str] = []
        threading.Thread(target=lambda: [lines.append(x) for x in tunnel.stdout], daemon=True).start()
        url = None
        deadline = time.time() + 60
        while time.time() < deadline and url is None:
            url = extract_tunnel_url("".join(lines))
            if tunnel.poll() is not None:
                break
            time.sleep(0.5)
        if url is None:
            print("error: no tunnel URL from cloudflared:\n" + "".join(lines[-15:]), file=sys.stderr)
            return 1
        if not _wait_health(f"{url}/health", 90):
            print(f"error: {url}/health did not answer through the tunnel", file=sys.stderr)
            return 1
        print(f"tunnel up: {url}")

        old = CONFIG_JS.read_text(encoding="utf-8")
        new = set_api_base(old, url)
        if new != old:
            CONFIG_JS.write_text(new, encoding="utf-8")
            print(f"updated {CONFIG_JS}")
            if args.push:
                _run(["git", "add", str(CONFIG_JS)])
                if _run(["git", "commit", "-m", "chore: point the UI at the current tunnel"]) == 0:
                    _run(["git", "push"])
                    print("pushed; Vercel will redeploy the UI in about a minute")
            else:
                print("not pushed (use --push, or commit frontend/config.js yourself so Vercel redeploys)")
        print(f"\nLIVE: API {url}  |  UI origin {origin}\nCtrl+C to stop the tunnel and the container.")
        while tunnel.poll() is None:
            time.sleep(1)
        print("cloudflared exited", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        if tunnel and tunnel.poll() is None:
            tunnel.terminate()
        _run(["docker", "compose", "down"])
        print("stopped")


if __name__ == "__main__":
    sys.exit(main())
