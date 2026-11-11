# Deployment guide (no cost, no card)

The web UI is static files on **Vercel** (free Hobby plan). The API runs **on your own PC in Docker** and is exposed
to the internet through a free **Cloudflare quick tunnel**. Nothing here costs money or asks for a payment card.

```
Browser ──HTTPS──> Vercel (static files: frontend/)
   │
   └──HTTPS, CORS──> https://<random>.trycloudflare.com ──tunnel──> your PC: Docker container (API + index + model)
                                                                          │
                                                                          └──HTTPS──> Groq API (free keys, rotated)
```

**What "live" means with this setup.** The API answers only while three things run: your PC, Docker, and the tunnel
(`go_live` job). The quick-tunnel URL changes every time you start it, so the job rewrites `frontend/config.js` and can push
it so Vercel redeploys. That is fine for demos, interviews and a portfolio link you switch on when needed. It is not a
24/7 service, and Cloudflare offers quick tunnels for experimentation, without an uptime guarantee.

> **Why not Hugging Face Spaces, Render or Vercel for the API?** Hugging Face's docs say "Gradio and Docker Spaces run on
> compute and require a paid plan to create (PRO for personal accounts)"; only Static Spaces are free (an earlier version
> of this guide wrongly said otherwise). Free tiers of "web service" hosts are about 512 MB, and the app needs roughly 2 GB
> (embedding model 0.64 GB plus a 230 MB index). Vercel's serverless functions cannot hold the model. Free always-on options
> that need a card (Oracle Always Free, Google Cloud Run) are possible later: the Docker image runs unchanged on them.
> Platform terms change; check current pages before relying on any of this.

## What you need

- Windows, macOS or Linux with Docker running, and about 8 GB of RAM free for it.
- The built index at `.cache/index/v2` and the Docker image (`docker compose build`; see the README for the index).
- Three Groq keys in `.env` (`GROQ_API_KEY_1..3`; copy `.env.example`).
- A free Vercel account, and the code on GitHub (done: `AyushModi-byte/Fin-Rag`).
- `cloudflared`: `winget install Cloudflare.cloudflared` on Windows (open a new terminal afterwards); other systems:
  Cloudflare's download page. No Cloudflare account is needed for quick tunnels.

## 1. Deploy the UI on Vercel (once)

1. vercel.com > Add New > Project > import `AyushModi-byte/Fin-Rag`.
2. Settings: **Root Directory `frontend`**, Framework Preset **Other**, no build command, no install command, output
   directory empty (the files are served as they are).
3. Deploy. Note the URL, for example `https://fin-rag.vercel.app`. The UI will load and show "Offline" until the API is up.
   `frontend/vercel.json` already sets a strict Content-Security-Policy that allows `connect-src` to
   `https://*.trycloudflare.com` and nothing else.

Vercel Hobby is for personal, non-commercial use.

## 2. Go live (every time you want the API online)

```bash
python -m src.jobs.go_live --origin https://fin-rag.vercel.app --push
```

It starts the container with `FINRAG_CORS_ORIGINS` set to that origin (so the browser may call the API), starts the
tunnel, waits until `<tunnel>/health` answers, writes the tunnel URL into `frontend/config.js`, and with `--push` commits
and pushes that file so Vercel redeploys the UI (about a minute). Leave the terminal open; **Ctrl+C** stops the tunnel and
the container. Add `--build` after changing code. Without `--push`, commit `frontend/config.js` yourself.

The first question after starting is slow (the embedding model loads, about 12 s here). The UI says so and polls `/health`.
Keep the PC from sleeping while it is live (Windows: Settings > System > Power > Screen and sleep > never, while plugged in).

## 3. Check it in a real browser

- [ ] The status pill in the header turns **Online**.
- [ ] An example question returns an answer with citation chips, and a chip opens the matching passage.
- [ ] Ask 6 questions within a minute: the sixth shows "Too many questions" with a countdown (limit: 5 per minute).
- [ ] The browser console shows no CORS or CSP errors.
- [ ] **Client IP check (important).** The per-IP limit uses the entry `api.trusted_proxy_hops` from the right of
  `X-Forwarded-For` (default 1; Cloudflare adds the client's address as the last entry). Send requests from two different
  networks (for example Wi-Fi and a phone on mobile data) and look at the `client` label in the container's
  `"event": "request"` lines (`docker compose logs finrag`): the two labels must **differ**. If every request carries the same
  label, all visitors share one rate limit: raise `trusted_proxy_hops` in `config.yaml` and restart with `--build`.

## Quotas and what happens when they run out

- Groq free tier: 200,000 tokens per day per key. A question with 8 passages in the prompt cost 1.3k to 2.4k tokens in the
  runs measured here; the cap assumes 2.5k to stay safe. With three keys and the 80% safety stop, about 480k tokens are
  usable per day.
- `api.daily_global_cap: 180` accepted questions per UTC day keeps the total under that (180 x 2.5k = 450k tokens). When it
  is reached the API answers 503 "Daily capacity reached" and the UI says so; it resets at 00:00 UTC. The counter is kept in
  `.cache/state/` on your PC, so restarting the container does not reset it.
- If the LLM client's own budget guard trips first, the API answers 503 and the UI shows "Model quota used up for today".
- Repeated questions are served from the semantic cache without a model call; the cache also lives in `.cache/state/`.
- The same Groq keys are used for your own eval runs and share the same per-key counters (`.cache/llm/budget.json`), so a
  big eval run leaves fewer tokens for visitors.
- Tune `rate_limit` and `daily_global_cap` in `config.yaml`, not in code.

## Risks specific to this setup

- Only the container's port is published through the tunnel, no inbound port is opened on your router, and the container
  runs as a non-root user. Still, you are exposing an API from your own machine: keep Docker updated, keep the rate limit
  and the cap on, and stop the job when you are not demoing.
- Anyone with the tunnel URL can call the API (within the limits). Set `FINRAG_API_KEY` and `api.require_api_key: true` for a
  private session, and run the UI with the key only on a page you control (a key in public browser code is public).
- The quick-tunnel URL is random and changes each run, but it is not secret: treat it as public.

## If you later get a paid or card-backed host

The image runs unchanged on any container host with about 2 GB of RAM. Hugging Face PRO: `python -m src.jobs.export_space
--out ../finrag-space` assembles the Space repo (index through Git LFS, front matter `sdk: docker`, `app_port: 8000`); put
the Groq keys and `FINRAG_CORS_ORIGINS` in the Space's secrets, and allow `https://*.hf.space` in
`frontend/vercel.json`'s `connect-src`. Check the client-IP step above on whatever proxy chain the host uses.

## Updating and rolling back

- Code or config change: `python -m src.jobs.go_live --origin <ui-origin> --build --push`.
- Roll back: `git revert` and push; Vercel redeploys the UI, and `--build` rebuilds the API image.
- The index is an artifact: do not rebuild or overwrite a version that is in use without keeping a copy of the old one.

## Things that cannot be verified from here

- Cloudflare quick-tunnel behaviour and limits at the time you run it, and the exact `X-Forwarded-For` chain (the client IP
  check above is the test).
- Vercel Hobby limits and terms at the time you deploy.
