# Deployment guide (free tier only)

> **Correction (2026-11-11).** An earlier version of this guide said a Docker Space on Hugging Face's free CPU tier
> would host the API. That was wrong: Hugging Face's docs now say "Gradio and Docker Spaces run on compute and require a
> paid plan to create (PRO for personal accounts)"; only **Static** Spaces are free. The backend host for a no-cost
> deployment is therefore undecided, and the Hugging Face steps below apply only if you take a PRO plan. The Vercel
> (frontend) part is unaffected. Candidate free hosts and their trade-offs are listed in the README's deployment section.

Original target: the API on a **Hugging Face Space** (Docker) and the web UI on **Vercel** (free Hobby plan).
Nothing in this repository has been deployed yet: the steps below were prepared and the image they produce was built
and started locally, but the platform steps (accounts, pushing, secrets) have not been run.

```
Browser ──HTTPS──> Vercel (static files: frontend/)
   │
   └──HTTPS, CORS──> Hugging Face Space (Docker: FastAPI + index v2 + embedding model)
                          │
                          └──HTTPS──> Groq API (3 free keys, rotated)
```

## Why these two, and why not Render

| Platform | Role | Why |
|---|---|---|
| Hugging Face Spaces (Docker) | API | **Needs a paid plan (PRO).** Its hardware fits (16 GB RAM, 2 vCPU on CPU Basic), but creating a Docker Space is not free. Free instances on most "web service" hosts (for example Render's free tier, about 512 MB) are too small for the app's roughly 2 GB. |
| Vercel (static) | UI | The UI is plain files. Vercel's serverless functions cannot hold the model or index, so Vercel is not an option for the API. |

Platform limits change. Before you start, check the current free-tier pages for: Hugging Face Spaces hardware and
idle-sleep policy, and Vercel Hobby (it is for personal, non-commercial use).

## What you need

- A Hugging Face account and a Vercel account (both free), and `git` with **Git LFS** installed (`git lfs install`).
- The three Groq keys (`GROQ_API_KEY_1..3`).
- The built index at `.cache/index/v2` (it is not in git; building it takes about 1.6 h of CPU, see the README).

## 1. Backend on Hugging Face Spaces

1. **Create the Space.** huggingface.co > New Space > SDK **Docker** > hardware **CPU basic (free)** > **Public**
   (a browser must be able to reach it). Name it `finrag`. Its URL will be `https://<user>-finrag.hf.space`.
2. **Assemble the Space repo** from this repository (offline, no network):
   ```bash
   python -m src.jobs.export_space --out ../finrag-space
   ```
   This copies only what the image needs (Dockerfile, `config.yaml`, `src/`, `data/eval/docs.json`, the v2 index
   without its embedding shards), writes the Space README with the front matter (`sdk: docker`, `app_port: 8000`) and a
   `.gitattributes` that puts the index in Git LFS. It never copies `.env`, caches, tests, raw data or the frontend, and
   refuses to write into a non-empty directory.
3. **Push it.**
   ```bash
   cd ../finrag-space
   git init && git lfs install
   git remote add origin https://huggingface.co/spaces/<user>/finrag
   git add -A && git commit -m "FinRag API"
   git push origin HEAD:main
   ```
   The first push uploads about 240 MB. The Space then builds the image (the embedding model, 0.64 GB, is downloaded
   during the build) and starts it.
4. **Add secrets** (Space > Settings > Variables and secrets):

   | Name | Value |
   |---|---|
   | `GROQ_API_KEY_1`, `_2`, `_3` | your keys (secrets, not variables) |
   | `FINRAG_CORS_ORIGINS` | `https://<your-project>.vercel.app` (add the exact Vercel URL after step 2 of the next section; no trailing slash) |
   | `FINRAG_API_KEY` | optional owner key, so you can test without the per-IP limit |

5. **Smoke test the API:**
   ```bash
   curl https://<user>-finrag.hf.space/health
   curl -X POST https://<user>-finrag.hf.space/query -H "Content-Type: application/json" \
        -d '{"question": "What was Adobe total revenue in fiscal year 2022?"}'
   ```
   The first request after a deploy or an idle period is slow (model load, about 12 s locally, more on a cold host).

## 2. Frontend on Vercel

1. Edit `frontend/config.js`: set `apiBase` to `https://<user>-finrag.hf.space` (no trailing slash). Commit and push.
2. Vercel > Add New Project > import the GitHub repository. Settings: **Root Directory `frontend`**,
   Framework Preset **Other**, no build command, no install command (there are no dependencies).
3. Deploy. Copy the resulting URL into the Space secret `FINRAG_CORS_ORIGINS` (step 4 above) and let the Space restart.
4. `frontend/vercel.json` sets a strict Content-Security-Policy (`connect-src 'self' https://*.hf.space`), HSTS and
   other headers. If you host the API on another domain, add it to `connect-src` there.

## 3. Check it in a real browser

- [ ] The status pill in the header turns **Online** (the page polls `/health` and wakes a sleeping Space).
- [ ] One of the example questions returns an answer with citation chips, and a chip opens the matching passage.
- [ ] Ask 6 questions in a minute: the sixth shows "Too many questions" with a countdown (limit: 5 per minute).
- [ ] Browser console: no CORS or CSP errors.
- [ ] **Client IP check (important).** The per-IP limit uses the entry `api.trusted_proxy_hops` from the right of
  `X-Forwarded-For` (default 1). Open the Space logs and send a request from two different networks (for example Wi-Fi
  and a phone on mobile data). The `client` label in the `"event": "request"` lines must **differ** between the two. If
  every request shows the same label, the platform's proxy chain has more hops than configured and all visitors share
  one rate limit: raise `trusted_proxy_hops` in `config.yaml` of the Space repo and push again. If a client can pick its
  own label by sending a forged `X-Forwarded-For`, it is set too low: this is covered by tests, but only the real platform
  can confirm the hop count.

## Quotas and what happens when they run out

- Groq free tier: 200,000 tokens per day per key. A question with 8 passages in the prompt cost 1.3k to 2.4k tokens in the
  runs measured here; the cap assumes 2.5k to stay safe. With three keys and the 80% safety stop, about 480k tokens are usable per day.
- `api.daily_global_cap: 180` accepted questions per UTC day keeps the total under that (180 x 2.5k = 450k tokens). When it is
  reached the API answers 503 "Daily capacity reached" and the UI says so; it resets at 00:00 UTC.
- If the LLM client's own budget guard trips first, the API answers 503 and the UI shows "Model quota used up for today".
- Repeated questions are served from the semantic cache without a model call, but the cache is in memory/disk of the
  container and is lost whenever the Space restarts (free Spaces have no persistent disk).
- Tune `rate_limit` and `daily_global_cap` in `config.yaml`, not in code.

## Updating and rolling back

- Code or config change: re-run the export into an empty directory, copy over (or use `git` in the existing Space repo),
  commit, push. The Space rebuilds. A faster path for config-only edits is editing the file in the Space repo directly.
- Roll back: `git revert` in the Space repo and push, or restore a previous commit in the Space's history.
- The index is an artifact: a new index version means a new export with `--index-version`, and `config.yaml` must select it.
  Do not rebuild or overwrite a version that is already deployed without a copy of the old one.

## Things that cannot be verified from here

- The exact HF Space RAM, build time limits and idle-sleep behaviour, and whether the 240 MB LFS push needs a write token.
- The platform's proxy hop count (see the client IP check above).
- Vercel Hobby limits at the time you deploy.
