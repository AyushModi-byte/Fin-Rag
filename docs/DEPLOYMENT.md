# Deployment notes

**FinRag is not deployed, by decision (2026-11-11).** It runs locally and in Docker, and that is the supported way to use
it (see the README). A free public host was investigated and none fits, so a live demo was dropped rather than paying for one.
This page records what was checked, so nobody repeats the search, and what hosting it would take if that changes.

## What the app needs from a host

| Need | Value |
|---|---|
| RAM | about 2 GB (embedding model 0.64 GB, index 230 MB, ONNX runtime; the Docker image is 3.3 GB) |
| CPU | 2 vCPU is enough; one question takes about 2 to 3 s plus the LLM call |
| Port | 8000 (the API serves the web UI from the same origin) |
| Secrets | `GROQ_API_KEY_1..3` (free Groq keys, one is enough) |
| State | `.cache/state` (semantic cache and daily-cap counter) and `.cache/llm` (LLM response cache and per-key budget counters); losing them is harmless but resets the daily cap and the cached answers |
| Inbound | HTTPS to port 8000; the app needs outbound HTTPS to the Groq API |

## What was checked (all dated 2026-11-11; platform terms change)

| Host | Result |
|---|---|
| **Hugging Face Spaces, Docker SDK** | Not free. The docs state that "Gradio and Docker Spaces run on compute and require a paid plan to create (PRO for personal accounts)"; only Static Spaces are free. CPU Basic (2 vCPU, 16 GB) would fit. An earlier version of this guide wrongly called it free. |
| **Render, Koyeb and similar free web services** | About 512 MB RAM: too small for the embedding model. |
| **Vercel (serverless)** | Functions cannot hold the model and index. It would only host the static UI, and the UI needs the API. |
| **Google Cloud Run** | Possible. Free monthly quota (180,000 vCPU-seconds, 360,000 GiB-seconds, 2M requests) covers this app, but it needs a billing account with a card, scales to zero (cold start of 30 to 60 s while the model loads), and the 3.3 GB image exceeds the free 0.5 GB registry allowance (about $0.30 a month unless the image is shrunk). The image runs unchanged. |
| **Oracle Cloud Always Free (ARM VM)** | Possible. Up to 4 OCPU and 24 GB RAM, always on, but needs a card for verification (prepaid and virtual cards are often rejected), capacity is sometimes unavailable, and it needs an ARM image and HTTPS set up by hand. |
| **Modal (Starter plan, $30 free credits a month)** | Possible with a small wrapper around the app; the card requirement could not be confirmed. |
| **Your own machine through a free Cloudflare quick tunnel** | Works with no account or card, but is live only while the machine, Docker and the tunnel run, and the URL changes every run. Judged not worth keeping as project code. |

Sources: the Hugging Face Hub docs pages "Spaces overview" and "Docker Spaces"; Google's Cloud Run pricing page; Oracle's
Free Tier documentation; Modal's pricing page (third-party summaries only for its card requirement).

## If you host it anyway

1. Build the image where the index exists: `docker compose build` (the index is baked in; it is not in git, see the README).
2. Run it with the Groq keys as environment variables or secrets, and publish port 8000 over HTTPS. The UI is served by the
   API at `/`, so no CORS setup is needed unless the UI is hosted elsewhere (`api.cors_origins` / `FINRAG_CORS_ORIGINS`).
3. Keep the abuse protection on (`api.rate_limit`, `api.daily_global_cap`; they protect the free Groq quota) and set
   `FINRAG_API_KEY` with `api.require_api_key: true` if the instance should not be public.
4. **Check `api.trusted_proxy_hops` on the real platform.** The per-IP limit takes the entry that many positions from the
   right of `X-Forwarded-For`. Send requests from two different networks and look at the `client` label in the
   `"event": "request"` log lines: the labels must differ. If every request has the same label, all visitors share one
   limit and the number of proxy hops is higher than configured.
5. The free Groq quota is the real capacity limit: 200,000 tokens per day per key, about 1.3k to 2.4k tokens per question with
   8 passages. `daily_global_cap: 180` assumes three keys and 2.5k tokens per question; lower it if you run fewer keys.
