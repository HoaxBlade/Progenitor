# Phase 2: Make the real website faster — no code changes

**Goal:** The customer does one thing (deploy Progenitor in front). We do the rest. Their real URL gets faster for everyone. No code changes, no config editing.

---

## How it works

1. **Progenitor runs in front** of their site as a proxy (cache + gzip + better headers).
2. **They point their domain** (or a subdomain) to where Progenitor is running.
3. Visitors still type the same URL. Traffic hits Progenitor first; we serve from cache when possible or fetch from their origin and optimize. **Same URL, faster for everyone.**

---

## What the customer does (one-time)

### Option A: Deploy with Docker

```bash
# Build
docker build -f Dockerfile.proxy -t progenitor-proxy .

# Run (set ORIGIN to their current site URL)
docker run -e ORIGIN=https://their-current-site.com -p 8080:8080 progenitor-proxy
```

Then they point their domain (e.g. CNAME or A record) to the host running this container. If their domain is `www.example.com`, they make `www.example.com` resolve to this server. Visitors get the faster, cached version. No code changes.

### Option B: Deploy on Cloud Run / Fly.io / any host

- Set **ORIGIN** to their site URL (e.g. `https://krishi-sahayogi.nielitbhubaneswar.in`).
- Set **PORT** if the platform requires it (e.g. Cloud Run sets `PORT=8080`).
- Deploy the container or run `progenitor serve` (e.g. `progenitor serve --origin https://their-site.com`).
- Point their domain to the deployment.

### Option C: Run locally (for testing)

```bash
export ORIGIN=https://their-site.com
progenitor serve
```

Then visit `http://localhost:8080` — same content, served through Progenitor (cache + compress).

---

## What we do (no work for the customer)

- **Cache** — Repeat requests for the same path are served from memory; no round-trip to their origin.
- **Compress** — If the origin doesn’t send gzip, we compress before sending to the visitor.
- **Cache headers** — We can add or adjust Cache-Control so browsers cache when safe.

The customer does not edit code or server config. They deploy once, set ORIGIN, point their domain. We make the real website faster.

---

## Summary

| Customer action | Result |
|-----------------|--------|
| Deploy Progenitor proxy (Docker / Cloud Run / VPS) | One deploy |
| Set ORIGIN to their current site URL | One env var |
| Point their domain to the deployment | One DNS change |
| **No code changes** | Same URL, faster for everyone |
