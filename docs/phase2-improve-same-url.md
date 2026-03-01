# Phase 2: How we improve the same URL (no second URL, no fraud)

**Goal:** Same website, same URL, better performance. Site owner does one thing once.

---

## Option A: Progenitor in front (proxy / CDN layer)

**In short:** The site owner points their **domain** at us instead of directly at their host (e.g. Vercel). We sit in front: we cache and compress, then forward to their real server. Visitors still type the same URL (e.g. krishi-sahayogi.nielitbhubaneswar.in); they hit us first, we serve fast (from cache when possible) or fetch from their server and optimize. Same URL, same brand, we’re just the layer in front.

**Is this “localhost”?** No. Option A is **not** localhost. It’s either:
- **Our servers:** We run the proxy on our infra. Site owner changes DNS (or reverse proxy) so their domain points to us; we forward to their origin. So it’s a **public** layer in front of their site.
- **Localhost** would be: they run a proxy on their own machine; only they see the improvement. That’s a different thing and not what Option A means when we say “Progenitor in front.”

**What we do:**
1. We run a proxy service (cache + compress) on our side (or give them something they can run on their infra).
2. Site owner does **one thing once:** point their domain to our proxy (e.g. DNS CNAME to us, or put our proxy in front in their hosting).
3. After that, the **same URL** is improved because all traffic goes through our layer.

---

## Option B: Something we provide on their site (script / integration)

**In short:** They add a small thing we give them (e.g. a script tag, or a serverless function / middleware). That code runs on **their** site (same URL, same domain) and does optimizations (e.g. better caching headers, compression, lazy-loading). They add it once; the same URL gets faster.

**What we do:**
1. We provide a script or integration (e.g. JS snippet, or config for their framework).
2. Site owner adds it to their app once (one line or one deploy).
3. Same URL, same domain; our code runs on their page or server and improves performance.

---

## Summary

| Option | What site owner does once | Result |
|--------|---------------------------|--------|
| **A**  | Point domain / traffic to our proxy (or our proxy in their stack) | Same URL, traffic goes through our cache/compress layer |
| **B**  | Add our script or integration to their site | Same URL, our code optimizes on their page/server |

Both keep the same URL and same brand; no second URL, no fraud. To be done later.
