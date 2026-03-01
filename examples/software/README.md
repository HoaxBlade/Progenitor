# Phase 2: Software

Artifacts are directories containing a **progenitor.yaml** manifest. Progenitor only applies levers you explicitly enable (no auto-blast).

---

## All commands

### Inspect (see what the app returns)

| What you get | Command |
|--------------|---------|
| **Response headers** (status, content-type, etc.) | `curl -sI https://your-app.example.com/` |
| **First 100 lines of the page** (HTML body) | `curl -sL https://your-app.example.com/ \| head -100` |

### Measure (latency & throughput)

Run your server first, then:

```bash
python benchmarks/run_software.py --url https://your-app.example.com/ --repeat 50 --warmup 5
```

- `--repeat` = number of requests to measure (default 50).
- `--warmup` = requests to drop before counting (default 5).

### Enhance (apply Progenitor levers)

From the repo root:

| Goal | Command |
|------|---------|
| **Write env file (no levers yet)** | `progenitor enhance-software <artifact_dir>` |
| **Enable workers tuning** (safe cap by CPU count) | `progenitor enhance-software <artifact_dir> --tune-workers` |
| **Set workers to a specific number** | `progenitor enhance-software <artifact_dir> --tune-workers --workers N` |
| **Write env to a custom path** | `progenitor enhance-software <artifact_dir> --tune-workers --output-env /path/to/.env.progenitor` |

`<artifact_dir>` is the path to your project directory that contains a valid **progenitor.yaml**. After running, source the generated env file before starting your app (e.g. `source <artifact_dir>/.env.progenitor`).

### Typical workflow

1. **Baseline:** Start your app → run the benchmark (measure URL).
2. **Enhance:** `progenitor enhance-software <artifact_dir> --tune-workers` (and optionally `--workers N`).
3. **Apply:** `source <artifact_dir>/.env.progenitor` (or export the vars), then restart the app.
4. **Compare:** Run the benchmark again on the same URL; compare latency and throughput.

---

## Understanding the benchmark output

After running the benchmark you get:

- **Latency** — How long one request takes (in ms). Lower is better. *Average* = typical request; *Median (p50)* = half of requests are faster; *p99* = even the slowest 1% of requests.
- **Throughput** — How many requests per second the server can handle. Higher is better.
- **Summary** — One plain-English line (e.g. “One request takes about 766 ms; the server can handle about 1.3 requests per second”).
