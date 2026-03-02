# Phase 2: Progenitor compatibility spec (Software)

Progenitor Phase 2 applies the same “enhance to peak” idea to **non-ML software**: services, data pipelines, control logic, APIs. Input = software artifact + target platform → output = optimized build, config, or runtime profile.

**Status:** Phase 2 started; spec and first targets to be defined incrementally.

---

## Goal

- Take **compatible** software (single binary, service, pipeline, or config) and a **target platform** (OS, CPU, I/O, concurrency).
- Produce an **optimized** version or profile that measurably improves at least one of: latency, throughput, CPU use, I/O, or resource efficiency.
- No full replacement of the software; **enhance in place** (tuning, flags, config, optional selective changes) so legacy or certified stacks can stay, but run at peak.

---

## Scope (Phase 2)

### In scope (to be refined)

- **Compatible software types** (first slice TBD), e.g.:
  - Single-process CLI or service (e.g. C/C++, Go, Python).
  - Data-processing pipelines (batch or streaming; definition TBD).
  - Real-time or control-style loops (e.g. fixed-work-per-tick).
- **Target platforms:** Linux first (x86_64/ARM64); then other OS/runtime as needed.
- **Optimization levers:** Profiling-driven tuning, compiler/build flags, scheduling and concurrency, resource allocation (CPU affinity, memory), config tuning. Optional: small, targeted code or config rewrites with clear rollback.
- **Metrics:** Latency, throughput, CPU %, memory, I/O — before/after, reproducible.

### User control (no auto-blast)

- **You decide what gets improved.** Progenitor must not automatically “improve everything” at once. Pushing many levers at the same time (e.g. more workers + bigger buffers + higher limits + aggressive timeouts) can overload the system and crash the site or service. When testing on your own websites, changes are opt-in and controllable so you don't make them unusable.
- **Explicit opt-in per dimension or per lever.** You choose which improvements to apply: e.g. “only tune workers,” or “only latency,” or “workers + connection pool, nothing else.” Default is conservative: no change unless you enable a specific optimization or pass.
- **No surprise changes.** Every applied change is explicit and documented (what was tuned, before/after value). Rollback is possible (revert config, env, or build).
- **Safe-by-default.** If a lever has a known risk (e.g. raising workers can OOM), we document it and optionally cap or warn; we do not auto-apply aggressive values.

### Out of scope for Phase 2

- Full rewrites or new implementations of the application.
- ML model training or inference (that stays in Phase 1).
- Phase 3: infecting any device and boosting performance to 50×+ (separate scope).
- Every language, framework, and platform.

---

## Compatibility (to define)

A software artifact is **Progenitor-compatible for Phase 2** if (exact checklist TBD):

1. It is a clearly identified artifact: e.g. executable, service, pipeline definition, or config that we can run and measure.
2. We can run it on the target platform and collect metrics (latency, throughput, CPU, etc.) without changing its core behavior.
3. We have a defined, reversible way to apply enhancements (e.g. build flags, config, env, or a small patch set).
4. Enhancements are documented and reproducible.

**First targets:** To be chosen (e.g. one language/runtime, one “shape” of app: e.g. HTTP service, or batch pipeline, or control loop).

---

## Deliverables (Phase 2)

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | **Compatibility spec** | This document; expand with concrete software types, platforms, and “compatible” checklist. |
| 2 | **Progenitor software extension** | Entry point or API: software artifact + target platform → optimized artifact or profile (build/config/runtime). |
| 3 | **Optimization passes** | Concrete, automatic or guided optimizations (e.g. compiler flags, concurrency, config) applied where safe and documented. |
| 4 | **Benchmarks and metrics** | Before/after metrics; scripts to reproduce. Same principle as Phase 1: no hand-wavy “faster.” |
| 5 | **Docs and examples** | How to expose a software target to Progenitor; at least one end-to-end example. |

---

## Success criteria (Phase 2)

- For at least one supported software type and target platform, Progenitor produces an optimized version or profile that **measurably** improves at least one of: latency, throughput, CPU use, or memory, without changing correct behavior (or with documented, acceptable trade-offs).
- Process is **reproducible** and **documented**.

---

## Initial implementation

- **First target:** `python_http` — directory with `progenitor.yaml` + app (e.g. Flask/Gunicorn). Manifest defines `run_cmd` and `tune` levers (min/max/default).
- **First lever:** `workers` — opt-in via `--tune-workers`. Safe cap by CPU count and manifest max. Explicit value: `--workers N`.
- **CLI:** `progenitor enhance-software <artifact_dir> [--tune-workers] [--workers N]` writes `.env.progenitor`. No levers applied unless you pass `--tune-workers`.
- **Benchmark:** `python benchmarks/run_software.py --url <base_url> --repeat 50` — you start the server before/after; script measures latency and throughput.

---

## Next steps (immediate)

1. **Pick first software target:** e.g. one of: single-binary CLI/service (C/Go/Python), HTTP API, batch pipeline, or control loop. Document it in this spec.
2. **Define “artifact” and “run”:** How we point Progenitor at the software (path to binary + config? repo + build recipe? pipeline YAML?) and how we run it to collect baseline metrics.
3. **Implement minimal pipeline:** Load/identify artifact → apply one or two optimizations (e.g. compiler flags or config) → re-run → measure. Mirror Phase 1’s “minimal Progenitor” idea.
4. **Add one example and benchmark:** One end-to-end software example and a benchmark script that reports before/after.

---

## Version

- **Spec version:** 0.1  
- **Phase:** 2 (Software)  
- **Last updated:** 2026-03  
