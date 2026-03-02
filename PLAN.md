# Progenitor — Project Plan

**Vision:** A system that “enhances” compatible targets to their peak performance. Inspired by the Progenitor virus (Resident Evil): selective, transformative, applied in defence-relevant domains. Not biological — pure code.

**Order of attack:** ML algorithms → Software → Any device (infect and boost to 50×+).

---

## Phase 1: ML algorithms

**Goal:** Expose an ML model or training/inference pipeline to Progenitor; it pushes that algo to peak performance (speed, throughput, accuracy-efficiency trade-off) on the given hardware.

### Scope (Phase 1)

- **In scope**
  - Inference: take a trained model (or pipeline) and optimize it for peak inference (latency, throughput, memory).
  - Training (optional for v1): hyperparameter/search and training-loop optimizations to reach “peak” training efficiency.
  - Support a small set of “compatible” frameworks and hardware targets first (e.g. PyTorch/ONNX, CPU + one GPU/NPU type).
  - Clear definition of “compatibility”: which model types, formats, and runtimes we support (the “host” for the virus).

- **Out of scope for Phase 1**
  - General-purpose software or physical machine control.
  - Every framework and every hardware target.

### Deliverables (Phase 1)

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | **Compatibility spec** | Document: which model formats, frameworks, and hardware are “Progenitor-compatible” for Phase 1. |
| 2 | **Progenitor core (ML)** | A single entry point or API: “feed model/pipeline + target hardware → get optimized artifact or runtime.” |
| 3 | **Optimization passes** | Concrete optimizations (e.g. quantization, pruning, graph/op fusion, kernel selection, memory layout) applied automatically where safe. |
| 4 | **Benchmarks & metrics** | Before/after metrics: latency, throughput, memory, and (if applicable) accuracy. Scripts to reproduce. |
| 5 | **Docs & examples** | How to “expose” a model to Progenitor; one or two end-to-end examples (e.g. image classifier, small LLM or embedding model). |

### Success criteria (Phase 1)

- Given a supported model and target device, Progenitor produces an optimized version that measurably improves at least one of: latency, throughput, or memory, with acceptable accuracy loss (if any).
- Process is reproducible and documented.

---

## Phase 1.5: Next level — pruning + sparse (same model, 5–15× target)

**Goal:** Same ONNX model, same graph; many weights set to zero. Run with sparse inference so we get **5–15×** on the same CPU (no GPU, no replacement model).

### Why this phase

- Graph opts + INT8 give ~1.05–2.7× depending on model and CPU; ResNet-50 on some Macs gets ~1.05× (graph-only) or slower with INT8.
- To reach a real “next level” speedup (5–15×) on the **same** model and CPU we need to **do less work per inference** — i.e. prune weights to zero and run sparse.

### Scope (Phase 1.5)

- **In scope**
  - **Pruning pass:** Load ONNX → identify weight initializers (Conv, MatMul, Gemm) → zero out a chosen fraction (e.g. 80–90%) of weights (e.g. smallest magnitude) → save ONNX. Same graph, same op list; only initializer values change.
  - **Sparse inference:** Run the pruned model so that zero weights are skipped (sparse kernels or sparse-aware runtime). If ONNX Runtime or another backend has sparse support, use it; otherwise document and measure “pruned but dense run” (smaller model, accuracy check) until sparse path exists.
  - **CLI/API:** e.g. `progenitor enhance model.onnx --prune 0.9` produces pruned ONNX; benchmark compares original vs pruned (same device).
  - **Accuracy:** Pruning can reduce accuracy. Document and optionally add a small validation step (e.g. run N samples, compare outputs or metrics before/after).

- **Out of scope for Phase 1.5**
  - Changing model architecture or replacing with a different model (no surrogate).
  - GPU-only optimizations; focus is same CPU.

### Deliverables (Phase 1.5)

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | **Pruning pass** | Implement in `progenitor/optimizations/`: load ONNX, prune initializers by magnitude (configurable ratio), write back ONNX. Same graph, sparse weights. |
| 2 | **Integration** | Wire `--prune` (or similar) into `enhance()` and CLI; output is pruned ONNX. |
| 3 | **Benchmark** | Benchmark script runs “before” = original, “after” = pruned; report speedup. If sparse runtime is used, we expect 5–15×; if dense run, we at least have pruned model and baseline. |
| 4 | **Sparse path** | Investigate and use ONNX Runtime sparse execution and/or other backends (e.g. oneDNN sparse, OpenVINO) so pruned model actually runs faster. Document options and limitations per platform. |
| 5 | **Docs** | Update compatibility spec and README: when to use `--prune`, expected speedup range, accuracy trade-off. |

### Success criteria (Phase 1.5)

- Pruned ONNX is produced from the same model (same graph structure); no replacement model.
- On at least one supported setup (e.g. ResNet-50 on Linux with sparse backend), benchmark shows **≥ 5×** speedup (original vs pruned, same CPU).
- Process is reproducible and documented; accuracy impact is stated or measurable.

### Implementation order

1. **Pruning pass:** Implement weight pruning (magnitude-based, configurable sparsity). Save pruned ONNX. No new runtime yet.
2. **Benchmark + CLI:** Add `--prune` to enhance and benchmark; measure “before” vs “after” (pruned may run dense initially — still validates pipeline and model size/accuracy).
3. **Sparse execution:** Add or integrate sparse inference (ORT sparse, oneDNN, or other) so pruned model runs faster; re-run benchmark to confirm 5–15× where supported.
4. **Docs and compatibility:** Document pruning in `docs/compatibility-phase1.md`; note CPU/backend requirements for sparse speedup.

---

## Phase 2: Software

**Goal:** Apply Progenitor to “software” — non-ML applications and services (e.g. control logic, data pipelines, APIs). Enhance their performance to peak on the target platform (CPU, I/O, concurrency, etc.).

### Scope (Phase 2) — high level

- Identify “compatible” software types: e.g. real-time control loops, data-processing pipelines, specific languages/runtimes.
- Progenitor analyzes and applies optimizations: profiling-driven tuning, compiler flags, scheduling, resource allocation, maybe selective rewriting or config tuning.
- Defence angle: legacy or certified software that cannot be fully replaced but can be “enhanced.”

### Deliverables (Phase 2)

- **Compatibility spec** for software targets → [docs/compatibility-phase2.md](docs/compatibility-phase2.md).
- **Progenitor extension:** input = software artifact + target platform → output = optimized build/config or runtime profile.
- **Benchmarks and before/after metrics** (latency, throughput, CPU, memory — reproducible).

### Phase 2 next steps (immediate)

1. **Pick first software target:** One concrete type (e.g. single-binary CLI/service in C/Go/Python, or HTTP API, or batch pipeline). Document in compatibility-phase2.md.
2. **Define artifact and run:** How we point at the software (binary + config, repo + build, pipeline YAML) and how we run it to collect baseline metrics.
3. **Minimal pipeline:** Identify artifact → apply one or two optimizations (e.g. compiler flags, config) → re-run → measure. One entry point or API for “enhance software.”
4. **One example + benchmark:** One end-to-end software example and a benchmark script that reports before/after.

---

## Phase 3: Any device — infect and boost to peak (50× or more)

**Goal:** Progenitor can **infect any type of device** and boost that device’s performance to **50× or more**. One strain, any compatible device: same device, enhanced to its absolute peak. This is the final phase.

### First supported device types (current limitations)

- For now we limit to **phones and PCs**. More device types (drones, servers, vehicles, IoT, etc.) will be added later. The vision remains “any device.”

### How it runs and consent

- Progenitor runs as a **background service or daemon**, not as a visible “app” — clearer and more trustworthy.
- We only run on a device when the **user has explicitly asked** to enhance that device (e.g. “enhance this phone”). That request **is** the permission; we do not add a separate permission hassle. No infection without the user saying to do it.

### How we reach the device (no plug, no link; you have control)

We avoid: (1) plugging the device into your laptop, (2) sending the customer a link, (3) a welcome page or anything the customer has to tap. Preferred approach:

- **Same-network, you control:** The customer is on **your** network (e.g. they come to your shop and connect to your WiFi). **You** have control — not the customer. They tell you “enhance my device”; you do it from your side. No welcome page, no captive portal, no tap from them.
- **You trigger enhancement:** From your infra on that network (e.g. a server or tool you run), you target their device over the network and run Progenitor. They asked; you execute. What runs on their device (profile, agent, etc.) is still TBD; the **model** is: they’re on your network → they say “enhance this” → you apply enhancement to their device from your side.
- **No customer control flow:** The customer does not get a page to click or an app to approve; they just asked you. You’re the one who initiates and controls the enhancement.

### Scope (Phase 3) — high level

- Progenitor targets **any type of device** (phones, PCs first; then drones, servers, vehicles, IoT, etc.) that meets compatibility.
- Once it infects a device, it drives that device’s performance to a massive uplift (50× or more, where achievable).
- How we “infect” and what we tune (OS, config, resources, etc.) will depend on the device type; the outcome is always: **same device, peak performance**.
- Safety and certification constraints must be explicit (read-only analysis first, or opt-in changes with rollback).

### Deliverables (Phase 3) — to detail later

- Compatibility spec: which device types Progenitor can infect and how (starting with phones and PCs).
- Progenitor extension: infect a device → measure baseline → apply enhancements → measure again (target 50×+ where possible).
- Measurable “peak” metrics per device type (throughput, latency, battery, mission completion, etc.).

---

## Repo structure (suggested)

Keep it flat at first; split when needed.

```
Progenitor/
├── README.md
├── PLAN.md
├── LICENSE
├── docs/              # Compatibility specs, design notes
├── progenitor/        # Core library (Phase 1: ML first)
├── optimizations/     # Optimization passes (ML, then software, then control)
├── benchmarks/        # Scripts and configs for before/after runs
├── examples/          # Example models, pipelines, and usage
└── tests/
```

You can add `software/` and `devices/` (or similar) under `progenitor` when you start Phase 2 and 3.

---

## Principles

- **Compatibility over universality:** We define and document “compatible hosts”; we don’t promise to enhance everything.
- **Measurable peak:** Every phase has clear metrics (latency, throughput, accuracy, mission KPIs). No hand-wavy “faster.”
- **Incremental:** One framework, one hardware target, one model type for Phase 1 — then expand.
- **Defence-minded:** Logging, reproducibility, and (later) safety and explainability of changes.

---

## Next steps (immediate)

1. Lock Phase 1 scope: pick one framework (e.g. PyTorch or ONNX) and one target (e.g. x86 CPU or one GPU).
2. Write the Phase 1 compatibility spec in `docs/`.
3. Implement the minimal Progenitor ML pipeline: load model → apply one or two optimizations → export/run → measure.
4. Add one end-to-end example and a benchmark script.

After that, iterate on more optimizations and more compatible targets before moving to Phase 2.
