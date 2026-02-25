# Progenitor — Project Plan

**Vision:** A system that “enhances” compatible targets to their peak performance. Inspired by the Progenitor virus (Resident Evil): selective, transformative, applied in defence-relevant domains. Not biological — pure code.

**Order of attack:** ML algorithms → Software → Machines (physical systems).

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

## Phase 2: Software

**Goal:** Apply Progenitor to “software” — non-ML applications and services (e.g. control logic, data pipelines, APIs). Enhance their performance to peak on the target platform (CPU, I/O, concurrency, etc.).

### Scope (Phase 2) — high level

- Identify “compatible” software types: e.g. real-time control loops, data-processing pipelines, specific languages/runtimes.
- Progenitor analyzes and applies optimizations: profiling-driven tuning, compiler flags, scheduling, resource allocation, maybe selective rewriting or config tuning.
- Defence angle: legacy or certified software that cannot be fully replaced but can be “enhanced.”

### Deliverables (Phase 2) — to detail later

- Compatibility spec for software targets.
- Progenitor extension: input = software artifact + target platform → output = optimized build/config or runtime profile.
- Benchmarks and before/after metrics.

---

## Phase 3: Machines (physical systems)

**Goal:** Progenitor interacts with software that controls physical machines (drones, vehicles, sensors, actuators). Enhance that control stack so the machine operates at peak (responsiveness, stability, resource use, mission effectiveness).

### Scope (Phase 3) — high level

- Interface with existing control software and (where applicable) simulators or test rigs.
- Optimizations may include: control-loop tuning, sensor-fusion params, task scheduling, communication batching, power/thermal awareness.
- Safety and certification constraints must be explicit (read-only analysis first, or opt-in changes with rollback).

### Deliverables (Phase 3) — to detail later

- Compatibility spec for machine/control types.
- Progenitor extension for control software and/or configs.
- Measurable “peak” metrics (e.g. latency, tracking error, mission completion time).

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

You can add `software/` and `machines/` under `progenitor` or `optimizations` when you start Phase 2 and 3.

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
