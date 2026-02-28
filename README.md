# Progenitor

A system that enhances compatible targets to **peak performance** on **the hardware they already have** — ML algorithms first, then software, then physical machines. Defence-oriented; code-only (no biology).

Inspired by the Progenitor virus: one “strain,” selective compatibility, measurable enhancement. **Same device, enhanced to its peak** — no GPU required; built for legacy and edge (e.g. older military systems).

- **Plan:** [PLAN.md](./PLAN.md)
- **Phase 1 compatibility (all passes, targets, validation):** [docs/compatibility-phase1.md](./docs/compatibility-phase1.md)

## Phase 1: ML (ONNX)

**Install (editable):**

```bash
pip install -e .
```

### Phase 1 workflow

1. **Get an ONNX model** — Export from PyTorch/TensorFlow, or use the examples (tiny, small MLP, ResNet-50).
2. **Enhance** — Run Progenitor with the target and (optionally) optimization flags. Use `--max-speed` for architecture-specific chained passes (CNN ~6–9×, large MLP ~123× with sparse backend).
3. **Benchmark** — Compare before/after latency and throughput on the same device.
4. **Validate** — Use `--validate` to check cosine similarity and MSE between baseline and enhanced outputs.

```bash
# Minimal: graph-only enhance
progenitor enhance path/to/model.onnx --target cpu -o path/to/model_enhanced.onnx
python benchmarks/run.py path/to/model.onnx --target cpu

# Max speed (architecture-specific: CNN gets conv prune + block removal + calibration + INT8)
progenitor enhance path/to/model.onnx --target cpu --max-speed
python benchmarks/run.py path/to/model.onnx --target cpu --max-speed --validate --repeat 50
```

Full list of passes and when they apply: [docs/compatibility-phase1.md](./docs/compatibility-phase1.md).

### Quick examples

**Tiny model (single op):**

```bash
python examples/create_tiny_onnx.py
progenitor enhance examples/tiny.onnx --target cpu --output examples/tiny_enhanced.onnx
python benchmarks/run.py examples/tiny.onnx --target cpu
```

**INT8 (2–4× on CPU):**

```bash
progenitor enhance path/to/model.onnx --target cpu --quantize -o path/to/model_quantized.onnx
python benchmarks/run.py path/to/model.onnx --target cpu --quantize --repeat 50
```

**Pruning (sparse backend for 5–15×):** Same graph, many weights zeroed. Use a sparse-capable backend for real speedup.

```bash
progenitor enhance path/to/model.onnx --prune 0.9 -o model_pruned.onnx
python benchmarks/run.py path/to/model.onnx --prune 0.9
```

**CNN max-speed (~6–9×, cosine preserved):** ResNet-style models get validation-guided conv prune + block removal + calibration + static INT8.

```bash
python examples/download_large_model.py   # if you don't have resnet50.onnx
progenitor enhance examples/resnet50.onnx --target cpu --max-speed
python benchmarks/run.py examples/resnet50.onnx --target cpu --max-speed --validate --repeat 50
```

**Large MLP (~123× with sparse backend):** `--max-speed` on large MLPs applies struct + lowrank + high sparsity; run with sparse backend for full speedup.

**Python API:**

```python
from progenitor import enhance

result = enhance("model.onnx", "cpu")
if result.compatible:
    print("Enhanced:", result.output_path)

# With options
result = enhance("model.onnx", "cpu", max_speed=True, output_path="out.onnx")
```

See [examples/README.md](./examples/README.md) for more usage and export scripts.

### How Progenitor works on larger models

**Same workflow.** You use the same commands; only the model file changes. Replace `path/to/your_model.onnx` with the **actual path** to your `.onnx` file on disk (e.g. `/Users/you/models/resnet50.onnx`):

```bash
progenitor enhance path/to/your_model.onnx --target cpu -o path/to/your_model_enhanced.onnx
python benchmarks/run.py path/to/your_model.onnx --target cpu   # optional: before/after numbers
```

**What happens under the hood:**

1. **Enhance:** Progenitor loads your ONNX, runs graph-level passes (e.g. shape inference), and saves an enhanced ONNX. That artifact is still standard ONNX; it’s tuned so that when you run it with ONNX Runtime and **full graph optimization** (fusion, constant folding, better kernels), the runtime can optimize it effectively.

2. **Run / deploy:** You run the **enhanced** model with ONNX Runtime and `graph_optimization_level=ORT_ENABLE_ALL` (our benchmark does this for “After”). In your own app, use the enhanced file and enable graph opts in `SessionOptions` so you get the same peak behavior.

**Why it helps on larger models:** Bigger graphs have more ops (convs, matmuls, attention, etc.). The runtime can then fuse chains (e.g. Conv+BN+ReLU into one kernel), fold constants, and pick faster kernels. The one-time cost of optimization is small compared to the gain. On a single-op toy model that cost can dominate; on real networks you typically see lower latency and higher throughput.

**Getting a larger ONNX:** Export from PyTorch (`torch.onnx.export`), TensorFlow, or use pre-exported models (e.g. [ONNX Model Zoo](https://github.com/onnx/models), Hugging Face with `optimum`). Then pass that `.onnx` path to `progenitor enhance` and use the enhanced file for inference.

### Commands to test on a larger model (from repo root)

Use the included script to create a small multi-layer ONNX (3-layer MLP), then enhance and benchmark. **Run step 1 first** so `small_mlp.onnx` exists. No PyTorch needed.

```bash
# 1. Create the small MLP ONNX (run this first)
python examples/export_small_model.py

# 2. Enhance and benchmark
progenitor enhance examples/small_mlp.onnx --target cpu -o examples/small_mlp_enhanced.onnx
python benchmarks/run.py examples/small_mlp.onnx --target cpu --repeat 50

# Optional: max-speed (per-layer tune / block sparse) and accuracy check
python benchmarks/run.py examples/small_mlp.onnx --target cpu --max-speed --validate --repeat 50
```

### Commands to test on a **large** model (ResNet-50, ~97 MB)

Download ResNet-50 ONNX, then enhance and benchmark:

```bash
# 1. Download ResNet-50 ONNX (~97 MB; run once)
python examples/download_large_model.py

# 2. Graph-only enhance
progenitor enhance examples/resnet50.onnx --target cpu -o examples/resnet50_enhanced.onnx
python benchmarks/run.py examples/resnet50.onnx --target cpu --repeat 20

# 3. Max-speed: validation-guided conv prune + block removal + calibration + INT8 (~6–9×, cosine preserved)
progenitor enhance examples/resnet50.onnx --target cpu --max-speed
python benchmarks/run.py examples/resnet50.onnx --target cpu --max-speed --validate --repeat 50

# 4. INT8 only (2–4×)
python benchmarks/run.py examples/resnet50.onnx --target cpu --quantize --repeat 20
```

ResNet is a real large graph (conv, batch norm). **`--max-speed`** gives the best CPU speedup with cosine preserved; see [docs/compatibility-phase1.md](./docs/compatibility-phase1.md) for the full pass list.