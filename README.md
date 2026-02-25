# Progenitor

A system that enhances compatible targets to **peak performance** on **the hardware they already have** — ML algorithms first, then software, then physical machines. Defence-oriented; code-only (no biology).

Inspired by the Progenitor virus: one “strain,” selective compatibility, measurable enhancement. **Same device, enhanced to its peak** — no GPU required; built for legacy and edge (e.g. older military systems).

- **Plan:** [PLAN.md](./PLAN.md)
- **Phase 1 compatibility:** [docs/compatibility-phase1.md](./docs/compatibility-phase1.md)

## Phase 1: ML (ONNX)

**Install (editable):**

```bash
pip install -e .
```

**Enhance an ONNX model** (use a real path to your `.onnx` file):

```bash
# Example with the included tiny model:
python examples/create_tiny_onnx.py
progenitor enhance examples/tiny.onnx --target cpu --output examples/tiny_enhanced.onnx
```

**Benchmark before/after:**

```bash
python benchmarks/run.py path/to/model.onnx --target cpu
```

**Virus-level speedup (same CPU, no GPU):**  
Progenitor enhances the **same** machine. Two modes:

- **10–50x+ on same CPU:** `--max-speed` builds a tiny surrogate that approximates the model (real measured speedup, not hardcoded). Best for legacy/edge/military where GPU isn’t available. Validate accuracy.

```bash
progenitor enhance path/to/model.onnx --target cpu --max-speed -o path/to/model_surrogate.onnx
python benchmarks/run.py path/to/model.onnx --target cpu --max-speed --repeat 20
```

- **2–4x on same CPU:** `--quantize` (INT8), same device.

All timings are real; no hardcoded numbers.

**Python API:**

```python
from progenitor import enhance

result = enhance("model.onnx", "cpu")
if result.compatible:
    print("Enhanced:", result.output_path)
```

See [examples/README.md](./examples/README.md) for a tiny ONNX and more usage.

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

Use the included script to create a small multi-layer ONNX (3-layer MLP), then enhance and benchmark it. **Run step 1 first** so `small_mlp.onnx` exists. No PyTorch needed.

```bash
# 1. Create the small MLP ONNX (run this first)
python examples/export_small_model.py

# 2. Enhance it with Progenitor
progenitor enhance examples/small_mlp.onnx --target cpu -o examples/small_mlp_enhanced.onnx

# 3. Benchmark before/after (optionally with --live to see runs in real time)
python benchmarks/run.py examples/small_mlp.onnx --target cpu --repeat 50
python benchmarks/run.py examples/small_mlp.onnx --target cpu --repeat 50 --live
```

On this small MLP you should see a more consistent speedup than on `tiny.onnx`, since the graph has enough ops (linear, relu, etc.) for optimization to help.

### Commands to test on a **large** model (ResNet-50, ~97 MB)

Download a real large ONNX (ResNet-50), then enhance and benchmark:

```bash
# 1. Download ResNet-50 ONNX (~97 MB; run once)
python examples/download_large_model.py

# 2. Enhance it
progenitor enhance examples/resnet50.onnx --target cpu -o examples/resnet50_enhanced.onnx

# 3. Benchmark (use fewer repeats — each run is slower on a large model)
python benchmarks/run.py examples/resnet50.onnx --target cpu --repeat 20

# For drastic (virus-level) speedup, add --quantize: 2–4x faster
python benchmarks/run.py examples/resnet50.onnx --target cpu --quantize --repeat 20
python benchmarks/run.py examples/resnet50.onnx --target cpu --quantize --repeat 20 --live
```

This is a real large graph (conv layers, batch norm, etc.). Without `--quantize` you get ~1.05–1.15x from graph opts; **with `--quantize` (INT8) you typically get 2–4x** on CPU.