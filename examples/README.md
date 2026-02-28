# Progenitor examples (Phase 1: ML)

Full pass list and compatibility: [docs/compatibility-phase1.md](../docs/compatibility-phase1.md).

## Quick start

1. **Enhance an ONNX model** (CLI):

   ```bash
   progenitor enhance path/to/model.onnx --target cpu --output path/to/model_enhanced.onnx
   ```

2. **In Python:**

   ```python
   from progenitor import enhance

   result = enhance("model.onnx", "cpu", output_path="model_enhanced.onnx")
   if result.compatible:
       print("Enhanced:", result.output_path)
   ```

3. **Benchmark before/after** (optionally with accuracy validation):

   ```bash
   python benchmarks/run.py path/to/model.onnx --target cpu --repeat 100
   python benchmarks/run.py path/to/model.onnx --target cpu --max-speed --validate --repeat 50
   ```

   Use `--live` to stream each run; use `--validate` to report cosine similarity and MSE between baseline and enhanced outputs.

## Getting a small ONNX model

If you don't have an ONNX model yet:

- **PyTorch:** `torch.onnx.export(model, dummy_input, "model.onnx")`
- **Hugging Face:** Many models have ONNX weights or you can export with `optimum` (e.g. `optimum-cli export onnx --model ...`).
- **Samples:** You can create a minimal ONNX for testing with the script below (run from repo root):

```python
# examples/create_tiny_onnx.py
import numpy as np
from onnx import helper, TensorProto, save
from onnx import numpy_helper

# Minimal add/matmul graph
X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
node = helper.make_node("Add", ["x", "y"], ["z"])
graph = helper.make_graph([node], "tiny", [X, Y], [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
save(model, "tiny.onnx")
```

Then: `progenitor enhance tiny.onnx -o tiny_enhanced.onnx` and `python benchmarks/run.py tiny.onnx`.

### Testing on a larger model (small MLP)

To test Progenitor on a model with more than one op (so optimization can actually help):

```bash
python examples/export_small_model.py
progenitor enhance examples/small_mlp.onnx -o examples/small_mlp_enhanced.onnx
python benchmarks/run.py examples/small_mlp.onnx --target cpu --repeat 50
```

See the main [README](../README.md) for the full command list.

### Large model (ResNet-50)

To test on a **large** ONNX model (~97 MB, real conv/BatchNorm graph):

```bash
python examples/download_large_model.py
progenitor enhance examples/resnet50.onnx -o examples/resnet50_enhanced.onnx
python benchmarks/run.py examples/resnet50.onnx --target cpu --repeat 20
```

**Max-speed (CNN ~6–9×, cosine preserved):** Validation-guided conv prune + block removal + calibration + static INT8:

```bash
progenitor enhance examples/resnet50.onnx --target cpu --max-speed
python benchmarks/run.py examples/resnet50.onnx --target cpu --max-speed --validate --repeat 50
```

### Large MLP (~123× with sparse backend)

Create and run the large MLP example; use `--max-speed` for struct + lowrank + high sparsity. Full speedup requires a sparse-capable backend.

```bash
python examples/export_large_mlp.py
progenitor enhance examples/large_mlp.onnx --target cpu --max-speed
python benchmarks/run.py examples/large_mlp.onnx --target cpu --max-speed
```

---

**Why does tiny.onnx sometimes show *slower* "After"?**  
`tiny.onnx` is a single Add op. Full graph optimization adds a small fixed overhead. For one op that overhead can outweigh any benefit. Progenitor is meant for **larger** models (real networks with many layers), where those optimizations typically give a clear speedup.
